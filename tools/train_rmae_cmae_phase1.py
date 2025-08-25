# tools/train_rmae_cmae_phase1.py
"""
R-MAE + CMAE-3D Phase 1: Teacher-Student Training Script

기존 train_voxel_mae.py를 확장하여 Teacher-Student 기능 추가:
- 기존 R-MAE pretraining 완전 호환
- Teacher-Student architecture 지원
- 점진적 확장을 위한 기반 마련

Usage:
    # Phase 1 Pretraining
    python train_rmae_cmae_phase1.py \
        --cfg_file cfgs/custom_models/rmae_cmae_isarc_4class_pretraining_phase1.yaml \
        --batch_size 8 --epochs 30 --extra_tag rmae_cmae_pretraining_phase1
        
    # Phase 1 Fine-tuning은 기존 dist_train.sh 사용:
    # bash scripts/dist_train.sh 1 \
    #     --cfg_file cfgs/custom_models/rmae_cmae_isarc_4class_finetune_phase1.yaml \
    #     --pretrained_model [checkpoint_path] --batch_size 8 \
    #     --extra_tag rmae_cmae_finetune_phase1
"""

import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
import numpy as np
import tqdm

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import save_checkpoint, checkpoint_state


def parse_config():
    """🎯 Phase 1 argument parsing"""
    parser = argparse.ArgumentParser(description='R-MAE + CMAE-3D Phase 1 Training')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='phase1', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distributed training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def model_fn_decorator():
    """🔥 Phase 1 model function decorator"""
    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return loss, tb_dict, disp_dict

    return model_func


def load_data_to_gpu(batch_dict):
    """GPU로 데이터 로드"""
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def train_one_epoch_phase1(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                          rank, tbar, tb_log=None, leave_pbar=False, total_it_each_epoch=None, dataloader_iter=None):
    """🔥 Phase 1 한 epoch 훈련"""
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_func(model, batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        
        # 📍 Phase 1 특화 로깅
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})
        
        # Teacher-Student 상태 로깅
        if hasattr(model, 'module'):
            model_obj = model.module
        else:
            model_obj = model
            
        if hasattr(model_obj, 'enable_teacher_student'):
            disp_dict['ts_enabled'] = model_obj.enable_teacher_student
        
        if tb_log is not None:
            tb_log.add_scalar('train/loss', loss, accumulated_iter)
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
            
            # Phase 1 메트릭 로깅
            tb_log.add_scalar('phase1/total_loss', loss, accumulated_iter)
            for key, val in tb_dict.items():
                if 'phase1' in key or 'rmae' in key or 'teacher_student' in key:
                    tb_log.add_scalar(f'phase1/{key}', val, accumulated_iter)
                else:
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)

        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model_phase1(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                      start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir,
                      train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                      max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False):
    """🔥 Phase 1 모델 훈련"""
    accumulated_iter = start_iter
    
    # Phase 1 특화 정보 출력
    print("🔥 Phase 1 Training Started:")
    if hasattr(model, 'module'):
        model_obj = model.module
    else:
        model_obj = model
        
    if hasattr(model_obj, 'enable_teacher_student'):
        print(f"   - Teacher-Student enabled: {model_obj.enable_teacher_student}")
    
    if hasattr(cfg, 'LOSS_CONFIG'):
        print(f"   - R-MAE weight: {cfg.LOSS_CONFIG.get('RMAE_WEIGHT', 1.0)}")
        print(f"   - Teacher-Student weight: {cfg.LOSS_CONFIG.get('TEACHER_STUDENT_WEIGHT', 0.0)}")
    
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

        dataloader_iter = iter(train_loader)
        
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # Scheduler 선택
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            # 한 epoch 훈련
            accumulated_iter = train_one_epoch_phase1(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg, rank=rank, tbar=tbar,
                tb_log=tb_log, leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch, dataloader_iter=dataloader_iter
            )

            # Checkpoint 저장
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), 
                    filename=ckpt_name,
                )

    logger.info('🔥 Phase 1 Training Finished')


def main():
    """🔥 Phase 1 메인 함수"""
    args, cfg = parse_config()
    
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_save_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # Phase 1 정보 로깅
    logger.info('🔥 R-MAE + CMAE-3D Phase 1 Training')
    logger.info('=' * 80)
    log_config_to_file(cfg, logger=logger)
    logger.info('=' * 80)

    # Dataset 구축
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    # Model 구축
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # Optimizer 구축 (차등 학습률 지원)
    if cfg.OPTIMIZATION.get('DIFFERENTIAL_LR', False):
        from train_utils.optimization.differential_lr import create_optimizer_with_differential_lr
        optimizer = create_optimizer_with_differential_lr(model, cfg.OPTIMIZATION)
    else:
        optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # Checkpoint 로드
    start_epoch = 0
    model.train()
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=False, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=False, optimizer=optimizer, logger=logger)
    else:
        ckpt_list = glob.glob(str(ckpt_save_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=False, optimizer=optimizer, logger=logger
            )

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    # Learning rate scheduler
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=start_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # Tensorboard
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    else:
        tb_log = None

    # 🔥 Phase 1 Training 실행
    logger.info('🔥 Phase 1 Training Starting...')
    
    train_model_phase1(
        model,
        optimizer,
        train_loader,
        model_fn_decorator(),
        lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=0,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_save_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
    )

    logger.info('🔥 Phase 1 Training Complete!')


if __name__ == '__main__':
    main()