#!/usr/bin/env python
"""
tools/train_rmae_cmae.py

기존 성공한 train_voxel_mae.py를 기반으로 CMAE 요소를 점진적으로 추가
- 기존 R-MAE 성공 훈련 로직 100% 유지
- CMAE 로깅 및 모니터링 안전하게 추가

Usage:
    python train_rmae_cmae.py \
        --cfg_file cfgs/custom_models/rmae_cmae_voxelnext_pretraining.yaml \
        --batch_size 4 \
        --extra_tag rmae_cmae_integration
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
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler


def parse_config():
    """✅ 기존 성공 로직 그대로"""
    parser = argparse.ArgumentParser(description='R-MAE + CMAE-3D Training')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='rmae_cmae', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
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
    """✅ 기존 성공 로직에 CMAE 로깅 추가"""
    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        
        # ➕ CMAE 손실 컴포넌트 로깅 추가
        if isinstance(tb_dict, dict):
            # CMAE 관련 손실들을 따로 표시
            cmae_losses = {k: v for k, v in tb_dict.items() if 'cmae' in k.lower()}
            rmae_losses = {k: v for k, v in tb_dict.items() if 'rmae' in k.lower()}
            
            if cmae_losses:
                print(f"   ➕ CMAE Losses: {cmae_losses}")
            if rmae_losses:
                print(f"   ✅ R-MAE Losses: {rmae_losses}")

        return loss, tb_dict, disp_dict

    return model_func


def load_data_to_gpu(batch_dict):
    """✅ 기존 성공 로직 그대로"""
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def train_one_epoch_rmae_cmae(model, optimizer, train_loader, model_func, lr_scheduler, 
                              optim_cfg, rank, tbar, accumulated_iter, tb_log):
    """✅ 기존 성공한 train_one_epoch_mae 로직에 CMAE 모니터링 추가"""
    if rank == 0:
        pbar = tqdm.tqdm(total=len(train_loader), leave=False, desc='train', dynamic_ncols=True)

    for cur_it, batch in enumerate(train_loader):
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

        # ✅ 기존 성공 로직: NaN/Inf 체크
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN/Inf loss detected at iter {accumulated_iter}, skipping...")
            accumulated_iter += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # ➕ CMAE 강화된 로깅
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                
                # ➕ CMAE 컴포넌트별 로깅
                for key, val in tb_dict.items():
                    if 'cmae' in key.lower():
                        tb_log.add_scalar('train/cmae/' + key, val, accumulated_iter)
                    elif 'rmae' in key.lower():
                        tb_log.add_scalar('train/rmae/' + key, val, accumulated_iter)
                    else:
                        tb_log.add_scalar('train/' + key, val, accumulated_iter)
                
                # ➕ 추가 통계
                if accumulated_iter % 100 == 0:
                    print(f"🎯 Iter {accumulated_iter}: Total Loss = {loss.item():.4f}")
                    if 'rmae_occupancy' in tb_dict:
                        print(f"   ✅ R-MAE Occupancy: {tb_dict['rmae_occupancy']:.4f}")
                    if 'cmae_contrastive' in tb_dict:
                        print(f"   ➕ CMAE Contrastive: {tb_dict['cmae_contrastive']:.4f}")
                    if 'cmae_feature' in tb_dict:
                        print(f"   ➕ CMAE Feature: {tb_dict['cmae_feature']:.4f}")

    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model_rmae_cmae(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                          start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir,
                          train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                          max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False):
    """✅ 기존 성공한 train_model_mae 로직에 CMAE 모니터링 추가"""
    
    accumulated_iter = start_iter
    
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # ✅ 기존 성공 로직: warmup scheduler
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            # ➕ Epoch 시작 로깅 강화
            if rank == 0:
                print(f"\n🚀 Epoch {cur_epoch}/{total_epochs} 시작")
                print(f"   Current LR: {optimizer.param_groups[0]['lr']:.6f}")

            # ✅ 기존 성공 로직 + CMAE 모니터링
            accumulated_iter = train_one_epoch_rmae_cmae(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                optim_cfg=optim_cfg, rank=rank, tbar=tbar, 
                accumulated_iter=accumulated_iter, tb_log=tb_log
            )

            # ✅ 기존 성공 로직: checkpoint 저장
            if rank == 0 and (cur_epoch % ckpt_save_interval == 0 or cur_epoch == total_epochs - 1):
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % cur_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), 
                    filename=ckpt_name,
                )
                
                # ➕ CMAE 체크포인트 정보
                print(f"✅ Checkpoint saved: {ckpt_name}")
                print(f"   R-MAE + CMAE-3D integrated model")
                print(f"   Epoch: {cur_epoch}, Iter: {accumulated_iter}")


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    """✅ 기존 성공 로직 그대로"""
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename='checkpoint'):
    """✅ 기존 성공 로직 그대로"""
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def main():
    """✅ 기존 성공 로직에 CMAE 설정 추가"""
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

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # ➕ CMAE 실험 정보 로깅
    logger.info('**********************R-MAE + CMAE-3D Integration**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    
    log_config_to_file(cfg, logger=logger)
    
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # ✅ 기존 성공 로직: dataloader & network & optimizer
    logger.info('**********************Building Dataset**********************')
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=args.epochs
    )

    logger.info('**********************Building Network**********************')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    start_epoch = it = 0
    last_epoch = -1
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1

    model.train()
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # ✅ 기존 성공 로직으로 훈련 시작
    logger.info('**********************Start R-MAE + CMAE-3D Training**********************')
    logger.info('R-MAE radial masking + CMAE-3D contrastive learning integration')
    
    train_model_rmae_cmae(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
    )

    logger.info('**********************R-MAE + CMAE-3D Training Finished**********************')


if __name__ == '__main__':
    main()