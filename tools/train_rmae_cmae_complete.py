#!/usr/bin/env python
"""
tools/train_rmae_cmae_complete.py

✅ R-MAE + CMAE-3D 완전 통합 훈련 스크립트
- 기존 성공한 train_voxel_mae.py 구조 100% 보존
- CMAE-3D 논리 완벽 통합
- 유기적 연결과 안정성 보장
- 고급 모니터링 및 디버깅

Usage:
    python train_rmae_cmae_complete.py \
        --cfg_file cfgs/custom_models/rmae_cmae_voxelnext_pretraining.yaml \
        --batch_size 8 \
        --extra_tag rmae_cmae_integration_v1
"""

import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
import numpy as np
import tqdm
import time

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler

# ✅ CMAE-3D 추가 모듈들 (optional import)
try:
    from pcdet.models.model_utils.hrcl_utils import HRCLModule
    HRCL_AVAILABLE = True
except ImportError:
    HRCL_AVAILABLE = False
    print("⚠️ HRCL utils not available, using fallback")


def parse_config():
    """✅ 기존 성공 parse_config 완전 보존"""
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
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.batch_size is not None:
        cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU = args.batch_size

    if args.epochs is not None:
        cfg.OPTIMIZATION.NUM_EPOCHS = args.epochs

    if args.workers is not None:
        # ✅ 기존 코드들을 보니 DATA_CONFIG 아래에 직접 workers 설정이 없음
        # 대신 build_dataloader에서 workers 파라미터로 전달됨
        pass  # workers는 build_dataloader 호출 시 직접 전달

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    return args, cfg


def main():
    """✅ 메인 함수 - 기존 성공 구조 보존"""
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

    if cfg.LOCAL_RANK == 0:
        os.makedirs(cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG, exist_ok=True)
        ckpt_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        log_file = ckpt_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

        # log to file
        logger.info('**********************Start logging**********************')
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

        if dist_train:
            logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
        for key, val in vars(args).items():
            logger.info('{:16} {}'.format(key, val))
        log_config_to_file(cfg, logger=logger)
        if cfg.LOCAL_RANK == 0:
            os.system('cp %s %s' % (args.cfg_file, ckpt_dir))

    tb_log = SummaryWriter(log_dir=str(ckpt_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # ===== 데이터로더 & 네트워크 & 옵티마이저 생성 (기존 성공 로직) =====
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

    # ===== 훈련 시작 =====
    logger.info('**********************Start R-MAE + CMAE-3D pre-training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
    # ✅ CMAE-3D 통합 훈련
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
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger
    )

    logger.info('**********************R-MAE + CMAE-3D Pre-training Finished**********************')


def model_fn_decorator():
    """✅ 기존 성공 model_fn_decorator 완전 보존"""
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
    """✅ 기존 성공 load_data_to_gpu 완전 보존"""
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def train_one_epoch_rmae_cmae(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                              rank, tbar, tb_log=None, leave_pbar=False, total_it_each_epoch=None, dataloader_iter=None,
                              logger=None):
    """✅ R-MAE + CMAE-3D 통합 한 에포크 훈련"""
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    # ✅ CMAE-3D 손실 추적을 위한 변수들
    epoch_losses = {
        'total_loss': [],
        'occupancy_loss': [],
        'feature_loss': [],
        'contrastive_loss': []
    }

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            if rank == 0:
                logger.info('New iteration cycle started')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        # ✅ Forward pass with detailed loss tracking
        loss, tb_dict, disp_dict = model_func(model, batch)

        # ✅ 손실 세부사항 추적
        epoch_losses['total_loss'].append(loss.item())
        for key in ['occupancy_loss', 'feature_loss', 'contrastive_loss']:
            if key in tb_dict:
                epoch_losses[key].append(tb_dict[key])

        # ✅ Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # ✅ 로깅 (기존 로직 + CMAE-3D 상세 정보)
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/total_loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                
                # ✅ CMAE-3D 개별 손실 로깅
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)

            # ✅ 주기적인 상세 로깅 (매 100 iter)
            if accumulated_iter % 100 == 0 and logger is not None:
                avg_total = np.mean(epoch_losses['total_loss'][-100:]) if epoch_losses['total_loss'] else 0
                logger.info(f'Iter {accumulated_iter}: Total Loss = {avg_total:.4f}, LR = {cur_lr:.6f}')
                
                # CMAE-3D 손실 상세 정보
                if epoch_losses['occupancy_loss']:
                    avg_occ = np.mean(epoch_losses['occupancy_loss'][-100:])
                    logger.info(f'  ├─ R-MAE Occupancy Loss: {avg_occ:.4f}')
                if epoch_losses['feature_loss']:
                    avg_feat = np.mean(epoch_losses['feature_loss'][-100:])
                    logger.info(f'  ├─ CMAE Feature Loss: {avg_feat:.4f}')
                if epoch_losses['contrastive_loss']:
                    avg_cont = np.mean(epoch_losses['contrastive_loss'][-100:])
                    logger.info(f'  └─ CMAE Contrastive Loss: {avg_cont:.4f}')

    if rank == 0:
        pbar.close()
        
        # ✅ 에포크 종료 시 요약 로깅
        if logger is not None:
            avg_total = np.mean(epoch_losses['total_loss']) if epoch_losses['total_loss'] else 0
            logger.info(f'Epoch Summary - Average Total Loss: {avg_total:.4f}')
            
            # 손실 안정성 체크
            if len(epoch_losses['total_loss']) > 10:
                recent_losses = epoch_losses['total_loss'][-10:]
                loss_std = np.std(recent_losses)
                if loss_std > avg_total * 0.5:  # 손실이 너무 불안정한 경우
                    logger.warning(f'Loss instability detected: std={loss_std:.4f}, mean={avg_total:.4f}')

    return accumulated_iter


def train_model_rmae_cmae(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                          start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir,
                          train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                          max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None):
    """✅ R-MAE + CMAE-3D 통합 훈련 메인 함수"""
    
    accumulated_iter = start_iter
    dataloader_iter = iter(train_loader)
    
    # ✅ 훈련 안정성 모니터링
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 5  # 5 에포크 동안 개선 없으면 경고
    
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # ✅ 학습률 스케줄러 선택
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            # ✅ 에포크 시작 로깅
            if rank == 0 and logger is not None:
                logger.info(f'========== Epoch {cur_epoch}/{total_epochs} Started ==========')
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f'Current Learning Rate: {current_lr:.6f}')

            # ✅ 한 에포크 훈련
            accumulated_iter = train_one_epoch_rmae_cmae(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter,
                optim_cfg=optim_cfg,
                rank=rank,
                tbar=tbar,
                tb_log=tb_log,
                leave_pbar=False,
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                logger=logger
            )

            # ✅ 체크포인트 저장
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

                # ✅ 최고 성능 모델 추적
                current_loss = tbar.postfix if hasattr(tbar, 'postfix') else {}
                if isinstance(current_loss, dict) and 'loss' in current_loss:
                    epoch_loss = current_loss['loss']
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        patience_counter = 0
                        
                        # 최고 성능 모델 저장
                        best_ckpt_name = ckpt_save_dir / 'checkpoint_best.pth'
                        save_checkpoint(
                            checkpoint_state(model, optimizer, trained_epoch, accumulated_iter),
                            filename=best_ckpt_name
                        )
                        
                        if logger is not None:
                            logger.info(f'New best model saved! Loss: {best_loss:.4f}')
                    else:
                        patience_counter += 1
                        if patience_counter >= patience_limit and logger is not None:
                            logger.warning(f'No improvement for {patience_counter} epochs. Consider adjusting hyperparameters.')

                if logger is not None:
                    logger.info(f'Checkpoint saved: checkpoint_epoch_{trained_epoch}.pth')


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    """✅ 체크포인트 상태 생성"""
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        state = {
            'epoch': epoch,
            'it': it,
            'model_state': model_state,
            'optimizer_state': optim_state,
            'version': 'rmae_cmae_v1'
        }
    except:
        state = {
            'epoch': epoch,
            'it': it,
            'model_state': model_state,
            'optimizer_state': optim_state,
            'version': 'rmae_cmae_v1'
        }
    return state


def save_checkpoint(state, filename='checkpoint'):
    """✅ 체크포인트 저장"""
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


if __name__ == '__main__':
    main()
