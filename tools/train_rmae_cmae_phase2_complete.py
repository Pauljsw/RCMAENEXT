#!/usr/bin/env python
"""
tools/train_rmae_cmae_phase2_complete.py

🔥 CMAE-3D Phase 2 완전 통합 훈련 스크립트
- Multi-scale Latent Feature Reconstruction (MLFR)
- Voxel-level Contrastive Learning  
- Frame-level Contrastive Learning
- 모든 기능 통합 및 최적화

Usage:
    # Phase 2 완전체 Pretraining
    python train_rmae_cmae_phase2_complete.py \
        --cfg_file cfgs/custom_models/rmae_cmae_isarc_4class_pretraining_phase2.yaml \
        --batch_size 8 --epochs 30 --extra_tag rmae_cmae_phase2_complete
        
    # Phase 2 Fine-tuning은 기존 dist_train.sh 사용
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
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import save_checkpoint, checkpoint_state


class Phase2TrainingMonitor:
    """🔍 Phase 2 전용 training monitor"""
    
    def __init__(self):
        self.loss_history = {
            'total': [],
            'rmae': [],
            'mlfr': [],
            'voxel_contrastive': [],
            'frame_contrastive': []
        }
        self.epoch_stats = []
        self.best_loss = float('inf')
        self.best_epoch = 0
        
    def update(self, epoch, losses):
        """Loss 업데이트 및 모니터링"""
        total_loss = losses.get('total_loss', 0)
        
        self.loss_history['total'].append(total_loss)
        self.loss_history['rmae'].append(losses.get('rmae_loss', 0))
        self.loss_history['mlfr'].append(losses.get('mlfr_loss', 0))
        self.loss_history['voxel_contrastive'].append(losses.get('voxel_contrastive_loss', 0))
        self.loss_history['frame_contrastive'].append(losses.get('frame_contrastive_loss', 0))
        
        # Best model tracking
        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.best_epoch = epoch
            
        # Epoch statistics
        epoch_stat = {
            'epoch': epoch,
            'total_loss': total_loss,
            'rmae_loss': losses.get('rmae_loss', 0),
            'mlfr_loss': losses.get('mlfr_loss', 0),
            'voxel_contrastive_loss': losses.get('voxel_contrastive_loss', 0),
            'frame_contrastive_loss': losses.get('frame_contrastive_loss', 0),
            'mlfr_enabled': losses.get('mlfr_enabled', False),
            'voxel_contrastive_enabled': losses.get('voxel_contrastive_enabled', False),
            'frame_contrastive_enabled': losses.get('frame_contrastive_enabled', False)
        }
        self.epoch_stats.append(epoch_stat)
    
    def print_summary(self):
        """Phase 2 훈련 요약 출력"""
        print("=" * 80)
        print("🚀 Phase 2 Complete Training Summary")
        print("=" * 80)
        print(f"Best Total Loss: {self.best_loss:.6f} (Epoch {self.best_epoch})")
        
        if len(self.loss_history['total']) > 0:
            final_losses = {
                'Total': self.loss_history['total'][-1],
                'R-MAE': self.loss_history['rmae'][-1], 
                'MLFR': self.loss_history['mlfr'][-1],
                'Voxel Contrastive': self.loss_history['voxel_contrastive'][-1],
                'Frame Contrastive': self.loss_history['frame_contrastive'][-1]
            }
            
            print("\n📊 Final Loss Breakdown:")
            for name, value in final_losses.items():
                print(f"   - {name}: {value:.6f}")
        
        print("\n🎯 Phase 2 Features Verified:")
        if len(self.epoch_stats) > 0:
            final_stat = self.epoch_stats[-1]
            print(f"   - MLFR: {'✅' if final_stat['mlfr_enabled'] else '❌'}")
            print(f"   - Voxel Contrastive: {'✅' if final_stat['voxel_contrastive_enabled'] else '❌'}")
            print(f"   - Frame Contrastive: {'✅' if final_stat['frame_contrastive_enabled'] else '❌'}")
        
        print("=" * 80)


def parse_config():
    """Phase 2 argument parsing"""
    parser = argparse.ArgumentParser(description='CMAE-3D Phase 2 Complete Training')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='phase2_complete', help='extra tag')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='checkpoint save interval')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max checkpoint save number')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
        
    return args, cfg


def model_fn_decorator():
    """Model function for Phase 2 training"""
    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)
        
        loss = ret_dict['loss'].mean()
        
        # Phase 2 specific logging
        if 'phase2_step3_active' in disp_dict:
            print(f"🔥 Phase 2 Step 3 Active - Total Loss: {loss.item():.6f}")
            
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()
            
        return loss, tb_dict, disp_dict
    
    return model_func


def load_data_to_gpu(batch_dict):
    """GPU 데이터 로딩"""
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def train_one_epoch_phase2(model, optimizer, train_loader, model_func, lr_scheduler, 
                          accumulated_iter, optim_cfg, rank, tbar, tb_log=None, 
                          leave_pbar=False, total_it_each_epoch=None, dataloader_iter=None,
                          monitor=None):
    """Phase 2 전용 one epoch training"""
    
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
        
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
    
    epoch_losses = {'total_loss': 0, 'rmae_loss': 0, 'mlfr_loss': 0, 'voxel_contrastive_loss': 0, 'frame_contrastive_loss': 0}
    num_batches = 0
    
    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            
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
        
        # Phase 2 loss 누적
        epoch_losses['total_loss'] += loss.item()
        epoch_losses['rmae_loss'] += disp_dict.get('rmae_loss', 0)
        epoch_losses['mlfr_loss'] += disp_dict.get('mlfr_loss', 0)
        epoch_losses['voxel_contrastive_loss'] += disp_dict.get('voxel_contrastive_loss', 0)
        epoch_losses['frame_contrastive_loss'] += disp_dict.get('frame_contrastive_loss', 0)
        num_batches += 1
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        
        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})
        
        # Console & tensorboard logging
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()
            
            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    
    # Epoch average losses 계산
    if num_batches > 0:
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        # Monitor 업데이트
        if monitor is not None:
            epoch_losses.update({
                'mlfr_enabled': disp_dict.get('mlfr_enabled', False),
                'voxel_contrastive_enabled': disp_dict.get('voxel_contrastive_enabled', False),
                'frame_contrastive_enabled': disp_dict.get('frame_contrastive_enabled', False)
            })
    
    if rank == 0:
        pbar.close()
        
    return accumulated_iter, epoch_losses


def train_model_phase2(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                      start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir,
                      train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                      max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False):
    """Phase 2 완전체 training loop"""
    
    accumulated_iter = start_iter
    dataloader_iter = iter(train_loader)
    monitor = Phase2TrainingMonitor()
    
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)
                
            # Learning rate scheduler
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
                
            # Train one epoch with Phase 2 monitoring
            accumulated_iter, epoch_losses = train_one_epoch_phase2(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter,
                optim_cfg=optim_cfg,
                rank=rank,
                tbar=tbar,
                tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                monitor=monitor
            )
            
            # Monitor 업데이트
            monitor.update(cur_epoch, epoch_losses)
            
            # Checkpoint saving
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
                
    # 훈련 완료 후 요약 출력
    if rank == 0:
        monitor.print_summary()


def main():
    """Phase 2 완전체 메인 실행"""
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
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    
    # Phase 2 시작 로그
    logger.info('**********************🚀 CMAE-3D Phase 2 Complete Training Start 🚀**********************')
    logger.info('Phase 2 Features:')
    logger.info('  - ✅ Multi-scale Latent Feature Reconstruction (MLFR)')
    logger.info('  - ✅ Voxel-level Contrastive Learning')  
    logger.info('  - ✅ Frame-level Contrastive Learning')
    logger.info('  - 🎯 Expected improvement: +3~5 mAP')
    
    log_config_to_file(cfg, logger=logger)
    if dist_train:
        logger.info('Training in distributed mode : total_gpus: %d, local_rank: %d' % (total_gpus, cfg.LOCAL_RANK))
        
    logger.info('Loading data from config: %s' % (args.cfg_file))
    
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None
    
    # Build dataloader & network
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
    
    # Phase 2 완전체 훈련 시작
    logger.info('**********************Start Phase 2 Complete pretraining %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
    train_model_phase2(
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
    
    logger.info('**********************🎉 Phase 2 Complete pretraining is finished 🎉**********************')


if __name__ == '__main__':
    main()