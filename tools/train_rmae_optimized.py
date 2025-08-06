# tools/train_rmae_optimized.py
"""
최적화된 R-MAE VoxelNeXt 훈련 스크립트

기존 train_voxel_mae.py를 기반으로 성능 최적화:
1. Enhanced monitoring and logging
2. Progressive training support
3. Advanced checkpoint management
4. Real-time performance tracking
5. Automatic hyperparameter adjustment

사용법:
python train_rmae_optimized.py \
    --cfg_file cfgs/custom_models/rmae_voxelnext_optimized_pretraining.yaml \
    --batch_size 6 \
    --epochs 35 \
    --extra_tag rmae_optimized_v1
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
import json

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler


class OptimizedTrainingManager:
    """🚀 최적화된 훈련 관리자"""
    
    def __init__(self, cfg, logger, tb_log, output_dir):
        self.cfg = cfg
        self.logger = logger
        self.tb_log = tb_log
        self.output_dir = Path(output_dir)
        
        # 성능 추적
        self.training_stats = {
            'epoch_losses': [],
            'mask_ratios': [],
            'voxel_counts': [],
            'learning_rates': [],
            'training_times': []
        }
        
        # Best model tracking
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.max_patience = cfg.OPTIMIZATION.get('PATIENCE', 10)
        
        # Progressive training 설정
        self.progressive_training = cfg.MODEL.get('PROGRESSIVE_TRAINING', False)
        self.warmup_epochs = cfg.MODEL.get('WARMUP_EPOCHS', 5)
        
        print(f"🚀 Optimized Training Manager initialized")
        print(f"   📊 Progressive training: {self.progressive_training}")
        print(f"   🎯 Max patience: {self.max_patience}")
        print(f"   📈 Warmup epochs: {self.warmup_epochs}")
    
    def log_epoch_start(self, epoch):
        """Epoch 시작 로깅"""
        self.logger.info(f"🚀 Starting Epoch {epoch}/{self.cfg.OPTIMIZATION.NUM_EPOCHS}")
        self.logger.info(f"   📊 Current patience: {self.patience_counter}/{self.max_patience}")
        self.logger.info(f"   🎯 Best loss so far: {self.best_loss:.6f} (Epoch {self.best_epoch})")
    
    def log_epoch_end(self, epoch, epoch_loss, lr, epoch_time):
        """Epoch 종료 로깅 및 통계 업데이트"""
        # 통계 업데이트
        self.training_stats['epoch_losses'].append(epoch_loss)
        self.training_stats['learning_rates'].append(lr)
        self.training_stats['training_times'].append(epoch_time)
        
        # Best model 체크
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_epoch = epoch
            self.patience_counter = 0
            is_best = True
        else:
            self.patience_counter += 1
            is_best = False
        
        # 로깅
        self.logger.info(f"✅ Epoch {epoch} completed:")
        self.logger.info(f"   📊 Loss: {epoch_loss:.6f} {'🎯 NEW BEST!' if is_best else ''}")
        self.logger.info(f"   📈 Learning Rate: {lr:.8f}")
        self.logger.info(f"   ⏱️  Time: {epoch_time:.2f}s")
        self.logger.info(f"   🎯 Patience: {self.patience_counter}/{self.max_patience}")
        
        # TensorBoard 로깅
        if self.tb_log:
            self.tb_log.add_scalar('epoch/loss', epoch_loss, epoch)
            self.tb_log.add_scalar('epoch/learning_rate', lr, epoch)
            self.tb_log.add_scalar('epoch/training_time', epoch_time, epoch)
            self.tb_log.add_scalar('epoch/patience', self.patience_counter, epoch)
        
        # 통계 분석
        if len(self.training_stats['epoch_losses']) >= 5:
            recent_losses = self.training_stats['epoch_losses'][-5:]
            loss_trend = (recent_losses[-1] - recent_losses[0]) / 5
            self.logger.info(f"   📊 Recent trend: {loss_trend:+.6f}/epoch")
            
            if self.tb_log:
                self.tb_log.add_scalar('epoch/loss_trend', loss_trend, epoch)
        
        return is_best
    
    def should_early_stop(self):
        """Early stopping 체크"""
        return self.patience_counter >= self.max_patience
    
    def save_training_stats(self):
        """훈련 통계 저장"""
        stats_file = self.output_dir / 'training_stats.json'
        
        # NumPy arrays를 list로 변환
        serializable_stats = {}
        for key, value in self.training_stats.items():
            if isinstance(value, list):
                serializable_stats[key] = value
            elif hasattr(value, 'tolist'):
                serializable_stats[key] = value.tolist()
            else:
                serializable_stats[key] = value
        
        # 추가 메타데이터
        serializable_stats.update({
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.training_stats['epoch_losses']),
            'final_patience': self.patience_counter
        })
        
        with open(stats_file, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"📊 Training statistics saved to {stats_file}")


def parse_config():
    """설정 파싱"""
    parser = argparse.ArgumentParser(description='Optimized R-MAE Training')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='rmae_optimized', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting time')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='')
    parser.add_argument('--max_ckpt_save_num', type=int, default=50, help='')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='')
    parser.add_argument('--pretrained_model', type=str, default=None, help='')
        
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def train_one_epoch_optimized(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, 
                             optim_cfg, rank, tbar, tb_log=None, leave_pbar=False, total_it_each_epoch=None, 
                             dataloader_iter=None, training_manager=None, epoch=None):
    """🚀 최적화된 한 에포크 훈련"""
    
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    # Epoch별 통계
    epoch_losses = []
    epoch_mask_ratios = []
    epoch_voxel_counts = []
    
    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            if rank == 0:
                print('New iteration cycle started')

        data_time.update(time.time() - end)

        # Progressive masking ratio 업데이트 (model에 epoch 정보 전달)
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch, dataloader_len=total_it_each_epoch)
        elif hasattr(model, 'module') and hasattr(model.module, 'set_epoch'):
            model.module.set_epoch(epoch, dataloader_len=total_it_each_epoch)

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()
        
        forward_start = time.time()
        loss, tb_dict, disp_dict = model_func(model, batch)
        forward_time.update(time.time() - forward_start)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        
        # 통계 수집
        if 'total_loss' in tb_dict:
            epoch_losses.append(tb_dict['total_loss'])
        if 'mask_ratio' in tb_dict:
            epoch_mask_ratios.append(tb_dict['mask_ratio'])
        if 'voxel_count' in tb_dict:
            epoch_voxel_counts.append(tb_dict['voxel_count'])

        # Logging
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})
        batch_time.update(time.time() - end)

        # Print progress
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                for key, val in tb_dict.items():
                    tb_log.add_scalar(f'train/{key}', val, accumulated_iter)

    if rank == 0:
        pbar.close()
        
        # Epoch 통계 계산
        epoch_stats = {
            'avg_loss': np.mean(epoch_losses) if epoch_losses else 0.0,
            'avg_mask_ratio': np.mean(epoch_mask_ratios) if epoch_mask_ratios else 0.0,
            'avg_voxel_count': np.mean(epoch_voxel_counts) if epoch_voxel_counts else 0.0,
            'lr': cur_lr
        }
        
        # Training manager에 통계 전달
        if training_manager:
            training_manager.training_stats['mask_ratios'].extend(epoch_mask_ratios)
            training_manager.training_stats['voxel_counts'].extend(epoch_voxel_counts)
        
        return accumulated_iter, epoch_stats
    
    return accumulated_iter, {}


def train_model_optimized(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg, 
                         start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, 
                         train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                         max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None):
    """🚀 최적화된 전체 훈련 루프"""
    
    accumulated_iter = start_iter
    
    # Training manager 초기화
    training_manager = OptimizedTrainingManager(
        cfg=cfg, logger=logger, tb_log=tb_log, output_dir=ckpt_save_dir
    )
    
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # LR warmup
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            # Epoch 시작 로깅
            if rank == 0:
                training_manager.log_epoch_start(cur_epoch)

            epoch_start_time = time.time()
            
            # 한 에포크 훈련
            accumulated_iter, epoch_stats = train_one_epoch_optimized(
                model, optimizer, train_loader, model_func, cur_scheduler, accumulated_iter, optim_cfg,
                rank, tbar, tb_log=tb_log, leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch, dataloader_iter=dataloader_iter,
                training_manager=training_manager, epoch=cur_epoch
            )
            
            epoch_time = time.time() - epoch_start_time

            # Epoch 종료 처리 (rank 0에서만)
            if rank == 0:
                # Epoch 통계 로깅
                is_best = training_manager.log_epoch_end(
                    cur_epoch, epoch_stats['avg_loss'], epoch_stats['lr'], epoch_time
                )

                # 체크포인트 저장
                trained_epoch = cur_epoch + 1
                if trained_epoch % ckpt_save_interval == 0:
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
                
                # Best model 저장
                if is_best:
                    best_ckpt_name = ckpt_save_dir / 'checkpoint_best'
                    save_checkpoint(
                        checkpoint_state(model, optimizer, trained_epoch, accumulated_iter),
                        filename=best_ckpt_name,
                    )
                    logger.info(f"🎯 Best model saved at epoch {trained_epoch}")

                # Early stopping 체크
                if training_manager.should_early_stop():
                    logger.info(f"⏹️  Early stopping triggered at epoch {cur_epoch}")
                    logger.info(f"   📊 Best loss: {training_manager.best_loss:.6f} at epoch {training_manager.best_epoch}")
                    break

    # 훈련 완료 처리
    if rank == 0:
        # 최종 통계 저장
        training_manager.save_training_stats()
        
        # 훈련 요약
        logger.info("🎉 Training completed!")
        logger.info(f"   📊 Best loss: {training_manager.best_loss:.6f} at epoch {training_manager.best_epoch}")
        logger.info(f"   ⏱️  Total training time: {sum(training_manager.training_stats['training_times']):.2f}s")
        
        # 최종 모델 저장
        final_ckpt_name = ckpt_save_dir / 'checkpoint_final'
        save_checkpoint(
            checkpoint_state(model, optimizer, total_epochs, accumulated_iter),
            filename=final_ckpt_name,
        )


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    """체크포인트 state 생성"""
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {
        'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version
    }


def save_checkpoint(state, filename='checkpoint'):
    """체크포인트 저장"""
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def model_fn_decorator():
    """Model function decorator"""
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
    """데이터를 GPU로 로드"""
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def main():
    """메인 함수"""
    args, _ = parse_config()
    
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

    if getattr(args, 'fix_random_seed', False):
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # 설정 정보 로깅
    logger.info('🚀 Optimized R-MAE Training Started')
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # TensorBoard
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    else:
        tb_log = None

    # 데이터셋 및 데이터로더
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

    # 모델 생성
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # Optimizer 및 Scheduler
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    
    # 체크포인트에서 시작
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])

    # Scheduler
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # 훈련 시작
    logger.info('🎯 Start training (Optimized R-MAE)')
    train_model_optimized(
        model,
        optimizer,
        train_loader,
        model_fn_decorator(),
        lr_scheduler,
        cfg.OPTIMIZATION,
        start_epoch,
        args.epochs,
        it,
        cfg.LOCAL_RANK,
        tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('🎉 Optimized R-MAE Pre-training Finished!')


if __name__ == '__main__':
    main()