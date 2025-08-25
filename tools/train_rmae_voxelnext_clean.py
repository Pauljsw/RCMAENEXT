#!/usr/bin/env python
"""
tools/train_rmae_voxelnext_clean.py

🎯 Clean R-MAE + VoxelNeXt Training Script with Overfitting Prevention
공식 R-MAE GitHub 코드 기반 + 완전한 validation monitoring

핵심 기능:
1. 공식 R-MAE 코드의 단순하고 효과적인 훈련 로직
2. ✅ Validation monitoring 및 overfitting prevention
3. ✅ Early stopping with patience
4. ✅ Learning curve 추적 및 분석
5. ✅ 기존 train/val split 활용

Usage:
    # Pretraining (validation monitoring 자동 활성화)
    python train_rmae_voxelnext_clean.py \
        --cfg_file cfgs/custom_models/rmae_voxelnext_clean_pretraining.yaml \
        --batch_size 4 \
        --epochs 30 \
        --extra_tag rmae_clean_pretraining
    
    # Fine-tuning (기존 방식 그대로)
    bash scripts/dist_train.sh 1 \
        --cfg_file cfgs/custom_models/rmae_voxelnext_isarc_4class_finetune.yaml \
        --pretrained_model output/.../checkpoint_epoch_30.pth \
        --batch_size 4 \
        --extra_tag rmae_finetune
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


class OverfittingDetector:
    """🔍 과적합 탐지 및 경고 시스템"""
    
    def __init__(self, patience=8, min_delta=0.005, lookback_window=5):
        self.patience = patience
        self.min_delta = min_delta
        self.lookback_window = lookback_window
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
        
    def update(self, train_loss, val_loss, epoch):
        """Loss 업데이트 및 과적합 체크"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # 1. Early stopping 체크
        early_stop = self._check_early_stopping(val_loss, epoch)
        
        # 2. 과적합 경고 체크
        overfitting_warning = self._check_overfitting_trend()
        
        return {
            'early_stop': early_stop,
            'overfitting_warning': overfitting_warning,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'best_epoch': self.best_epoch
        }
    
    def _check_early_stopping(self, val_loss, epoch):
        """Early stopping 로직"""
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience
    
    def _check_overfitting_trend(self):
        """과적합 경향 감지"""
        if len(self.train_losses) < self.lookback_window:
            return False
            
        recent_train = self.train_losses[-self.lookback_window:]
        recent_val = self.val_losses[-self.lookback_window:]
        
        # 최근 N epoch에서 train loss는 감소하지만 val loss는 증가하는 경우
        train_trend = np.polyfit(range(len(recent_train)), recent_train, 1)[0]
        val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]
        
        return train_trend < -0.001 and val_trend > 0.001
    
    def get_summary(self):
        """훈련 요약 정보"""
        if len(self.val_losses) == 0:
            return "No validation data available"
            
        return {
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
            'final_val_loss': self.val_losses[-1] if self.val_losses else 0,
            'total_epochs': len(self.val_losses)
        }


def parse_config():
    """🎯 공식 R-MAE 스타일의 argument parsing"""
    parser = argparse.ArgumentParser(description='Clean R-MAE VoxelNeXt Training')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='clean_rmae', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model for fine-tuning')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def build_validation_loader(logger):
    """🔍 Validation dataloader 생성 (기존 train/val split 활용)"""
    try:
        val_set, val_loader, val_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=2,  # Validation용 작은 배치
            dist=False,
            workers=2,
            logger=logger,
            training=False,  # ✅ Validation 모드
            merge_all_iters_to_one_epoch=False,
            total_epochs=1
        )
        if logger:
            logger.info(f"🔍 Validation loader created: {len(val_loader)} batches")
        return val_loader
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Failed to create validation loader: {e}")
        return None


def validate_model(model, val_loader, model_func, logger=None):
    """🔍 Validation 실행"""
    if val_loader is None:
        return None
        
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                loss, _, _ = model_func(model, batch)
                val_losses.append(loss.item())
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Validation batch failed: {e}")
                continue
    
    model.train()
    
    if len(val_losses) > 0:
        avg_val_loss = np.mean(val_losses)
        if logger:
            logger.info(f"🔍 Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
    else:
        if logger:
            logger.warning("⚠️ No valid validation batches")
        return None


def model_fn_decorator():
    """🎯 공식 R-MAE 스타일의 model function decorator"""
    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        
        # Global step 업데이트 (training 시에만)
        if model.training:
            if hasattr(model, 'update_global_step'):
                model.update_global_step()
            else:
                model.module.update_global_step()

        return loss, tb_dict, disp_dict

    return model_func


def load_data_to_gpu(batch_dict):
    """🎯 공식 스타일의 GPU 데이터 로딩"""
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def train_one_epoch_clean(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                         rank, tbar, tb_log=None, leave_pbar=False, total_it_each_epoch=None, dataloader_iter=None):
    """🎯 공식 R-MAE 스타일의 clean epoch training"""
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    epoch_losses = []

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

        # 📍 Forward pass
        loss, tb_dict, disp_dict = model_func(model, batch)
        epoch_losses.append(loss.item())

        # 📍 Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # 📍 Logging (공식 스타일로 단순화)
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
    
    if rank == 0:
        pbar.close()
    
    avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
    return accumulated_iter, avg_epoch_loss


def train_model_clean(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                     start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir,
                     train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                     max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None):
    """🎯 Validation monitoring과 overfitting prevention이 통합된 clean model training"""
    
    accumulated_iter = start_iter
    dataloader_iter = iter(train_loader)
    
    # 📍 Pretraining/Fine-tuning 모드 체크
    is_pretraining = (hasattr(cfg.MODEL.BACKBONE_3D, 'PRETRAINING') and 
                     cfg.MODEL.BACKBONE_3D.PRETRAINING)
    mode_name = "Pretraining" if is_pretraining else "Fine-tuning"
    
    # 📍 Validation 설정 (Pretraining 시에만)
    val_loader = None
    overfitting_detector = None
    if is_pretraining and rank == 0:
        # Validation dataloader 생성
        val_loader = build_validation_loader(logger)
        # Overfitting 탐지기 초기화
        overfitting_detector = OverfittingDetector(
            patience=getattr(cfg.OPTIMIZATION, 'EARLY_STOP_PATIENCE', 8),
            min_delta=getattr(cfg.OPTIMIZATION, 'EARLY_STOP_MIN_DELTA', 0.005)
        )
        if logger:
            logger.info("🔍 Validation monitoring enabled for overfitting prevention")
    
    print(f"🎯 Clean R-MAE {mode_name} Started")
    
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # 📍 Scheduler 선택
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            # 📍 한 epoch 훈련
            accumulated_iter, avg_train_loss = train_one_epoch_clean(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg, rank=rank, tbar=tbar,
                tb_log=tb_log, leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch, dataloader_iter=dataloader_iter
            )

            # 📍 Validation 및 과적합 체크 (Pretraining 시에만)
            if is_pretraining and overfitting_detector is not None and rank == 0:
                avg_val_loss = validate_model(model, val_loader, model_func, logger)
                
                if avg_val_loss is not None:
                    # Tensorboard logging
                    if tb_log is not None:
                        tb_log.add_scalar('val/loss', avg_val_loss, cur_epoch)
                        tb_log.add_scalar('train/epoch_loss', avg_train_loss, cur_epoch)
                    
                    # 과적합 체크
                    detector_result = overfitting_detector.update(avg_train_loss, avg_val_loss, cur_epoch)
                    
                    # 과적합 경고
                    if detector_result['overfitting_warning']:
                        if logger:
                            logger.warning(f"⚠️ Overfitting detected at epoch {cur_epoch}")
                    
                    # Early stopping 체크
                    if detector_result['early_stop']:
                        if logger:
                            logger.info(f"🛑 Early stopping triggered at epoch {cur_epoch}")
                            logger.info(f"   Best validation loss: {detector_result['best_val_loss']:.4f} at epoch {detector_result['best_epoch']}")
                        break
                    
                    # 진행 상황 로깅
                    if logger and cur_epoch % 5 == 0:
                        logger.info(f"Epoch {cur_epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
                        logger.info(f"   Best Val Loss = {detector_result['best_val_loss']:.4f}, Patience = {detector_result['patience_counter']}/{overfitting_detector.patience}")

            # 📍 Checkpoint 저장
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )
    
    # 📍 훈련 완료 요약
    if is_pretraining and overfitting_detector is not None and rank == 0 and logger:
        summary = overfitting_detector.get_summary()
        logger.info("🎯 Pretraining Summary:")
        logger.info(f"   Best Val Loss: {summary['best_val_loss']:.4f} at epoch {summary['best_epoch']}")
        logger.info(f"   Final Train Loss: {summary['final_train_loss']:.4f}")
        logger.info(f"   Final Val Loss: {summary['final_val_loss']:.4f}")
        logger.info(f"   Total Epochs: {summary['total_epochs']}")


def main():
    """🎯 Clean R-MAE VoxelNeXt 메인 함수"""
    args, cfg = parse_config()
    
    # 📍 분산 훈련 설정
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

    # 📍 출력 디렉토리 설정
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # 📍 로그 설정
    logger.info('**********************Start logging**********************')
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

    # 📍 데이터로더 & 네트워크 & 옵티마이저 생성
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
    
    # 📍 체크포인트 로딩
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)
    
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

    # 📍 훈련 시작
    mode_name = "Pretraining" if getattr(cfg.MODEL.BACKBONE_3D, 'PRETRAINING', False) else "Fine-tuning"
    logger.info('**********************Start Clean R-MAE %s %s/%s(%s)**********************'
                % (mode_name, cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
    train_model_clean(
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

    logger.info('**********************Clean R-MAE %s is finished**********************' % mode_name)


if __name__ == '__main__':
    main()