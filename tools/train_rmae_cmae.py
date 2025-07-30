#!/usr/bin/env python
"""
tools/train_rmae_cmae.py

R-MAE + CMAE-3D 완전 훈련 스크립트

기존 성공한 train_voxel_mae.py를 기반으로 CMAE-3D 요소를 완벽하게 통합:
- ✅ R-MAE 성공 훈련 로직 100% 유지
- ➕ CMAE-3D 상세 로깅 및 모니터링
- ➕ Teacher-Student 동기화 체크
- ➕ Training stability 실시간 모니터링
- ➕ 고급 체크포인트 관리

Usage:
    python train_rmae_cmae.py \
        --cfg_file cfgs/custom_models/rmae_cmae_voxelnext_pretraining.yaml \
        --batch_size 8 \
        --extra_tag rmae_cmae_integration_complete
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
import subprocess
import torch.multiprocessing as mp
import torch.distributed as dist

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler

# CMAE-3D 유틸리티 import
try:
    from pcdet.utils.cmae_utils import (
        MemoryQueueManager, 
        ContrastivePairGenerator,
        TrainingStabilityChecker,
        log_contrastive_metrics
    )
    CMAE_UTILS_AVAILABLE = True
    print("✅ CMAE-3D utilities loaded successfully")
except ImportError as e:
    CMAE_UTILS_AVAILABLE = False
    print(f"⚠️  CMAE-3D utilities not available: {e}")


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


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """✅ 기존 성공 로직 그대로"""
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)

    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    """✅ 기존 성공 로직 그대로"""
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        init_method=f'tcp://127.0.0.1:{tcp_port}',
        world_size=num_gpus,
        rank=local_rank,
    )
    return num_gpus, local_rank


def model_fn_decorator():
    """✅ R-MAE + CMAE-3D model function (기존 성공 로직 기반)"""
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


class CMAETrainingManager:
    """
    ➕ CMAE-3D 전용 훈련 관리자
    
    기존 R-MAE 훈련에 CMAE-3D 모니터링 및 관리 기능 추가:
    - Teacher-Student 동기화 체크
    - Contrastive learning 메트릭 추적
    - Training stability 모니터링
    - 고급 체크포인트 관리
    """
    
    def __init__(self, model, tb_log, logger, full_cfg):
        self.model = model
        self.tb_log = tb_log
        self.logger = logger
        self.cfg = full_cfg  # 전체 config 저장
        
        # CMAE-3D 유틸리티 초기화
        if CMAE_UTILS_AVAILABLE:
            self.stability_checker = TrainingStabilityChecker(window_size=50)
            self.pair_generator = ContrastivePairGenerator(hard_negative_ratio=0.3)
            self.cmae_monitoring = True
            print("✅ CMAE-3D monitoring enabled")
        else:
            self.cmae_monitoring = False
            print("⚠️  CMAE-3D monitoring disabled (utils not available)")
        
        # 메트릭 히스토리
        self.loss_history = []
        self.alignment_history = []
        self.best_loss = float('inf')
        
        # 체크포인트 관리 (전체 cfg 전달)
        self.checkpoint_manager = CheckpointManager(full_cfg)
        
    def log_training_metrics(self, epoch, it, loss_dict, tb_dict, elapsed_time):
        """
        ➕ CMAE-3D 상세 훈련 메트릭 로깅
        """
        # ✅ 기존 R-MAE 로깅 유지
        if self.tb_log is not None:
            # 기본 손실 로깅
            for key, val in tb_dict.items():
                self.tb_log.add_scalar(f'train/{key}', val, it)
            
            # 학습 진행 상황
            self.tb_log.add_scalar('train/learning_rate', tb_dict.get('lr', 0.0), it)
            self.tb_log.add_scalar('train/epoch', epoch, it)
        
        # ➕ CMAE-3D 전용 로깅
        if self.cmae_monitoring and self.tb_log is not None:
            # 개별 손실 분석
            for loss_name in ['occupancy_loss', 'mlfr_loss', 'contrastive_loss']:
                if loss_name in tb_dict:
                    self.tb_log.add_scalar(f'cmae/{loss_name}', tb_dict[loss_name], it)
            
            # 손실 비율 분석
            total_loss = tb_dict.get('total_loss', loss_dict.get('loss', 0))
            if total_loss > 0:
                for loss_name in ['occupancy_loss', 'mlfr_loss', 'contrastive_loss']:
                    if loss_name in tb_dict:
                        ratio = tb_dict[loss_name] / total_loss
                        self.tb_log.add_scalar(f'cmae/loss_ratio_{loss_name}', ratio, it)
            
            # Training stability 체크
            if len(self.loss_history) > 0:
                stability_flags = self.stability_checker.check_loss_stability(loss_dict)
                for loss_name, is_stable in stability_flags.items():
                    self.tb_log.add_scalar(f'stability/{loss_name}_stable', float(is_stable), it)
        
        # 성능 메트릭
        if self.tb_log is not None:
            self.tb_log.add_scalar('performance/iter_time', elapsed_time, it)
            
            # GPU 메모리 사용량
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
                self.tb_log.add_scalar('performance/gpu_memory_gb', memory_used, it)
        
        # 손실 히스토리 업데이트
        self.loss_history.append(loss_dict)
        if len(self.loss_history) > 1000:  # 메모리 관리
            self.loss_history.pop(0)
    
    def check_teacher_student_alignment(self, batch_dict, it):
        """
        ➕ Teacher-Student feature alignment 체크
        """
        if not self.cmae_monitoring:
            return
            
        # Teacher-Student features 추출
        student_features = batch_dict.get('student_features')
        teacher_features = batch_dict.get('teacher_features')
        
        if student_features is not None and teacher_features is not None:
            # Multi-scale features에서 최종 feature 추출
            if isinstance(student_features, dict) and 'multi_scale_3d_features' in student_features:
                student_feat = student_features['multi_scale_3d_features']['x_conv4']
                teacher_feat = teacher_features['multi_scale_3d_features']['x_conv4']
                
                # Sparse tensor features 추출
                if hasattr(student_feat, 'features') and hasattr(teacher_feat, 'features'):
                    student_dense = student_feat.features
                    teacher_dense = teacher_feat.features
                    
                    # Feature alignment 체크
                    if student_dense.shape == teacher_dense.shape:
                        alignment_score = self.stability_checker.check_teacher_student_alignment(
                            student_dense, teacher_dense
                        )
                        
                        # Tensorboard 로깅
                        if self.tb_log is not None:
                            self.tb_log.add_scalar('cmae/teacher_student_alignment', alignment_score, it)
                        
                        self.alignment_history.append(alignment_score)
                        
                        # 낮은 alignment 경고
                        if alignment_score < 0.3:
                            self.logger.warning(f"Low Teacher-Student alignment: {alignment_score:.3f}")
    
    def save_checkpoint_if_best(self, epoch, it, loss_dict, model, optimizer, ckpt_save_dir):
        """
        ➕ 개선된 체크포인트 저장 (best model tracking)
        """
        current_loss = loss_dict.get('loss', float('inf'))
        
        # Best model 체크
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            
            # Best model 저장
            best_ckpt_name = ckpt_save_dir / f'checkpoint_best.pth'
            self.checkpoint_manager.save_checkpoint(
                model, optimizer, epoch, it, loss_dict, best_ckpt_name, is_best=True
            )
            
            self.logger.info(f"✅ New best model saved: loss={current_loss:.4f}")
        
        # 정기 체크포인트 저장
        if epoch % self.cfg.OPTIMIZATION.get('CKPT_SAVE_INTERVAL', 1) == 0:
            regular_ckpt_name = ckpt_save_dir / f'checkpoint_epoch_{epoch}.pth'
            self.checkpoint_manager.save_checkpoint(
                model, optimizer, epoch, it, loss_dict, regular_ckpt_name, is_best=False
            )
    
    def get_training_summary(self):
        """
        ➕ 훈련 요약 정보 생성
        """
        summary = {
            'total_iterations': len(self.loss_history),
            'best_loss': self.best_loss,
        }
        
        if self.cmae_monitoring and len(self.loss_history) > 0:
            # 최근 손실 평균
            recent_losses = self.loss_history[-10:] if len(self.loss_history) >= 10 else self.loss_history
            if recent_losses:
                for loss_name in recent_losses[0].keys():
                    loss_values = [item.get(loss_name, 0) for item in recent_losses]
                    summary[f'{loss_name}_recent_avg'] = np.mean(loss_values)
            
            # Teacher-Student alignment 요약
            if len(self.alignment_history) > 0:
                summary['alignment_mean'] = np.mean(self.alignment_history[-20:])
                summary['alignment_trend'] = np.mean(np.diff(self.alignment_history[-10:])) if len(self.alignment_history) > 1 else 0.0
        
        return summary


class CheckpointManager:
    """
    ➕ 고급 체크포인트 관리자
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_checkpoints = cfg.OPTIMIZATION.get('MAX_CKPT_SAVE_NUM', 30)
        
    def save_checkpoint(self, model, optimizer, epoch, it, loss_dict, ckpt_path, is_best=False):
        """체크포인트 저장 with 메타데이터"""
        checkpoint = {
            'epoch': epoch,
            'it': it,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss_dict': loss_dict,
            'config': self.cfg,
            'timestamp': datetime.datetime.now().isoformat(),
            'is_best': is_best,
        }
        
        torch.save(checkpoint, ckpt_path)
        
        # 오래된 체크포인트 정리 (best 제외)
        if not is_best:
            self._cleanup_old_checkpoints(ckpt_path.parent)
    
    def _cleanup_old_checkpoints(self, ckpt_dir):
        """오래된 체크포인트 정리"""
        checkpoint_files = list(ckpt_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoint_files) > self.max_checkpoints:
            # 에포크 순으로 정렬하여 오래된 것 삭제
            checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for old_ckpt in checkpoint_files[:-self.max_checkpoints]:
                old_ckpt.unlink()


def train_model_rmae_cmae(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                         start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir,
                         train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                         max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, full_cfg=None):
    """
    ➕ R-MAE + CMAE-3D 통합 훈련 함수
    
    기존 성공한 R-MAE 훈련 로직에 CMAE-3D 모니터링 및 관리 기능 추가
    """
    # CMAE 훈련 관리자 초기화 (전체 cfg 전달)
    cmae_manager = CMAETrainingManager(model, tb_log, logger, full_cfg)
    
    accumulated_iter = start_iter
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

            # 학습률 warmup
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            accumulated_iter_epoch = 0
            with tqdm.tqdm(total=total_it_each_epoch, leave=(rank == 0), desc='training', dynamic_ncols=True) as pbar:
                for cur_it in range(total_it_each_epoch):
                    try:
                        batch = next(dataloader_iter)
                    except StopIteration:
                        dataloader_iter = iter(train_loader)
                        batch = next(dataloader_iter)
                        print('new iters')

                    # ✅ 기존 R-MAE 훈련 로직
                    start_time = time.time()
                    
                    cur_scheduler.step(accumulated_iter)
                    try:
                        cur_lr = float(optimizer.lr)
                    except:
                        cur_lr = optimizer.param_groups[0]['lr']

                    if tb_log is not None:
                        tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

                    model.train()
                    optimizer.zero_grad()
                    
                    loss, tb_dict, disp_dict = model_func(model, batch)
                    
                    # 손실 정보 구성
                    loss_dict = {'loss': loss.item()}
                    if isinstance(tb_dict, dict):
                        loss_dict.update({k: v for k, v in tb_dict.items() if isinstance(v, (int, float))})

                    loss.backward()
                    clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                    optimizer.step()

                    accumulated_iter += 1
                    accumulated_iter_epoch += 1
                    
                    elapsed_time = time.time() - start_time

                    # ➕ CMAE-3D 상세 로깅
                    cmae_manager.log_training_metrics(cur_epoch, accumulated_iter, loss_dict, tb_dict, elapsed_time)
                    
                    # ➕ Teacher-Student alignment 체크
                    if hasattr(model, 'module'):
                        # DataParallel/DistributedDataParallel
                        batch_dict = getattr(model.module, '_last_batch_dict', {})
                    else:
                        batch_dict = getattr(model, '_last_batch_dict', {})
                    cmae_manager.check_teacher_student_alignment(batch_dict, accumulated_iter)

                    # Progress bar 업데이트
                    pbar.update()
                    pbar.set_postfix(dict(loss=f"{loss:.3f}", lr=f"{cur_lr:.6f}"))
                    tbar.set_postfix(disp_dict)

            # 에포크 완료 후 체크포인트 저장
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                cmae_manager.save_checkpoint_if_best(
                    trained_epoch, accumulated_iter, loss_dict, model, optimizer, ckpt_save_dir
                )

    # 훈련 완료 요약
    if rank == 0:
        training_summary = cmae_manager.get_training_summary()
        logger.info("🎯 R-MAE + CMAE-3D Training Summary:")
        for key, value in training_summary.items():
            logger.info(f"   - {key}: {value}")


def main():
    args, cfg = parse_config()
    
    # ✅ 기존 성공 로직: 분산 훈련 설정
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = init_dist_slurm(args.tcp_port, args.local_rank) if args.launcher == 'slurm' else \
                                     init_dist_pytorch(args.tcp_port, args.local_rank)
        dist_train = True

    # ✅ 기존 성공 로직: 배치 크기 및 에포크 설정
    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, f'Batch size should be matched with GPUS: ({args.batch_size}, {total_gpus})'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    # ✅ 기존 성공 로직: 출력 디렉토리 설정
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # 설정 로깅
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

    # ✅ 기존 성공 로직: 데이터셋 및 모델 구축
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

    # ✅ 기존 성공 로직: 체크포인트 로딩
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

    # ✅ 훈련 시작
    logger.info('**********************🎯 Start R-MAE + CMAE-3D Training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
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
        logger=logger,
        full_cfg=cfg  # 전체 cfg 전달
    )

    logger.info('**********************🎯 R-MAE + CMAE-3D Training Finished**********************')

if __name__ == '__main__':
    main()