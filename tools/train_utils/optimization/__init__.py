from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


def build_optimizer(model, optim_cfg):
    """
    🔥 확장된 optimizer builder: 차등 학습률 지원
    
    기존 기능 + 차등 학습률 옵션 추가
    """
    
    # 📍 차등 학습률 활성화 시
    if optim_cfg.get('DIFFERENTIAL_LR', False):
        return create_optimizer_with_differential_lr(model, optim_cfg)
    
    # 📍 기존 optimizer 로직 (그대로 유지)
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer


def create_optimizer_with_differential_lr(model, optim_cfg):
    """
    🔥 차등 학습률 Optimizer 생성
    
    Backbone (pretrained) vs Dense Head (새로 학습)에 서로 다른 학습률 적용
    """
    
    # 📍 1. 모델 파라미터 그룹 분리
    backbone_params = []
    dense_head_params = []
    other_params = []
    
    print("🎯 Separating parameters for differential learning rates:")
    
    # 모델 구조 분석
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 3D Sparse CNN Backbone (pretrained)
        if any(backbone_key in name for backbone_key in [
            'backbone_3d', 'vfe', 'map_to_bev', 'backbone_2d'
        ]):
            backbone_params.append(param)
            print(f"   Backbone: {name}")
        
        # Dense Head (needs higher LR for adaptation)
        elif 'dense_head' in name:
            dense_head_params.append(param)
            print(f"   Dense Head: {name}")
        
        # 기타 파라미터들
        else:
            other_params.append(param)
            print(f"   Other: {name}")
    
    # 📍 2. 차등 학습률 설정
    base_lr = optim_cfg.LR
    backbone_lr = base_lr * optim_cfg.get('BACKBONE_LR_RATIO', 0.1)  # Backbone: 10%
    dense_head_lr = base_lr * optim_cfg.get('DENSE_HEAD_LR_RATIO', 2.0)  # Dense Head: 200%
    
    print(f"\n🎯 Differential Learning Rates:")
    print(f"   - Base LR: {base_lr:.6f}")
    print(f"   - Backbone LR: {backbone_lr:.6f} (ratio: {optim_cfg.get('BACKBONE_LR_RATIO', 0.1)})")
    print(f"   - Dense Head LR: {dense_head_lr:.6f} (ratio: {optim_cfg.get('DENSE_HEAD_LR_RATIO', 2.0)})")
    print(f"   - Other LR: {base_lr:.6f}")
    
    # 📍 3. Parameter Groups 생성
    param_groups = []
    
    if len(backbone_params) > 0:
        param_groups.append({
            'params': backbone_params,
            'lr': backbone_lr,
            'weight_decay': optim_cfg.WEIGHT_DECAY * 0.5,  # backbone은 더 적은 regularization
            'name': 'backbone'
        })
        print(f"   - Backbone group: {len(backbone_params)} parameters")
    
    if len(dense_head_params) > 0:
        param_groups.append({
            'params': dense_head_params, 
            'lr': dense_head_lr,
            'weight_decay': optim_cfg.WEIGHT_DECAY,
            'name': 'dense_head'
        })
        print(f"   - Dense Head group: {len(dense_head_params)} parameters")
    
    if len(other_params) > 0:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': optim_cfg.WEIGHT_DECAY,
            'name': 'other'
        })
        print(f"   - Other group: {len(other_params)} parameters")
    
    # 📍 4. Optimizer 생성 (기존 로직 활용)
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(param_groups)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(param_groups, momentum=optim_cfg.MOMENTUM)
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        # adam_onecycle의 경우 기본 Adam으로 대체 (parameter groups 호환성)
        optimizer = optim.Adam(param_groups, betas=(0.9, 0.99))
        print("   Note: Using Adam instead of adam_onecycle for differential LR compatibility")
    else:
        # Fallback to Adam
        print(f"Warning: Unknown optimizer {optim_cfg.OPTIMIZER}, using Adam")
        optimizer = optim.Adam(param_groups, betas=(0.9, 0.99))
    
    print(f"✅ Created {optim_cfg.OPTIMIZER} optimizer with {len(param_groups)} parameter groups")
    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    """기존 scheduler builder (그대로 유지)"""
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    
    # 📍 차등 학습률 사용 시 adam_onecycle 비호환성 처리
    if optim_cfg.OPTIMIZER == 'adam_onecycle' and not optim_cfg.get('DIFFERENTIAL_LR', False):
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * total_iters_each_epoch,
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler


# 📍 추가 유틸리티 함수들
def adjust_learning_rate_by_epoch(optimizer, epoch, optim_cfg):
    """
    에포크별 학습률 세부 조정 (차등 학습률용)
    초기에는 dense head가 더 빠르게 학습하도록 조정
    """
    if epoch < 10:  # 초기: dense head 강화
        backbone_factor = 0.5
        dense_head_factor = 2.0
    elif epoch < 30:  # 중기: 균형
        backbone_factor = 0.8  
        dense_head_factor = 1.2
    else:  # 후기: 전체적으로 fine-tuning
        backbone_factor = 1.0
        dense_head_factor = 1.0
    
    # Parameter group별 학습률 조정
    for param_group in optimizer.param_groups:
        if param_group.get('name') == 'backbone':
            param_group['lr'] *= backbone_factor
        elif param_group.get('name') == 'dense_head':
            param_group['lr'] *= dense_head_factor


def log_learning_rates(optimizer, logger, epoch=None):
    """현재 학습률 로깅"""
    epoch_str = f"Epoch {epoch}: " if epoch is not None else ""
    
    for i, param_group in enumerate(optimizer.param_groups):
        group_name = param_group.get('name', f'group_{i}')
        lr = param_group['lr']
        
        if logger:
            logger.info(f"{epoch_str}LR {group_name}: {lr:.6f}")
        else:
            print(f"{epoch_str}LR {group_name}: {lr:.6f}")