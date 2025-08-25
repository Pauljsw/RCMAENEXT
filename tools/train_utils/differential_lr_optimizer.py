"""
train_utils/differential_lr_optimizer.py

🎯 차등 학습률 Optimizer 모듈
기존 OpenPCDet 훈련 파이프라인에 간단히 통합 가능

사용법:
from train_utils.differential_lr_optimizer import build_differential_optimizer

# 기존 build_optimizer 대신 사용
optimizer = build_differential_optimizer(model, cfg.OPTIMIZATION)
"""

import torch


def build_differential_optimizer(model, optim_cfg):
    """
    🎯 차등 학습률 Optimizer 빌더
    
    기존 OpenPCDet의 build_optimizer를 대체하여 사용
    Config에서 차등 학습률 설정을 읽어와 자동 적용
    """
    
    # 차등 학습률 설정 확인
    if hasattr(optim_cfg, 'DIFFERENTIAL_LR') and optim_cfg.DIFFERENTIAL_LR.get('ENABLED', False):
        return _build_differential_lr_optimizer(model, optim_cfg)
    else:
        # 기존 방식 (호환성)
        from train_utils.optimization import build_optimizer
        return build_optimizer(model, optim_cfg)


def _build_differential_lr_optimizer(model, optim_cfg):
    """차등 학습률 Optimizer 구현"""
    
    # 기본 설정
    base_lr = optim_cfg.DIFFERENTIAL_LR.get('BASE_LR', optim_cfg.LR)
    lr_ratios = optim_cfg.DIFFERENTIAL_LR.get('LR_RATIOS', {
        'BACKBONE_EARLY': 0.1,
        'BACKBONE_LATE': 0.5, 
        'DETECTION_HEAD': 1.0,
        'OTHER': 0.3
    })
    weight_decay_cfg = optim_cfg.DIFFERENTIAL_LR.get('WEIGHT_DECAY', {
        'BACKBONE': 0.01,
        'HEAD': 0.001
    })
    
    # 파라미터 그룹 분류
    param_groups = []
    
    # Backbone Early Layers
    early_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and any(layer in name for layer in ['conv_input', 'conv1', 'conv2']):
            early_params.append(param)
    
    if early_params:
        param_groups.append({
            'params': early_params,
            'lr': base_lr * lr_ratios['BACKBONE_EARLY'],
            'weight_decay': weight_decay_cfg['BACKBONE'],
            'name': 'backbone_early'
        })
    
    # Backbone Late Layers  
    late_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and any(layer in name for layer in ['conv3', 'conv4']):
            late_params.append(param)
    
    if late_params:
        param_groups.append({
            'params': late_params,
            'lr': base_lr * lr_ratios['BACKBONE_LATE'],
            'weight_decay': weight_decay_cfg['BACKBONE'], 
            'name': 'backbone_late'
        })
    
    # Detection Head
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and any(head in name for head in ['dense_head', 'cls_head', 'reg_head', 'dir_head']):
            head_params.append(param)
    
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': base_lr * lr_ratios['DETECTION_HEAD'],
            'weight_decay': weight_decay_cfg['HEAD'],
            'name': 'detection_head'
        })
    
    # Other Parameters (나머지)
    processed_params = set()
    for group in param_groups:
        for param in group['params']:
            processed_params.add(id(param))
    
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in processed_params:
            other_params.append(param)
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr * lr_ratios['OTHER'],
            'weight_decay': weight_decay_cfg['BACKBONE'],
            'name': 'other'
        })
    
    # Optimizer 생성
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.99))
    elif optim_cfg.OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.Adam(param_groups)
    
    # 차등 학습률 정보 출력
    print("🎯 Differential Learning Rate Applied:")
    total_params = 0
    for group in param_groups:
        param_count = sum(p.numel() for p in group['params'])
        total_params += param_count
        print(f"   - {group['name']}: LR={group['lr']:.6f}, Params={param_count:,}")
    print(f"   - Total Parameters: {total_params:,}")
    
    return optimizer