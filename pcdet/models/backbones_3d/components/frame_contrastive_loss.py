# pcdet/models/backbones_3d/components/frame_contrastive_loss.py
"""
CMAE-3D Phase 2: Frame-level Contrastive Learning

CMAE-3D 논문의 핵심 아이디어:
- 인접 프레임 간 semantic similarity 강화 (시간적 일관성)
- False negative 문제 완화 (같은 객체가 연속 프레임에서 다르게 인식되는 문제)
- Memory bank를 통한 temporal feature 관리
- 건설장비 도메인 특화: 같은 장비 타입 = positive, 다른 장비 타입 = negative

이 모듈은 frame-level에서 temporal consistency를 학습하여 
robust하고 temporally consistent한 representation을 구축합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


class FrameContrastiveLoss(nn.Module):
    """
    🔥 CMAE-3D Frame-level Contrastive Learning
    
    핵심 아이디어:
    1. Memory bank: 이전 프레임들의 features 저장
    2. Temporal consistency: 인접 프레임 간 similarity 강화  
    3. Domain-specific: 건설장비 타입별 positive/negative 구분
    4. False negative 완화: 같은 장비의 연속 프레임 = positive
    5. InfoNCE + Momentum update 활용
    """
    
    def __init__(self, model_cfg):
        super().__init__()
        
        # Frame contrastive learning 설정
        self.temperature = model_cfg.get('FRAME_TEMPERATURE', 0.1)  # Frame-level temperature (낮게 설정)
        self.feature_dim = model_cfg.get('FEATURE_DIM', 128)
        self.projection_dim = model_cfg.get('PROJECTION_DIM', 128)
        
        # Memory bank 설정 (temporal consistency)
        self.memory_bank_size = model_cfg.get('MEMORY_BANK_SIZE', 16)  # 최근 16 프레임 저장
        self.momentum = model_cfg.get('MOMENTUM_UPDATE', 0.99)  # Momentum update coefficient
        
        # Equipment type contrastive learning 설정
        self.enable_domain_specific = model_cfg.get('ENABLE_DOMAIN_SPECIFIC', True)
        self.equipment_types = model_cfg.get('EQUIPMENT_TYPES', ['dumptruck', 'excavator', 'grader', 'roller'])
        
        # Negative sampling 설정
        self.max_negatives = model_cfg.get('MAX_FRAME_NEGATIVES', 1024)
        self.hard_negative_ratio = model_cfg.get('HARD_NEGATIVE_RATIO', 0.2)  # 20% hard negatives
        
        # 📍 Feature Projection Head
        self.frame_projector = self._build_projection_head(self.feature_dim, self.projection_dim)
        
        # 📍 Memory Bank 초기화 (queue 구조)
        self.register_buffer('memory_bank', torch.randn(self.memory_bank_size, self.projection_dim))
        self.register_buffer('memory_timestamps', torch.zeros(self.memory_bank_size))
        self.register_buffer('memory_equipment_types', torch.zeros(self.memory_bank_size))
        self.register_buffer('bank_ptr', torch.zeros(1, dtype=torch.long))
        
        # L2 normalize memory bank
        self.memory_bank = F.normalize(self.memory_bank, dim=1)
        
        print(f"🔥 Frame Contrastive Loss initialized:")
        print(f"   - Frame temperature: {self.temperature}")
        print(f"   - Memory bank size: {self.memory_bank_size}")
        print(f"   - Momentum update: {self.momentum}")
        print(f"   - Domain-specific: {self.enable_domain_specific}")
        print(f"   - Equipment types: {self.equipment_types}")
    
    def _build_projection_head(self, input_dim, output_dim):
        """Frame-level contrastive learning을 위한 projection head"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
    
    @torch.no_grad()
    def _update_memory_bank(self, features, timestamps, equipment_types):
        """
        Memory bank를 momentum update로 갱신
        
        Args:
            features: [B, projection_dim] Current frame features  
            timestamps: [B] Frame timestamps
            equipment_types: [B] Equipment type indices
        """
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        # Circular queue update
        if ptr + batch_size <= self.memory_bank_size:
            # Normal case: 충분한 공간
            self.memory_bank[ptr:ptr+batch_size] = features
            self.memory_timestamps[ptr:ptr+batch_size] = timestamps
            self.memory_equipment_types[ptr:ptr+batch_size] = equipment_types
            ptr = (ptr + batch_size) % self.memory_bank_size
        else:
            # Wrap around case: queue 끝에 도달
            remaining = self.memory_bank_size - ptr
            self.memory_bank[ptr:] = features[:remaining]
            self.memory_bank[:batch_size-remaining] = features[remaining:]
            
            self.memory_timestamps[ptr:] = timestamps[:remaining]
            self.memory_timestamps[:batch_size-remaining] = timestamps[remaining:]
            
            self.memory_equipment_types[ptr:] = equipment_types[:remaining]
            self.memory_equipment_types[:batch_size-remaining] = equipment_types[remaining:]
            
            ptr = batch_size - remaining
        
        self.bank_ptr[0] = ptr
    
    def _find_temporal_positives(self, current_timestamps, current_equipment_types):
        """
        Temporal positive pairs 찾기
        
        Positive 조건:
        1. 시간적으로 가까운 프레임 (±2 프레임 이내)
        2. 같은 장비 타입 (domain-specific)
        
        Returns:
            positive_mask: [B, memory_bank_size] Boolean mask
        """
        batch_size = current_timestamps.size(0)
        positive_mask = torch.zeros(batch_size, self.memory_bank_size, dtype=torch.bool, device=current_timestamps.device)
        
        for i in range(batch_size):
            curr_time = current_timestamps[i]
            curr_type = current_equipment_types[i]
            
            # 시간적 근접성 체크 (±2 프레임)
            time_diff = torch.abs(self.memory_timestamps - curr_time)
            temporal_close = time_diff <= 2.0  # 2 프레임 이내
            
            if self.enable_domain_specific:
                # 도메인 특화: 같은 장비 타입
                same_equipment = self.memory_equipment_types == curr_type
                positive_mask[i] = temporal_close & same_equipment
            else:
                # 일반적: 시간적 근접성만
                positive_mask[i] = temporal_close
        
        return positive_mask
    
    def _compute_frame_infonce_loss(self, current_proj, positive_mask):
        """
        Frame-level InfoNCE loss 계산
        
        각 current frame에 대해:
        - Positives: Memory bank에서 temporal + domain 조건 만족하는 frames
        - Negatives: 나머지 모든 frames
        """
        batch_size = current_proj.size(0)
        
        if positive_mask.sum() == 0:
            # No positive pairs
            return torch.tensor(0.0, device=current_proj.device, requires_grad=True), {
                'frame_accuracy': 0.0, 'avg_positives': 0.0
            }
        
        # Cosine similarity: [B, memory_bank_size]
        similarity = torch.matmul(current_proj, self.memory_bank.T) / self.temperature
        
        total_loss = 0.0
        correct_predictions = 0
        total_positives = 0
        
        for i in range(batch_size):
            pos_mask = positive_mask[i]  # [memory_bank_size]
            num_positives = pos_mask.sum().item()
            
            if num_positives == 0:
                continue
            
            # Positive similarities
            pos_sim = similarity[i][pos_mask]  # [num_positives]
            
            # All similarities (positives + negatives)
            all_sim = similarity[i]  # [memory_bank_size]
            
            # InfoNCE loss for each positive
            pos_exp = torch.exp(pos_sim)
            all_exp = torch.exp(all_sim).sum()
            
            # Average over positives
            sample_loss = -torch.log(pos_exp.sum() / all_exp)
            total_loss += sample_loss
            
            # Accuracy: positive가 가장 높은 similarity를 가지는지
            max_sim_idx = all_sim.argmax()
            if pos_mask[max_sim_idx]:
                correct_predictions += 1
                
            total_positives += num_positives
        
        if batch_size == 0:
            return torch.tensor(0.0, device=current_proj.device, requires_grad=True), {
                'frame_accuracy': 0.0, 'avg_positives': 0.0
            }
        
        avg_loss = total_loss / batch_size
        accuracy = correct_predictions / batch_size if batch_size > 0 else 0.0
        avg_positives = total_positives / batch_size if batch_size > 0 else 0.0
        
        return avg_loss, {
            'frame_accuracy': accuracy,
            'avg_positives': avg_positives
        }
    
    def forward(self, current_features, timestamps, equipment_types):
        """
        Frame-level contrastive learning forward pass
        
        Args:
            current_features: [B, feature_dim] Current frame features
            timestamps: [B] Frame timestamps (frame indices)  
            equipment_types: [B] Equipment type indices (0: dumptruck, 1: excavator, ...)
        
        Returns:
            frame_contrastive_results: Dict containing frame contrastive loss and stats
        """
        batch_size = current_features.size(0)
        
        # 📍 1. Feature Projection
        current_proj = self.frame_projector(current_features)  # [B, projection_dim]
        current_proj = F.normalize(current_proj, dim=1)
        
        # 📍 2. Find Temporal Positive Pairs
        positive_mask = self._find_temporal_positives(timestamps, equipment_types)
        
        # 📍 3. Compute Frame InfoNCE Loss
        frame_loss, stats = self._compute_frame_infonce_loss(current_proj, positive_mask)
        
        # 📍 4. Update Memory Bank (no grad)
        with torch.no_grad():
            self._update_memory_bank(current_proj.detach(), timestamps, equipment_types)
        
        return {
            'frame_contrastive_loss': frame_loss,
            'frame_accuracy': stats['frame_accuracy'],
            'avg_temporal_positives': stats['avg_positives'],
            'memory_bank_usage': int(self.bank_ptr[0].item()),
            'total_positives': positive_mask.sum().item()
        }
    
    def get_memory_bank_status(self):
        """Memory bank 상태 정보 반환 (debugging용)"""
        return {
            'bank_size': self.memory_bank_size,
            'current_usage': int(self.bank_ptr[0].item()),
            'momentum': self.momentum,
            'temperature': self.temperature
        }