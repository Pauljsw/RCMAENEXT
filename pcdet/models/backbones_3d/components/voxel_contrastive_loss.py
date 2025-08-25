# pcdet/models/backbones_3d/components/voxel_contrastive_loss.py
"""
CMAE-3D Phase 2: Voxel-level Contrastive Learning

CMAE-3D 논문의 핵심 아이디어:
- Teacher (complete view)와 Student (masked view)의 voxel features 간 contrastive learning
- 같은 공간 위치의 voxel = positive pair
- 다른 공간 위치의 voxel = negative pair
- InfoNCE loss로 spatial consistency 학습

이 모듈은 voxel-level에서 teacher-student features 간의 
contrastive learning을 수행하여 robust representation을 학습합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VoxelContrastiveLoss(nn.Module):
    """
    🔥 CMAE-3D Voxel-level Contrastive Learning
    
    핵심 아이디어:
    1. Teacher features (complete view): 완전한 공간 정보 포함
    2. Student features (masked view): 부분적 공간 정보 포함  
    3. 같은 좌표 voxel = positive pair (공간적 일관성)
    4. 다른 좌표 voxel = negative pair (공간적 구분성)
    5. InfoNCE loss로 contrastive learning 수행
    """
    
    def __init__(self, model_cfg):
        super().__init__()
        
        # Contrastive learning 설정
        self.temperature = model_cfg.get('CONTRASTIVE_TEMPERATURE', 0.07)  # CMAE-3D 논문 기본값
        self.feature_dim = model_cfg.get('FEATURE_DIM', 128)
        self.projection_dim = model_cfg.get('PROJECTION_DIM', 128)
        
        # Positive pair 찾기 설정
        self.coord_tolerance = model_cfg.get('COORD_TOLERANCE', 0)  # 정확한 좌표 매칭 (0) vs 근사 매칭 (>0)
        self.max_negative_samples = model_cfg.get('MAX_NEGATIVE_SAMPLES', 4096)  # 메모리 효율성
        
        # Hard negative mining 설정 
        self.enable_hard_negative_mining = model_cfg.get('ENABLE_HARD_NEGATIVE_MINING', True)
        self.hard_negative_ratio = model_cfg.get('HARD_NEGATIVE_RATIO', 0.3)  # 30% hard negatives
        
        # 📍 Feature Projection Heads (Teacher/Student features → contrastive space)
        self.teacher_projector = self._build_projection_head(self.feature_dim, self.projection_dim)
        self.student_projector = self._build_projection_head(self.feature_dim, self.projection_dim)
        
        print(f"🔥 Voxel Contrastive Loss initialized:")
        print(f"   - Temperature: {self.temperature}")
        print(f"   - Feature dim: {self.feature_dim} → Projection dim: {self.projection_dim}")
        print(f"   - Hard negative mining: {self.enable_hard_negative_mining}")
        print(f"   - Max negative samples: {self.max_negative_samples}")
    
    def _build_projection_head(self, input_dim, output_dim):
        """Contrastive learning을 위한 projection head"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim)  # 중요: contrastive learning에서 BN이 성능 향상
        )
    
    def forward(self, teacher_features, student_features, teacher_coords, student_coords):
        """
        Voxel-level contrastive learning forward pass
        
        Args:
            teacher_features: [N_t, feature_dim] Teacher voxel features
            student_features: [N_s, feature_dim] Student voxel features  
            teacher_coords: [N_t, 4] Teacher voxel coordinates (batch, z, y, x)
            student_coords: [N_s, 4] Student voxel coordinates (batch, z, y, x)
        
        Returns:
            contrastive_results: Dict containing contrastive loss and statistics
        """
        # 📍 1. Feature Projection (Teacher/Student → Contrastive space)
        teacher_proj = self.teacher_projector(teacher_features)  # [N_t, projection_dim]
        student_proj = self.student_projector(student_features)  # [N_s, projection_dim]
        
        # L2 normalization (contrastive learning에서 중요)
        teacher_proj = F.normalize(teacher_proj, dim=1)  # [N_t, projection_dim]
        student_proj = F.normalize(student_proj, dim=1)  # [N_s, projection_dim]
        
        # 📍 2. Positive Pairs 찾기 (같은 좌표 voxel)
        positive_pairs, positive_indices = self._find_positive_pairs(
            teacher_coords, student_coords
        )
        
        if len(positive_pairs) == 0:
            # Positive pair가 없으면 contrastive learning 불가
            print("⚠️ No positive pairs found for voxel contrastive learning")
            return {
                'voxel_contrastive_loss': torch.tensor(0.0, device=teacher_features.device, requires_grad=True),
                'num_positive_pairs': 0,
                'num_negative_pairs': 0,
                'contrastive_acc': 0.0
            }
        
        # 📍 3. InfoNCE Loss 계산
        contrastive_loss, stats = self._compute_infonce_loss(
            teacher_proj, student_proj, positive_pairs, positive_indices
        )
        
        return {
            'voxel_contrastive_loss': contrastive_loss,
            'num_positive_pairs': len(positive_pairs),
            'num_negative_pairs': stats['num_negatives'],
            'contrastive_acc': stats['accuracy'],
            'avg_positive_sim': stats['avg_positive_sim'],
            'avg_negative_sim': stats['avg_negative_sim']
        }
    
    def _find_positive_pairs(self, teacher_coords, student_coords):
        """
        같은 공간 좌표를 가진 teacher-student voxel pairs 찾기
        
        Returns:
            positive_pairs: List of (teacher_idx, student_idx) tuples
            positive_indices: Dict for efficient lookup
        """
        # 좌표를 string key로 변환 (batch, z, y, x)
        teacher_keys = {}
        student_keys = {}
        
        for idx, coord in enumerate(teacher_coords):
            key = f"{coord[0].item()}_{coord[1].item()}_{coord[2].item()}_{coord[3].item()}"
            teacher_keys[key] = idx
        
        for idx, coord in enumerate(student_coords):
            key = f"{coord[0].item()}_{coord[1].item()}_{coord[2].item()}_{coord[3].item()}"
            if key not in student_keys:
                student_keys[key] = []
            student_keys[key].append(idx)
        
        # 공통 좌표 찾기 (positive pairs)
        positive_pairs = []
        positive_indices = {'teacher': [], 'student': []}
        
        for key in teacher_keys:
            if key in student_keys:
                teacher_idx = teacher_keys[key]
                # 같은 좌표에 여러 student voxel이 있을 수 있음 (rare case)
                for student_idx in student_keys[key]:
                    positive_pairs.append((teacher_idx, student_idx))
                    positive_indices['teacher'].append(teacher_idx)
                    positive_indices['student'].append(student_idx)
        
        return positive_pairs, positive_indices
    
    def _compute_infonce_loss(self, teacher_proj, student_proj, positive_pairs, positive_indices):
        """
        InfoNCE (Contrastive) Loss 계산
        
        각 student voxel에 대해:
        - Positive: 같은 좌표의 teacher voxel (1개)  
        - Negatives: 다른 좌표의 모든 teacher voxels
        
        Loss = -log(exp(pos_sim/τ) / (exp(pos_sim/τ) + Σexp(neg_sim/τ)))
        """
        if len(positive_pairs) == 0:
            return torch.tensor(0.0, device=teacher_proj.device, requires_grad=True), {
                'num_negatives': 0, 'accuracy': 0.0, 'avg_positive_sim': 0.0, 'avg_negative_sim': 0.0
            }
        
        total_loss = 0.0
        num_pairs = len(positive_pairs)
        positive_sims = []
        negative_sims = []
        correct_predictions = 0
        
        # Negative sampling을 위한 teacher features (메모리 효율성)
        if teacher_proj.size(0) > self.max_negative_samples:
            neg_indices = torch.randperm(teacher_proj.size(0))[:self.max_negative_samples]
            teacher_neg_pool = teacher_proj[neg_indices]  # [max_neg_samples, projection_dim]
        else:
            teacher_neg_pool = teacher_proj  # [N_t, projection_dim]
        
        # 📍 각 positive pair에 대해 InfoNCE loss 계산
        for teacher_idx, student_idx in positive_pairs:
            student_feat = student_proj[student_idx:student_idx+1]  # [1, projection_dim]
            teacher_pos_feat = teacher_proj[teacher_idx:teacher_idx+1]  # [1, projection_dim]
            
            # Positive similarity
            pos_sim = torch.mm(student_feat, teacher_pos_feat.t()) / self.temperature  # [1, 1]
            positive_sims.append(pos_sim.item())
            
            # Negative similarities (student vs all other teachers)
            neg_sim = torch.mm(student_feat, teacher_neg_pool.t()) / self.temperature  # [1, max_neg_samples]
            
            # Hard negative mining (선택적)
            if self.enable_hard_negative_mining:
                neg_sim = self._apply_hard_negative_mining(neg_sim, pos_sim)
            
            negative_sims.extend(neg_sim.squeeze().tolist())
            
            # InfoNCE loss = -log(exp(pos) / (exp(pos) + Σexp(neg)))
            all_sims = torch.cat([pos_sim, neg_sim], dim=1)  # [1, 1+neg_samples]
            log_softmax = F.log_softmax(all_sims, dim=1)
            pair_loss = -log_softmax[0, 0]  # Positive sample이 첫 번째 (index 0)
            
            total_loss += pair_loss
            
            # Accuracy 계산 (positive sample이 가장 높은 similarity를 가지는가?)
            if torch.argmax(all_sims, dim=1) == 0:
                correct_predictions += 1
        
        # Average loss
        avg_loss = total_loss / num_pairs
        
        # Statistics
        stats = {
            'num_negatives': len(negative_sims),
            'accuracy': correct_predictions / num_pairs if num_pairs > 0 else 0.0,
            'avg_positive_sim': np.mean(positive_sims) if positive_sims else 0.0,
            'avg_negative_sim': np.mean(negative_sims) if negative_sims else 0.0
        }
        
        return avg_loss, stats
    
    def _apply_hard_negative_mining(self, neg_sim, pos_sim):
        """
        Hard negative mining: 가장 어려운 negative samples 선택
        
        너무 쉬운 negatives (similarity가 매우 낮음)는 제외하고
        어려운 negatives (similarity가 높지만 positive보다는 낮음)를 선택
        """
        # Hard negatives 선택: positive similarity에 가까운 negative samples
        num_hard_negatives = max(1, int(neg_sim.size(1) * self.hard_negative_ratio))
        
        # Top-k negative similarities 선택 (positive보다는 낮아야 함)
        hard_neg_indices = torch.topk(neg_sim, k=min(num_hard_negatives, neg_sim.size(1)), dim=1)[1]
        hard_neg_sim = torch.gather(neg_sim, 1, hard_neg_indices)
        
        return hard_neg_sim
    
    def get_contrastive_config(self):
        """Contrastive learning 설정 반환 (debugging용)"""
        return {
            'temperature': self.temperature,
            'feature_dim': self.feature_dim,
            'projection_dim': self.projection_dim,
            'hard_negative_mining': self.enable_hard_negative_mining,
            'max_negative_samples': self.max_negative_samples
        }