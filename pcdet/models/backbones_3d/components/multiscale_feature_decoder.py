# pcdet/models/backbones_3d/components/multiscale_feature_decoder.py
"""
CMAE-3D Phase 2: Multi-scale Latent Feature Reconstruction (MLFR) Decoder

CMAE-3D 논문의 핵심 아이디어:
- 기존 MAE: 단순한 occupancy prediction (low-level)
- CMAE-3D MLFR: Multi-scale semantic feature reconstruction (high-level)

이 모듈은 Student의 masked features를 입력받아 
Teacher의 complete features를 재구성하는 multi-scale decoder입니다.
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from functools import partial
import torch.nn.functional as F


class MultiScaleFeatureDecoder(nn.Module):
    """
    🔥 CMAE-3D Multi-scale Latent Feature Reconstruction Decoder
    
    VoxelNeXt의 multi-scale features를 활용한 semantic reconstruction:
    - Scale 1: 16 channels (early features)
    - Scale 2: 32 channels (mid features)  
    - Scale 3: 64 channels (late features)
    - Scale 4: 128 channels (final features)
    
    각 scale별로 독립적인 decoder를 구성하여 
    해당 scale의 semantic 정보를 정확하게 재구성
    """
    
    def __init__(self, model_cfg):
        super().__init__()
        
        # MLFR 설정
        self.enable_mlfr = model_cfg.get('ENABLE_MLFR', True)
        self.mlfr_scales = model_cfg.get('MLFR_SCALES', ['scale_1', 'scale_2', 'scale_3', 'scale_4'])
        self.mlfr_loss_weights = model_cfg.get('MLFR_LOSS_WEIGHTS', {
            'scale_1': 0.5,  # Early features: lower weight
            'scale_2': 0.8,  # Mid features: medium weight
            'scale_3': 1.0,  # Late features: high weight  
            'scale_4': 1.2   # Final features: highest weight
        })
        
        # Sparse convolution normalization
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        # 📍 Scale별 Feature Decoder 구성
        self.scale_decoders = nn.ModuleDict()
        
        # Scale 1: 16 channels reconstruction
        if 'scale_1' in self.mlfr_scales:
            self.scale_decoders['scale_1'] = spconv.SparseSequential(
                # 16 -> 32 -> 16 (U-Net style)
                spconv.SubMConv3d(16, 32, 3, padding=1, bias=False, indice_key='mlfr_scale1_up'),
                norm_fn(32), nn.ReLU(),
                spconv.SubMConv3d(32, 16, 3, padding=1, bias=False, indice_key='mlfr_scale1_out'),
                norm_fn(16), nn.ReLU(),
                # Final projection
                spconv.SubMConv3d(16, 16, 1, bias=True, indice_key='mlfr_scale1_final')
            )
        
        # Scale 2: 32 channels reconstruction  
        if 'scale_2' in self.mlfr_scales:
            self.scale_decoders['scale_2'] = spconv.SparseSequential(
                # 32 -> 64 -> 32
                spconv.SubMConv3d(32, 64, 3, padding=1, bias=False, indice_key='mlfr_scale2_up'),
                norm_fn(64), nn.ReLU(),
                spconv.SubMConv3d(64, 32, 3, padding=1, bias=False, indice_key='mlfr_scale2_out'),
                norm_fn(32), nn.ReLU(),
                # Final projection
                spconv.SubMConv3d(32, 32, 1, bias=True, indice_key='mlfr_scale2_final')
            )
        
        # Scale 3: 64 channels reconstruction
        if 'scale_3' in self.mlfr_scales:
            self.scale_decoders['scale_3'] = spconv.SparseSequential(
                # 64 -> 128 -> 64
                spconv.SubMConv3d(64, 128, 3, padding=1, bias=False, indice_key='mlfr_scale3_up'),
                norm_fn(128), nn.ReLU(),
                spconv.SubMConv3d(128, 64, 3, padding=1, bias=False, indice_key='mlfr_scale3_out'),
                norm_fn(64), nn.ReLU(),
                # Final projection
                spconv.SubMConv3d(64, 64, 1, bias=True, indice_key='mlfr_scale3_final')
            )
        
        # Scale 4: 128 channels reconstruction (가장 중요한 final features)
        if 'scale_4' in self.mlfr_scales:
            self.scale_decoders['scale_4'] = spconv.SparseSequential(
                # 128 -> 256 -> 128 (더 큰 capacity)
                spconv.SubMConv3d(128, 256, 3, padding=1, bias=False, indice_key='mlfr_scale4_up'),
                norm_fn(256), nn.ReLU(),
                spconv.SubMConv3d(256, 128, 3, padding=1, bias=False, indice_key='mlfr_scale4_mid'),
                norm_fn(128), nn.ReLU(),
                spconv.SubMConv3d(128, 128, 3, padding=1, bias=False, indice_key='mlfr_scale4_out'),
                norm_fn(128), nn.ReLU(),
                # Final projection
                spconv.SubMConv3d(128, 128, 1, bias=True, indice_key='mlfr_scale4_final')
            )
        
        print(f"🔥 MLFR Decoder initialized:")
        print(f"   - Enabled scales: {self.mlfr_scales}")
        print(f"   - Loss weights: {self.mlfr_loss_weights}")
        print(f"   - Total decoders: {len(self.scale_decoders)}")
    
    def forward(self, student_features, teacher_features):
        """
        Multi-scale feature reconstruction forward pass
        
        Args:
            student_features: Dict of student multi-scale features
                {
                    'scale_1': SparseConvTensor (16 channels),
                    'scale_2': SparseConvTensor (32 channels),
                    'scale_3': SparseConvTensor (64 channels),  
                    'scale_4': SparseConvTensor (128 channels)
                }
            teacher_features: Dict of teacher multi-scale features (targets)
        
        Returns:
            mlfr_results: Dict containing reconstructed features and losses
                {
                    'reconstructed_features': Dict of reconstructed features,
                    'mlfr_losses': Dict of scale-wise losses,
                    'total_mlfr_loss': Weighted total loss
                }
        """
        if not self.enable_mlfr:
            return {
                'reconstructed_features': {},
                'mlfr_losses': {},
                'total_mlfr_loss': torch.tensor(0.0, device='cuda', requires_grad=True)
            }
        
        reconstructed_features = {}
        mlfr_losses = {}
        total_mlfr_loss = 0.0
        
        # 📍 Scale별 feature reconstruction
        for scale in self.mlfr_scales:
            if scale in student_features and scale in teacher_features and scale in self.scale_decoders:
                
                # Student feature에서 teacher feature 재구성
                student_tensor = student_features[scale]
                teacher_tensor = teacher_features[scale]
                
                # Decoder forward
                reconstructed_tensor = self.scale_decoders[scale](student_tensor)
                reconstructed_features[scale] = reconstructed_tensor
                
                # MSE Loss 계산 (semantic feature space에서)
                pred_features = reconstructed_tensor.features  # [N, C]
                target_features = teacher_tensor.features.detach()  # [N, C], stop gradient
                
                # Feature alignment (같은 좌표의 voxel끼리 매칭)
                pred_coords = reconstructed_tensor.indices  # [N, 4] (batch, z, y, x)
                target_coords = teacher_tensor.indices      # [M, 4]
                
                # 좌표 매칭을 통한 supervised reconstruction
                aligned_pred, aligned_target = self._align_features_by_coords(
                    pred_features, pred_coords, target_features, target_coords
                )
                
                if aligned_pred.size(0) > 0:  # 매칭되는 voxel이 있는 경우
                    scale_loss = F.mse_loss(aligned_pred, aligned_target)
                    mlfr_losses[scale] = scale_loss
                    
                    # Weight 적용
                    weighted_loss = scale_loss * self.mlfr_loss_weights.get(scale, 1.0)
                    total_mlfr_loss += weighted_loss
                    
                    print(f"   MLFR {scale}: {aligned_pred.size(0)} voxels, loss={scale_loss:.6f}")
                else:
                    mlfr_losses[scale] = torch.tensor(0.0, device='cuda', requires_grad=True)
        
        return {
            'reconstructed_features': reconstructed_features,
            'mlfr_losses': mlfr_losses, 
            'total_mlfr_loss': total_mlfr_loss
        }
    
    def _align_features_by_coords(self, pred_features, pred_coords, target_features, target_coords):
        """
        좌표 기반 feature alignment
        
        같은 (batch, z, y, x) 좌표를 가진 voxel끼리 매칭하여
        supervised reconstruction loss 계산을 위한 alignment 수행
        """
        # 좌표를 string으로 변환하여 매칭 (batch, z, y, x)
        pred_coord_keys = [f"{b}_{z}_{y}_{x}" for b, z, y, x in pred_coords.cpu().numpy()]
        target_coord_keys = [f"{b}_{z}_{y}_{x}" for b, z, y, x in target_coords.cpu().numpy()]
        
        # 공통 좌표 찾기
        pred_key_to_idx = {key: idx for idx, key in enumerate(pred_coord_keys)}
        target_key_to_idx = {key: idx for idx, key in enumerate(target_coord_keys)}
        
        common_keys = set(pred_key_to_idx.keys()) & set(target_key_to_idx.keys())
        
        if len(common_keys) == 0:
            # 매칭되는 voxel이 없으면 빈 tensor 반환
            return torch.empty((0, pred_features.size(1)), device=pred_features.device), \
                   torch.empty((0, target_features.size(1)), device=target_features.device)
        
        # 매칭되는 feature들만 선택
        pred_indices = [pred_key_to_idx[key] for key in common_keys]
        target_indices = [target_key_to_idx[key] for key in common_keys]
        
        aligned_pred = pred_features[pred_indices]
        aligned_target = target_features[target_indices]
        
        return aligned_pred, aligned_target
    
    def get_mlfr_loss_weights(self):
        """MLFR loss weight 반환 (debugging용)"""
        return self.mlfr_loss_weights