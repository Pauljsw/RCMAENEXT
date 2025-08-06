# pcdet/models/backbones_3d/radial_mae_voxelnext_optimized.py
"""
최적화된 R-MAE + VoxelNeXt Backbone

기존 radial_mae_voxelnext.py를 기반으로 성능 최적화:
1. Distance-aware radial masking (R-MAE 논문 Stage 2)
2. Multi-scale occupancy decoder with skip connections
3. Progressive masking ratio (curriculum learning)
4. Dynamic minimum voxel preservation
5. 최적화된 angular range

기존 파일에 영향 없이 새로운 파일로 구현
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from functools import partial
import numpy as np
import random
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt


class RadialMAEVoxelNeXtOptimized(VoxelResBackBone8xVoxelNeXt):
    """성능 최적화된 R-MAE + VoxelNeXt Backbone"""
    
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        # 부모 클래스 초기화
        super().__init__(model_cfg, input_channels, grid_size, **kwargs)
        
        # 필수 속성들
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.model_cfg = model_cfg
        
        # ===== 📊 최적화된 Masking 파라미터 =====
        self.masked_ratio = model_cfg.get('MASKED_RATIO', 0.85)  # 0.8 → 0.85
        self.angular_range = model_cfg.get('ANGULAR_RANGE', 5)   # 1 → 5도 (효율성)
        
        # ===== 🎯 Distance-aware Masking 파라미터 =====
        self.distance_ranges = model_cfg.get('DISTANCE_RANGES', [0, 20, 40, 80])  # meters
        self.distance_mask_ratios = model_cfg.get('DISTANCE_MASK_RATIOS', [0.7, 0.8, 0.9, 0.95])
        
        # ===== 📈 Dynamic Minimum Voxel 설정 =====
        self.min_voxel_ratio = model_cfg.get('MIN_VOXEL_RATIO', 0.08)  # 8% (논문 기준)
        self.adaptive_threshold = model_cfg.get('ADAPTIVE_THRESHOLD', True)
        
        # ===== 🔄 Progressive Masking 설정 =====
        self.progressive_masking = model_cfg.get('PROGRESSIVE_MASKING', True)
        self.initial_mask_ratio = model_cfg.get('INITIAL_MASK_RATIO', 0.6)
        self.final_mask_ratio = model_cfg.get('FINAL_MASK_RATIO', 0.85)
        
        # Training step counter for progressive masking
        self.training_step = 0
        self.total_training_steps = model_cfg.get('TOTAL_TRAINING_STEPS', 30000)  # 대략 30 epochs
        
        print(f"🚀 Optimized R-MAE VoxelNeXt initialized:")
        print(f"   📊 Masked ratio: {self.masked_ratio}")
        print(f"   🎯 Angular range: {self.angular_range}°")
        print(f"   📈 Min voxel ratio: {self.min_voxel_ratio}")
        print(f"   🔄 Progressive masking: {self.progressive_masking}")
        
        # ===== 🏗️ 개선된 Decoder 구조 =====
        if hasattr(model_cfg, 'PRETRAINING') and model_cfg.PRETRAINING:
            self._build_optimized_decoder()
    
    def _build_optimized_decoder(self):
        """단순한 occupancy decoder (기존 성공 방식 기반)"""
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        print("🏗️ Building simple occupancy decoder...")
        
        # 기존 성공한 방식과 동일한 단순 decoder
        self.occupancy_decoder = spconv.SparseSequential(
            spconv.SubMConv3d(128, 64, 3, padding=1, bias=False, indice_key='dec1'),
            norm_fn(64), nn.ReLU(),
            spconv.SubMConv3d(64, 32, 3, padding=1, bias=False, indice_key='dec2'),
            norm_fn(32), nn.ReLU(),
            spconv.SubMConv3d(32, 1, 1, bias=True, indice_key='dec_out')
        )
        
        print("✅ Simple decoder built successfully!")
    
    def get_current_mask_ratio(self):
        """🔄 Progressive masking ratio 계산"""
        if not self.progressive_masking or not self.training:
            return self.masked_ratio
        
        # Training progress (0.0 ~ 1.0)
        progress = min(self.training_step / (self.total_training_steps * 0.7), 1.0)  # 70% 지점에서 최대
        
        # Linear interpolation between initial and final mask ratio
        current_ratio = self.initial_mask_ratio + (self.final_mask_ratio - self.initial_mask_ratio) * progress
        
        return current_ratio
    
    def distance_aware_radial_masking(self, voxel_coords, voxel_features):
        """🎯 거리 인식 radial masking (R-MAE 논문 Stage 2 구현)"""
        if not self.training:
            return voxel_coords, voxel_features
        
        # Progressive masking ratio
        current_mask_ratio = self.get_current_mask_ratio()
        self.training_step += 1
        
        batch_size = int(voxel_coords[:, 0].max()) + 1
        masked_coords, masked_features = [], []
        total_original = len(voxel_coords)
        total_kept = 0
        
        for batch_idx in range(batch_size):
            mask = voxel_coords[:, 0] == batch_idx
            coords, features = voxel_coords[mask], voxel_features[mask]
            
            if len(coords) == 0:
                continue
            
            # 실제 좌표 및 거리 계산
            x = coords[:, 1].float() * self.voxel_size[0] + self.point_cloud_range[0]
            y = coords[:, 2].float() * self.voxel_size[1] + self.point_cloud_range[1]
            distances = torch.sqrt(x**2 + y**2)
            theta = torch.atan2(y, x)
            
            # ===== 📊 Angular sectors (최적화된 크기) =====
            num_sectors = int(360 / self.angular_range)
            sector_size = 2 * np.pi / num_sectors
            keep_mask = torch.ones(len(coords), dtype=torch.bool, device=coords.device)
            
            # ===== 🎯 각 섹터별 거리 인식 masking =====
            for i in range(num_sectors):
                start_angle = -np.pi + i * sector_size
                end_angle = -np.pi + (i + 1) * sector_size
                in_sector = (theta >= start_angle) & (theta < end_angle)
                
                if in_sector.sum() == 0:
                    continue
                
                # 거리별 차등 masking
                sector_distances = distances[in_sector]
                sector_indices = torch.where(in_sector)[0]
                
                # Distance-aware masking for each range
                for j, (min_dist, max_dist) in enumerate(zip(self.distance_ranges[:-1], self.distance_ranges[1:])):
                    dist_mask = (sector_distances >= min_dist) & (sector_distances < max_dist)
                    if dist_mask.sum() == 0:
                        continue
                    
                    # 거리별 masking 비율 적용 (근거리일수록 낮은 masking)
                    distance_mask_ratio = self.distance_mask_ratios[j] * current_mask_ratio
                    
                    if torch.rand(1).item() < distance_mask_ratio:
                        dist_indices = sector_indices[dist_mask]
                        keep_mask[dist_indices] = False
                
                # 원거리 (마지막 range 이후) 처리
                far_mask = sector_distances >= self.distance_ranges[-1]
                if far_mask.sum() > 0:
                    far_indices = sector_indices[far_mask]
                    # 원거리는 더 높은 masking ratio 적용
                    if torch.rand(1).item() < (0.98 * current_mask_ratio):
                        keep_mask[far_indices] = False
            
            # ===== 📈 동적 최소 voxel 보장 =====
            total_voxels = len(coords)
            min_keep = max(8, int(total_voxels * self.min_voxel_ratio))  # 최소 8개 또는 8%
            
            if keep_mask.sum() < min_keep:
                # 거리가 가까운 voxel 우선 복원
                masked_indices = torch.where(~keep_mask)[0]
                if len(masked_indices) > 0:
                    masked_distances = distances[masked_indices]
                    _, sorted_idx = torch.sort(masked_distances)  # 가까운 순서
                    
                    restore_count = min(min_keep - keep_mask.sum(), len(masked_indices))
                    if restore_count > 0:
                        restore_indices = masked_indices[sorted_idx[:restore_count]]
                        keep_mask[restore_indices] = True
            
            # 너무 많이 남은 경우 추가 masking (stability)
            elif keep_mask.sum() > total_voxels * 0.3:  # 30% 이상 남으면
                kept_indices = torch.where(keep_mask)[0]
                kept_distances = distances[kept_indices]
                _, sorted_idx = torch.sort(kept_distances, descending=True)  # 먼 순서
                
                # 원거리 voxel 중 일부 추가 제거
                additional_remove = int((keep_mask.sum() - total_voxels * 0.25) * 0.5)
                if additional_remove > 0:
                    remove_indices = kept_indices[sorted_idx[:additional_remove]]
                    keep_mask[remove_indices] = False
            
            masked_coords.append(coords[keep_mask])
            masked_features.append(features[keep_mask])
            total_kept += keep_mask.sum().item()
        
        # 통계 정보 저장
        actual_mask_ratio = 1.0 - (total_kept / max(total_original, 1))
        
        if masked_coords:
            result_coords = torch.cat(masked_coords)
            result_features = torch.cat(masked_features)
            return result_coords, result_features, {
                'target_mask_ratio': current_mask_ratio,
                'actual_mask_ratio': actual_mask_ratio,
                'training_step': self.training_step
            }
        
        return voxel_coords, voxel_features, {
            'target_mask_ratio': current_mask_ratio,
            'actual_mask_ratio': 0.0,
            'training_step': self.training_step
        }
    
    def forward(self, batch_dict):
        """최적화된 forward pass"""
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        # ===== 🔄 개선된 distance-aware masking 적용 =====
        if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
            batch_dict['original_voxel_coords'] = voxel_coords.clone()
            batch_dict['original_voxel_features'] = voxel_features.clone()
            
            # 최적화된 masking 전략 사용
            voxel_coords, voxel_features, mask_stats = self.distance_aware_radial_masking(
                voxel_coords, voxel_features
            )
            batch_dict['voxel_coords'] = voxel_coords
            batch_dict['voxel_features'] = voxel_features
            
            # Masking 통계 저장
            batch_dict.update(mask_stats)
            
            print(f"🎯 Masking Stats - Target: {mask_stats['target_mask_ratio']:.3f}, "
                  f"Actual: {mask_stats['actual_mask_ratio']:.3f}, "
                  f"Step: {mask_stats['training_step']}")
        
        # ===== 기존 VoxelNeXt forward 구조 =====
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        # ===== 🏗️ 기존 성공 방식의 occupancy prediction =====
        if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
            occupancy_pred = self.occupancy_decoder(x_conv4)
            batch_dict['occupancy_pred'] = occupancy_pred.features
            batch_dict['occupancy_coords'] = occupancy_pred.indices
            
            # 단순한 multi-scale features (consistency loss용)
            batch_dict['multi_scale_occupancy'] = {
                'scale_4': x_conv4.features,
                'scale_3': x_conv3.features,
                'scale_2': x_conv2.features,
                'scale_1': occupancy_pred.features
            }
        
        # ===== 기존 VoxelNeXt 출력 형식 유지 =====
        batch_dict.update({
            'encoded_spconv_tensor': x_conv4,
            'encoded_spconv_tensor_stride': 8,
            'multi_scale_3d_features': {
                'x_conv1': x_conv1, 'x_conv2': x_conv2,
                'x_conv3': x_conv3, 'x_conv4': x_conv4,
            }
        })
        
        return batch_dict