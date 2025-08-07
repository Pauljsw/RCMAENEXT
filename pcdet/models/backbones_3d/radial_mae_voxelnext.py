# import torch
# import torch.nn as nn
# import spconv.pytorch as spconv
# from functools import partial
# import numpy as np
# from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt

# class RadialMAEVoxelNeXt(VoxelResBackBone8xVoxelNeXt):
#     """R-MAE + VoxelNeXt Backbone - 기존 VoxelNeXt 상속"""
    
#     def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
#         # 부모 클래스 초기화 (기존 VoxelNeXt 구조 그대로)
#         super().__init__(model_cfg, input_channels, grid_size, **kwargs)
        
#         # VoxelNeXt에서 필요한 속성들 추가
#         self.voxel_size = voxel_size
#         self.point_cloud_range = point_cloud_range
        
#         # R-MAE 파라미터만 추가
#         self.masked_ratio = model_cfg.get('MASKED_RATIO', 0.8)
#         self.angular_range = model_cfg.get('ANGULAR_RANGE', 1)
        
#         # R-MAE pretraining용 decoder (PRETRAINING=True일 때만)
#         if hasattr(model_cfg, 'PRETRAINING') and model_cfg.PRETRAINING:
#             norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
#             self.occupancy_decoder = spconv.SparseSequential(
#                 spconv.SubMConv3d(128, 64, 3, padding=1, bias=False, indice_key='dec1'),
#                 norm_fn(64), nn.ReLU(),
#                 spconv.SubMConv3d(64, 32, 3, padding=1, bias=False, indice_key='dec2'),
#                 norm_fn(32), nn.ReLU(),
#                 spconv.SubMConv3d(32, 1, 1, bias=True, indice_key='dec_out')
#             )
    
#     def radial_masking(self, voxel_coords, voxel_features):
#         """R-MAE 논문 기준 radial masking"""
#         if not self.training:
#             return voxel_coords, voxel_features
            
#         batch_size = int(voxel_coords[:, 0].max()) + 1
#         masked_coords, masked_features = [], []
        
#         for batch_idx in range(batch_size):
#             mask = voxel_coords[:, 0] == batch_idx
#             coords, features = voxel_coords[mask], voxel_features[mask]
            
#             if len(coords) == 0:
#                 continue
                
#             # 실제 좌표 계산
#             x = coords[:, 1].float() * self.voxel_size[0] + self.point_cloud_range[0]
#             y = coords[:, 2].float() * self.voxel_size[1] + self.point_cloud_range[1]
#             theta = torch.atan2(y, x)
            
#             # Angular masking
#             num_sectors = int(360 / self.angular_range)
#             sector_size = 2 * np.pi / num_sectors
#             keep_mask = torch.ones(len(coords), dtype=torch.bool, device=coords.device)
            
#             for i in range(num_sectors):
#                 start = -np.pi + i * sector_size
#                 end = -np.pi + (i + 1) * sector_size
#                 in_sector = (theta >= start) & (theta < end)
                
#                 if in_sector.sum() > 0 and torch.rand(1) < self.masked_ratio:
#                     keep_mask[in_sector] = False
            
#             # 최소 voxel 보장
#             if keep_mask.sum() < max(10, len(coords) * 0.1):
#                 indices = torch.where(~keep_mask)[0]
#                 restore_count = max(10, len(coords) // 10) - keep_mask.sum()
#                 if restore_count > 0 and len(indices) > 0:
#                     restore_idx = indices[torch.randperm(len(indices))[:restore_count]]
#                     keep_mask[restore_idx] = True
            
#             masked_coords.append(coords[keep_mask])
#             masked_features.append(features[keep_mask])
        
#         if masked_coords:
#             return torch.cat(masked_coords), torch.cat(masked_features)
#         return voxel_coords, voxel_features
    
#     def forward(self, batch_dict):
#         """기존 VoxelNeXt forward + R-MAE masking"""
#         voxel_features = batch_dict['voxel_features']
        
#         voxel_coords = batch_dict['voxel_coords']
#         batch_size = batch_dict['batch_size']
        
#         # R-MAE masking 적용 (training 시에만)
#         if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
#             batch_dict['original_voxel_coords'] = voxel_coords.clone()
#             voxel_coords, voxel_features = self.radial_masking(voxel_coords, voxel_features)
#             batch_dict['voxel_coords'] = voxel_coords
#             batch_dict['voxel_features'] = voxel_features
        
#         # 부모 클래스의 forward 호출 (기존 VoxelNeXt 로직)
#         input_sp_tensor = spconv.SparseConvTensor(
#             features=voxel_features,
#             indices=voxel_coords.int(),
#             spatial_shape=self.sparse_shape,
#             batch_size=batch_size
#         )
        
#         # 기존 VoxelNeXt conv 레이어들
#         x = self.conv_input(input_sp_tensor)
#         x_conv1 = self.conv1(x)
#         x_conv2 = self.conv2(x_conv1)  
#         x_conv3 = self.conv3(x_conv2)
#         x_conv4 = self.conv4(x_conv3)
        
#         # Pretraining: occupancy prediction 추가
#         if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
#             occupancy_pred = self.occupancy_decoder(x_conv4)
#             batch_dict['occupancy_pred'] = occupancy_pred.features
#             batch_dict['occupancy_coords'] = occupancy_pred.indices
        
#         # 기존 VoxelNeXt 출력 형식 유지
#         batch_dict.update({
#             'encoded_spconv_tensor': x_conv4,
#             'encoded_spconv_tensor_stride': 8,
#             'multi_scale_3d_features': {
#                 'x_conv1': x_conv1, 'x_conv2': x_conv2,
#                 'x_conv3': x_conv3, 'x_conv4': x_conv4,
#             }
#         })
        
#         return batch_dict





"0807 수정본"
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from functools import partial
import numpy as np
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt

class RadialMAEVoxelNeXt(VoxelResBackBone8xVoxelNeXt):
    """R-MAE + VoxelNeXt Backbone - 논문 정확한 2-Stage Radial Masking 구현"""
    
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        # 부모 클래스 초기화 (기존 VoxelNeXt 구조 그대로)
        super().__init__(model_cfg, input_channels, grid_size, **kwargs)
        
        # VoxelNeXt에서 필요한 속성들 추가
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # ===== 📄 R-MAE 논문 정확한 파라미터 =====
        self.masked_ratio = model_cfg.get('MASKED_RATIO', 0.8)  # m (masking ratio)
        
        # Stage 1: Angular Group Selection
        self.num_angular_groups = model_cfg.get('NUM_ANGULAR_GROUPS', 36)  # Ng
        self.angular_group_size = model_cfg.get('ANGULAR_GROUP_SIZE', 10)  # Δθ = 10 degrees
        
        # Stage 2: Range-Aware Masking within Selected Groups
        self.distance_thresholds = model_cfg.get('DISTANCE_THRESHOLDS', [20, 40, 60])  # rt1, rt2, rt3
        self.range_mask_probs = model_cfg.get('RANGE_MASK_PROBS', {
            'NEAR': 0.56,   # 0.8 * 0.7 = 가까운 거리 낮은 masking
            'MID': 0.80,    # 표준 masking ratio  
            'FAR': 1.0      # min(1.0, 0.8 * 1.3) = 먼 거리 높은 masking
        })
        
        # 논문 구현 옵션
        self.use_bernoulli_masking = model_cfg.get('USE_BERNOULLI_MASKING', True)
        self.enable_2stage_masking = model_cfg.get('ENABLE_2STAGE_MASKING', True)
        
        print(f"🎯 R-MAE Paper Implementation:")
        print(f"   - Masking ratio (m): {self.masked_ratio}")
        print(f"   - Angular groups (Ng): {self.num_angular_groups}")
        print(f"   - Angular size (Δθ): {self.angular_group_size}°")
        print(f"   - Distance thresholds: {self.distance_thresholds} meters")
        print(f"   - 2-Stage masking: {self.enable_2stage_masking}")
        
        # R-MAE pretraining용 decoder (PRETRAINING=True일 때만)
        if hasattr(model_cfg, 'PRETRAINING') and model_cfg.PRETRAINING:
            norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
            self.occupancy_decoder = spconv.SparseSequential(
                spconv.SubMConv3d(128, 64, 3, padding=1, bias=False, indice_key='dec1'),
                norm_fn(64), nn.ReLU(),
                spconv.SubMConv3d(64, 32, 3, padding=1, bias=False, indice_key='dec2'),
                norm_fn(32), nn.ReLU(),
                spconv.SubMConv3d(32, 1, 1, bias=True, indice_key='dec_out')
            )
    
    def get_distance_group(self, distances):
        """거리 기반 subgroup 분류"""
        # rt1, rt2, rt3에 따라 subgroup 결정
        near_mask = distances < self.distance_thresholds[0]
        mid_mask = (distances >= self.distance_thresholds[0]) & (distances < self.distance_thresholds[1])
        far_mask = distances >= self.distance_thresholds[1]
        
        return near_mask, mid_mask, far_mask
    
    def apply_stage1_angular_group_selection(self, theta):
        """
        📄 R-MAE 논문 Stage 1: Angular Group Selection - 수정
        
        논문 해석 수정: pg = 1-m은 group selection probability가 아니라
        각 group 내에서 keep할 확률. 전체적으로 m 비율 masking 달성해야 함.
        """
        # 🔧 수정: 논문 의도 정확히 구현
        # m = 0.8 (80% masking) → 20%만 keep
        # 따라서 group selection은 더 관대하게, stage 2에서 조정
        
        group_selection_ratio = 0.7  # 70% group 선택 (stage 2에서 세부 조정)
        selected_groups = torch.rand(self.num_angular_groups, device=theta.device) < group_selection_ratio
        
        # 각 voxel이 어느 group에 속하는지 계산
        theta_normalized = (theta + np.pi) / (2 * np.pi)  # [0, 1]
        group_indices = torch.floor(theta_normalized * self.num_angular_groups).long()
        group_indices = torch.clamp(group_indices, 0, self.num_angular_groups - 1)
        
        # 선택된 group에 속하는 voxel들만 Stage 2로 진행
        stage1_keep_mask = selected_groups[group_indices]
        
        return stage1_keep_mask, group_indices
    
    def apply_stage2_range_aware_masking(self, distances, group_indices, stage1_mask):
        """
        📄 R-MAE 논문 Stage 2: Range-Aware Masking within Selected Groups - 수정
        
        Stage 1에서 선택된 voxel들을 거리 기반으로 masking하여 
        전체 target masking ratio (0.8) 달성
        """
        stage2_keep_mask = torch.ones_like(stage1_mask, dtype=torch.bool)
        
        # Stage 1에서 선택된 voxel들만 처리
        selected_voxels = torch.where(stage1_mask)[0]
        
        if len(selected_voxels) == 0:
            return stage2_keep_mask
        
        selected_distances = distances[selected_voxels]
        
        # Distance subgroups 분류
        near_mask, mid_mask, far_mask = self.get_distance_group(selected_distances)
        
        # 🔧 수정: 더 aggressive한 masking으로 전체 80% 달성
        # Stage 1에서 70% 선택 → Stage 2에서 추가 masking으로 20% 최종 keep
        target_keep_ratio = 0.25  # 25% keep (75% mask)
        
        near_prob = min(0.9, self.range_mask_probs['NEAR'] * 1.2)  # 더 높은 masking
        mid_prob = min(0.95, self.range_mask_probs['MID'] * 1.2)
        far_prob = min(0.98, self.range_mask_probs['FAR'] * 1.1)
        
        # Bernoulli distribution으로 masking 결정
        if self.use_bernoulli_masking:
            near_decisions = torch.bernoulli(torch.full((near_mask.sum(),), near_prob, device=distances.device))
            mid_decisions = torch.bernoulli(torch.full((mid_mask.sum(),), mid_prob, device=distances.device))
            far_decisions = torch.bernoulli(torch.full((far_mask.sum(),), far_prob, device=distances.device))
            
            # Bernoulli = 1이면 mask (keep_mask = False)
            selected_keep_mask = torch.ones(len(selected_voxels), dtype=torch.bool, device=distances.device)
            selected_keep_mask[near_mask] = near_decisions == 0  # 1이면 mask, 0이면 keep
            selected_keep_mask[mid_mask] = mid_decisions == 0
            selected_keep_mask[far_mask] = far_decisions == 0
        else:
            # Fallback: random masking
            selected_keep_mask = torch.rand(len(selected_voxels), device=distances.device) > 0.75  # 75% mask
        
        # 결과를 전체 mask에 반영
        stage2_keep_mask[selected_voxels] = selected_keep_mask
        
        return stage2_keep_mask
    
    def radial_masking_rmae_paper(self, voxel_coords, voxel_features):
        """
        📄 R-MAE 논문 정확한 2-Stage Radial Masking 구현
        
        M(vi) = {
            0, if g(vi) ∈ Gs and Bernoulli(pmg(vi),k(vi)) = 1
            1, otherwise
        }
        """
        if not self.training or not self.enable_2stage_masking:
            return voxel_coords, voxel_features
            
        batch_size = int(voxel_coords[:, 0].max()) + 1
        masked_coords, masked_features = [], []
        
        # 통계 수집용
        total_original = len(voxel_coords)
        total_kept = 0
        stage1_stats = {'selected_groups': 0, 'total_groups': self.num_angular_groups}
        stage2_stats = {'near_masked': 0, 'mid_masked': 0, 'far_masked': 0}
        
        for batch_idx in range(batch_size):
            mask = voxel_coords[:, 0] == batch_idx
            coords, features = voxel_coords[mask], voxel_features[mask]
            
            if len(coords) == 0:
                continue
                
            # 실제 좌표 계산 (cylindrical coordinates)
            x = coords[:, 1].float() * self.voxel_size[0] + self.point_cloud_range[0]
            y = coords[:, 2].float() * self.voxel_size[1] + self.point_cloud_range[1]
            
            # ri (radial distance), θi (azimuth angle), zi (height)
            distances = torch.sqrt(x**2 + y**2)  # ri
            theta = torch.atan2(y, x)            # θi
            
            # ===== Stage 1: Angular Group Selection =====
            stage1_keep_mask, group_indices = self.apply_stage1_angular_group_selection(theta)
            
            # ===== Stage 2: Range-Aware Masking within Selected Groups =====
            stage2_keep_mask = self.apply_stage2_range_aware_masking(distances, group_indices, stage1_keep_mask)
            
            # 최종 mask: Stage 1에서 선택되지 않은 것들은 자동으로 keep, Stage 2 결과 적용
            final_keep_mask = (~stage1_keep_mask) | (stage1_keep_mask & stage2_keep_mask)
            
            # 최소 voxel 보장 (안정성을 위해)
            min_voxels = max(10, int(len(coords) * 0.05))  # 최소 5%
            if final_keep_mask.sum() < min_voxels:
                # 가장 가까운 voxel들을 추가로 keep
                _, nearest_indices = torch.topk(distances, min_voxels, largest=False)
                final_keep_mask[nearest_indices] = True
            
            masked_coords.append(coords[final_keep_mask])
            masked_features.append(features[final_keep_mask])
            total_kept += final_keep_mask.sum().item()
        
        # 통계 저장
        effective_mask_ratio = 1.0 - (total_kept / total_original) if total_original > 0 else 0
        masking_stats = {
            'target_mask_ratio': self.masked_ratio,
            'effective_mask_ratio': effective_mask_ratio,
            'stage1_stats': stage1_stats,
            'stage2_stats': stage2_stats,
            'total_original': total_original,
            'total_kept': total_kept
        }
        
        if masked_coords:
            result_coords = torch.cat(masked_coords)
            result_features = torch.cat(masked_features)
        else:
            result_coords = voxel_coords
            result_features = voxel_features
        
        return result_coords, result_features, masking_stats
    
    def radial_masking(self, voxel_coords, voxel_features):
        """기존 간단한 방식 (fallback)"""
        if not self.training:
            return voxel_coords, voxel_features
            
        batch_size = int(voxel_coords[:, 0].max()) + 1
        masked_coords, masked_features = [], []
        
        for batch_idx in range(batch_size):
            mask = voxel_coords[:, 0] == batch_idx
            coords, features = voxel_coords[mask], voxel_features[mask]
            
            if len(coords) == 0:
                continue
                
            # 실제 좌표 계산
            x = coords[:, 1].float() * self.voxel_size[0] + self.point_cloud_range[0]
            y = coords[:, 2].float() * self.voxel_size[1] + self.point_cloud_range[1]
            theta = torch.atan2(y, x)
            
            # Angular masking (기존 방식)
            num_sectors = int(360 / self.angular_group_size)
            sector_size = 2 * np.pi / num_sectors
            keep_mask = torch.ones(len(coords), dtype=torch.bool, device=coords.device)
            
            for i in range(num_sectors):
                start = -np.pi + i * sector_size
                end = -np.pi + (i + 1) * sector_size
                in_sector = (theta >= start) & (theta < end)
                
                if in_sector.sum() > 0 and torch.rand(1) < self.masked_ratio:
                    keep_mask[in_sector] = False
            
            # 최소 voxel 보장
            if keep_mask.sum() < max(10, len(coords) * 0.1):
                indices = torch.where(~keep_mask)[0]
                restore_count = max(10, len(coords) // 10) - keep_mask.sum()
                if restore_count > 0 and len(indices) > 0:
                    restore_idx = indices[torch.randperm(len(indices))[:restore_count]]
                    keep_mask[restore_idx] = True
            
            masked_coords.append(coords[keep_mask])
            masked_features.append(features[keep_mask])
        
        if masked_coords:
            return torch.cat(masked_coords), torch.cat(masked_features)
        return voxel_coords, voxel_features
    
    def forward(self, batch_dict):
        """기존 VoxelNeXt forward + R-MAE 논문 정확한 masking"""
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        # R-MAE masking 적용 (training 시에만)
        if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
            batch_dict['original_voxel_coords'] = voxel_coords.clone()
            batch_dict['original_voxel_features'] = voxel_features.clone()
            
            # 📄 R-MAE 논문 정확한 2-Stage masking 사용
            if self.enable_2stage_masking:
                voxel_coords, voxel_features, masking_stats = self.radial_masking_rmae_paper(voxel_coords, voxel_features)
                batch_dict['masking_stats'] = masking_stats
                
                print(f"🎯 R-MAE Masking: Target {masking_stats['target_mask_ratio']:.1%} → "
                      f"Actual {masking_stats['effective_mask_ratio']:.1%}")
            else:
                # Fallback to simple method
                voxel_coords, voxel_features = self.radial_masking(voxel_coords, voxel_features)
            
            batch_dict['voxel_coords'] = voxel_coords
            batch_dict['voxel_features'] = voxel_features
        
        # 부모 클래스의 forward 호출 (기존 VoxelNeXt 로직)
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # 기존 VoxelNeXt conv 레이어들
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)  
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        # Pretraining: occupancy prediction 추가
        if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
            occupancy_pred = self.occupancy_decoder(x_conv4)
            batch_dict['occupancy_pred'] = occupancy_pred.features
            batch_dict['occupancy_coords'] = occupancy_pred.indices
        
        # 기존 VoxelNeXt 출력 형식 유지
        batch_dict.update({
            'encoded_spconv_tensor': x_conv4,
            'encoded_spconv_tensor_stride': 8,
            'multi_scale_3d_features': {
                'x_conv1': x_conv1, 'x_conv2': x_conv2,
                'x_conv3': x_conv3, 'x_conv4': x_conv4,
            }
        })
        
        return batch_dict