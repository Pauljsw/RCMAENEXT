import torch
import torch.nn as nn
import spconv.pytorch as spconv
from functools import partial
import numpy as np
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt

class RadialMAEVoxelNeXt(VoxelResBackBone8xVoxelNeXt):
    """R-MAE + VoxelNeXt Backbone - 기존 VoxelNeXt 상속"""
    
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        # 부모 클래스 초기화 (기존 VoxelNeXt 구조 그대로)
        super().__init__(model_cfg, input_channels, grid_size, **kwargs)
        
        # VoxelNeXt에서 필요한 속성들 추가
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # R-MAE 파라미터만 추가
        self.masked_ratio = model_cfg.get('MASKED_RATIO', 0.8)
        self.angular_range = model_cfg.get('ANGULAR_RANGE', 1)
        
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
    
    def radial_masking(self, voxel_coords, voxel_features):
        """R-MAE 논문 기준 radial masking"""
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
            
            # Angular masking
            num_sectors = int(360 / self.angular_range)
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
        """기존 VoxelNeXt forward + R-MAE masking"""
        voxel_features = batch_dict['voxel_features']
        
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        # R-MAE masking 적용 (training 시에만)
        if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
            batch_dict['original_voxel_coords'] = voxel_coords.clone()
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





# "0807 수정본"
# import torch
# import torch.nn as nn
# import spconv.pytorch as spconv
# from functools import partial
# import numpy as np
# from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt

# class RadialMAEVoxelNeXt(VoxelResBackBone8xVoxelNeXt):
#     """R-MAE + VoxelNeXt Backbone - 논문 정확한 2-Stage Radial Masking 구현"""
    
#     def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
#         # 부모 클래스 초기화 (기존 VoxelNeXt 구조 그대로)
#         super().__init__(model_cfg, input_channels, grid_size, **kwargs)
        
#         # VoxelNeXt에서 필요한 속성들 추가
#         self.voxel_size = voxel_size
#         self.point_cloud_range = point_cloud_range
        
#         # ===== 📄 R-MAE 논문 정확한 파라미터 =====
#         self.masked_ratio = model_cfg.get('MASKED_RATIO', 0.8)  # m (masking ratio)
        
#         # Stage 1: Angular Group Selection
#         self.num_angular_groups = model_cfg.get('NUM_ANGULAR_GROUPS', 36)  # Ng
#         self.angular_group_size = model_cfg.get('ANGULAR_GROUP_SIZE', 10)  # Δθ = 10 degrees
        
#         # Stage 2: Range-Aware Masking within Selected Groups
#         self.distance_thresholds = model_cfg.get('DISTANCE_THRESHOLDS', [20, 40, 60])  # rt1, rt2, rt3
#         self.range_mask_probs = model_cfg.get('RANGE_MASK_PROBS', {
#             'NEAR': 0.56,   # 0.8 * 0.7 = 가까운 거리 낮은 masking
#             'MID': 0.80,    # 표준 masking ratio  
#             'FAR': 1.0      # min(1.0, 0.8 * 1.3) = 먼 거리 높은 masking
#         })
        
#         # 논문 구현 옵션
#         self.use_bernoulli_masking = model_cfg.get('USE_BERNOULLI_MASKING', True)
#         self.enable_2stage_masking = model_cfg.get('ENABLE_2STAGE_MASKING', True)
        
#         print(f"🎯 R-MAE Paper Implementation:")
#         print(f"   - Masking ratio (m): {self.masked_ratio}")
#         print(f"   - Angular groups (Ng): {self.num_angular_groups}")
#         print(f"   - Angular size (Δθ): {self.angular_group_size}°")
#         print(f"   - Distance thresholds: {self.distance_thresholds} meters")
#         print(f"   - 2-Stage masking: {self.enable_2stage_masking}")
        
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
    
#     def get_distance_group(self, distances):
#         """
#         🔧 수정: 원점 기준 거리를 사용한 near/mid/far 분류
#         """
#         # 당신의 실제 데이터 분포에 맞는 고정 임계값 사용
#         near_threshold = 10.0   # Near: <= 15m (63.80%)
#         mid_threshold = 30.0    # Mid: 15~50m (35.65%), Far: > 50m (0.56%)
        
#         near_mask = distances <= near_threshold
#         mid_mask = (distances > near_threshold) & (distances <= mid_threshold) 
#         far_mask = distances > mid_threshold
        
#         return near_mask, mid_mask, far_mask
    
#     def apply_stage1_angular_group_selection(self, theta):
#         """
#         📄 R-MAE 논문 Stage 1: Angular Group Selection - 수정
        
#         논문 해석 수정: pg = 1-m은 group selection probability가 아니라
#         각 group 내에서 keep할 확률. 전체적으로 m 비율 masking 달성해야 함.
#         """
#         # 🔧 수정: 논문 의도 정확히 구현
#         # m = 0.8 (80% masking) → 20%만 keep
#         # 따라서 group selection은 더 관대하게, stage 2에서 조정
        
#         group_selection_ratio = 0.7  # 70% group 선택 (stage 2에서 세부 조정)
#         selected_groups = torch.rand(self.num_angular_groups, device=theta.device) < group_selection_ratio
        
#         # 각 voxel이 어느 group에 속하는지 계산
#         theta_normalized = (theta + np.pi) / (2 * np.pi)  # [0, 1]
#         group_indices = torch.floor(theta_normalized * self.num_angular_groups).long()
#         group_indices = torch.clamp(group_indices, 0, self.num_angular_groups - 1)
        
#         # 선택된 group에 속하는 voxel들만 Stage 2로 진행
#         stage1_keep_mask = selected_groups[group_indices]
        
#         return stage1_keep_mask, group_indices
    
#     def apply_stage2_range_aware_masking(self, distances, group_indices, stage1_mask):
#         """
#         📄 R-MAE 논문 Stage 2: Range-Aware Masking within Selected Groups - 수정
        
#         Stage 1에서 선택된 voxel들을 거리 기반으로 masking하여 
#         전체 target masking ratio (0.8) 달성
#         """
#         stage2_keep_mask = torch.ones_like(stage1_mask, dtype=torch.bool)
        
#         # Stage 1에서 선택된 voxel들만 처리
#         selected_voxels = torch.where(stage1_mask)[0]
        
#         if len(selected_voxels) == 0:
#             return stage2_keep_mask
        
#         selected_distances = distances[selected_voxels]
        
#         # Distance subgroups 분류
#         near_mask, mid_mask, far_mask = self.get_distance_group(selected_distances)
        
#         # 🔧 수정: 더 aggressive한 masking으로 전체 80% 달성
#         # Stage 1에서 70% 선택 → Stage 2에서 추가 masking으로 20% 최종 keep
#         target_keep_ratio = 0.25  # 25% keep (75% mask)
        
#         near_prob = min(0.9, self.range_mask_probs['NEAR'] * 1.2)  # 더 높은 masking
#         mid_prob = min(0.95, self.range_mask_probs['MID'] * 1.2)
#         far_prob = min(0.98, self.range_mask_probs['FAR'] * 1.1)
        
#         # Bernoulli distribution으로 masking 결정
#         if self.use_bernoulli_masking:
#             near_decisions = torch.bernoulli(torch.full((near_mask.sum(),), near_prob, device=distances.device))
#             mid_decisions = torch.bernoulli(torch.full((mid_mask.sum(),), mid_prob, device=distances.device))
#             far_decisions = torch.bernoulli(torch.full((far_mask.sum(),), far_prob, device=distances.device))
            
#             # Bernoulli = 1이면 mask (keep_mask = False)
#             selected_keep_mask = torch.ones(len(selected_voxels), dtype=torch.bool, device=distances.device)
#             selected_keep_mask[near_mask] = near_decisions == 0  # 1이면 mask, 0이면 keep
#             selected_keep_mask[mid_mask] = mid_decisions == 0
#             selected_keep_mask[far_mask] = far_decisions == 0
#         else:
#             # Fallback: random masking
#             selected_keep_mask = torch.rand(len(selected_voxels), device=distances.device) > 0.75  # 75% mask
        
#         # 결과를 전체 mask에 반영
#         stage2_keep_mask[selected_voxels] = selected_keep_mask
        
#         return stage2_keep_mask
    
#     def radial_masking_rmae_paper(self, voxel_coords, voxel_features):
#         """
#         🎯 점군 직접 사용: batch_dict['points']로 거리 계산 후 voxel masking
#         """
#         if not self.training or not self.enable_2stage_masking:
#             return voxel_coords, voxel_features
            
#         batch_size = int(voxel_coords[:, 0].max()) + 1
#         masked_coords, masked_features = [], []
        
#         for batch_idx in range(batch_size):
#             mask = voxel_coords[:, 0] == batch_idx
#             coords, features = voxel_coords[mask], voxel_features[mask]
            
#             if len(coords) == 0:
#                 continue
            
#             # 🎯 Voxel 좌표를 실제 좌표로 변환 (단순하게)
#             # 각 voxel의 중심점 계산
#             voxel_x = coords[:, 1].float() * self.voxel_size[0] + self.point_cloud_range[0] + self.voxel_size[0] * 0.5
#             voxel_y = coords[:, 2].float() * self.voxel_size[1] + self.point_cloud_range[1] + self.voxel_size[1] * 0.5
#             voxel_z = coords[:, 3].float() * self.voxel_size[2] + self.point_cloud_range[2] + self.voxel_size[2] * 0.5
            
#             # 🎯 점군에서 했던 것과 동일한 거리 계산
#             distances = torch.sqrt(voxel_x**2 + voxel_y**2 + voxel_z**2)
            
#             # Stage 1: Angular Group Selection
#             theta = torch.atan2(voxel_y, voxel_x)
#             stage1_keep_mask, group_indices = self.apply_stage1_angular_group_selection(theta)
            
#             # Stage 2: Range-Aware Masking
#             stage2_keep_mask = self.apply_stage2_range_aware_masking(distances, group_indices, stage1_keep_mask)
            
#             # 결과 수집
#             final_coords = coords[stage2_keep_mask]
#             final_features = features[stage2_keep_mask]
            
#             if len(final_coords) > 0:
#                 masked_coords.append(final_coords)
#                 masked_features.append(final_features)
            
#             # 🔍 디버그 출력 (첫 배치만)
#             if batch_idx == 0:
#                 near_mask, mid_mask, far_mask = self.get_distance_group(distances)
                
#                 print(f"🎯 Direct Voxel Center Distance:")
#                 print(f"   - voxel_size: {self.voxel_size}")
#                 print(f"   - point_cloud_range: {self.point_cloud_range}")
#                 print(f"   - Voxel center coords range:")
#                 print(f"     - X: {voxel_x.min():.2f} ~ {voxel_x.max():.2f}")
#                 print(f"     - Y: {voxel_y.min():.2f} ~ {voxel_y.max():.2f}")
#                 print(f"     - Z: {voxel_z.min():.2f} ~ {voxel_z.max():.2f}")
#                 print(f"   - Distance range: {distances.min():.2f} ~ {distances.max():.2f}m")
#                 print(f"   - Near voxels (≤10m): {near_mask.sum()} ({near_mask.sum()/len(coords)*100:.1f}%)")
#                 print(f"   - Mid voxels (10~30m): {mid_mask.sum()} ({mid_mask.sum()/len(coords)*100:.1f}%)")
#                 print(f"   - Far voxels (>30m): {far_mask.sum()} ({far_mask.sum()/len(coords)*100:.1f}%)")
        
#         # 반환
#         if masked_coords:
#             return torch.cat(masked_coords, dim=0), torch.cat(masked_features, dim=0)
#         else:
#             return torch.empty((0, 4), dtype=voxel_coords.dtype, device=voxel_coords.device), \
#                 torch.empty((0, voxel_features.shape[1]), dtype=voxel_features.dtype, device=voxel_features.device)


#     # pcdet/models/backbones_3d/radial_mae_voxelnext.py 수정

#     # pcdet/models/backbones_3d/radial_mae_voxelnext.py 수정

#     # pcdet/models/backbones_3d/radial_mae_voxelnext.py 수정

#     def classify_voxels_accurate_distance(self, voxel_coords, voxel_features, batch_dict):
#         """
#         🎯 정확한 Voxel 분류: 각 voxel 내 점들의 평균 거리로 분류
#         """
#         if not self.training or not self.enable_2stage_masking:
#             return voxel_coords, voxel_features
            
#         if 'points' not in batch_dict:
#             return voxel_coords, voxel_features
            
#         points = batch_dict['points']  # [N, 4] (batch_idx, x, y, z)
#         batch_size = int(voxel_coords[:, 0].max()) + 1
#         masked_coords, masked_features = [], []
        
#         for batch_idx in range(batch_size):
#             # 해당 배치의 voxel과 점군 가져오기
#             voxel_mask = voxel_coords[:, 0] == batch_idx
#             coords, features = voxel_coords[voxel_mask], voxel_features[voxel_mask]
            
#             point_mask = points[:, 0] == batch_idx
#             batch_points = points[point_mask]
            
#             if len(coords) == 0 or len(batch_points) == 0:
#                 continue
            
#             # 🎯 점군의 거리 계산
#             point_x = batch_points[:, 1]
#             point_y = batch_points[:, 2]
#             point_z = batch_points[:, 3]
#             point_distances = torch.sqrt(point_x**2 + point_y**2 + point_z**2)
            
#             # 🎯 각 점이 속한 voxel 찾기 (올바른 좌표 순서 사용)
#             point_voxel_x = torch.floor((batch_points[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]).long()
#             point_voxel_y = torch.floor((batch_points[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]).long()
#             point_voxel_z = torch.floor((batch_points[:, 3] - self.point_cloud_range[2]) / self.voxel_size[2]).long()
            
#             # 🎯 각 voxel의 평균 거리 계산
#             voxel_avg_distances = torch.zeros(len(coords), device=coords.device)
#             matched_voxels = 0
            
#             for i, voxel_coord in enumerate(coords):
#                 # voxel 좌표 (z, y, x 순서)
#                 voxel_z = voxel_coord[1].item()
#                 voxel_y = voxel_coord[2].item()
#                 voxel_x = voxel_coord[3].item()
                
#                 # 해당 voxel에 속하는 점들 찾기
#                 points_in_voxel = (point_voxel_x == voxel_x) & (point_voxel_y == voxel_y) & (point_voxel_z == voxel_z)
                
#                 if points_in_voxel.sum() > 0:
#                     matched_voxels += 1
#                     # 🎯 해당 voxel에 속하는 점들의 평균 거리 사용
#                     voxel_avg_distances[i] = point_distances[points_in_voxel].mean()
#                 else:
#                     # 매칭되지 않는 voxel: voxel 중심점의 거리 사용
#                     voxel_center_x = voxel_x * self.voxel_size[0] + self.point_cloud_range[0] + self.voxel_size[0] * 0.5
#                     voxel_center_y = voxel_y * self.voxel_size[1] + self.point_cloud_range[1] + self.voxel_size[1] * 0.5
#                     voxel_center_z = voxel_z * self.voxel_size[2] + self.point_cloud_range[2] + self.voxel_size[2] * 0.5
                    
#                     # 🔧 Tensor로 변환
#                     voxel_center_x = torch.tensor(voxel_center_x, device=coords.device)
#                     voxel_center_y = torch.tensor(voxel_center_y, device=coords.device)
#                     voxel_center_z = torch.tensor(voxel_center_z, device=coords.device)
                    
#                     voxel_center_distance = torch.sqrt(voxel_center_x**2 + voxel_center_y**2 + voxel_center_z**2)
#                     voxel_avg_distances[i] = voxel_center_distance
            
#             # 🎯 평균 거리 기반으로 voxel 분류
#             near_voxels = voxel_avg_distances <= 10.0
#             mid_voxels = (voxel_avg_distances > 10.0) & (voxel_avg_distances <= 30.0)
#             far_voxels = voxel_avg_distances > 30.0
            
#             # 🎯 분류별 masking 확률 적용
#             keep_mask = torch.ones(len(coords), dtype=torch.bool, device=coords.device)
            
#             # 거리별 masking 확률
#             near_keep_prob = 1.0 - 0.50  # 50% mask
#             mid_keep_prob = 1.0 - 0.75   # 75% mask  
#             far_keep_prob = 1.0 - 0.90   # 90% mask
            
#             # 각 그룹별로 랜덤 masking
#             if near_voxels.sum() > 0:
#                 near_keep = torch.rand(near_voxels.sum(), device=coords.device) < near_keep_prob
#                 keep_mask[near_voxels] = near_keep
            
#             if mid_voxels.sum() > 0:
#                 mid_keep = torch.rand(mid_voxels.sum(), device=coords.device) < mid_keep_prob
#                 keep_mask[mid_voxels] = mid_keep
            
#             if far_voxels.sum() > 0:
#                 far_keep = torch.rand(far_voxels.sum(), device=coords.device) < far_keep_prob
#                 keep_mask[far_voxels] = far_keep
            
#             # 결과 수집
#             final_coords = coords[keep_mask]
#             final_features = features[keep_mask]
            
#             if len(final_coords) > 0:
#                 masked_coords.append(final_coords)
#                 masked_features.append(final_features)
            
#             # 🔍 상세 비교 출력 (첫 배치만)
#             if batch_idx == 0:
#                 # 점군 분포 다시 계산
#                 point_near = (point_distances <= 10.0).sum()
#                 point_mid = ((point_distances > 10.0) & (point_distances <= 30.0)).sum()
#                 point_far = (point_distances > 30.0).sum()
                
#                 print(f"🔍 Detailed Distribution Comparison:")
#                 print(f"   📊 Point Distribution:")
#                 print(f"      - Near (≤10m): {point_near} ({point_near/len(batch_points)*100:.1f}%)")
#                 print(f"      - Mid (10~30m): {point_mid} ({point_mid/len(batch_points)*100:.1f}%)")
#                 print(f"      - Far (>30m): {point_far} ({point_far/len(batch_points)*100:.1f}%)")
                
#                 print(f"   📊 Voxel Distribution (Average Distance):")
#                 print(f"      - Near (≤10m): {near_voxels.sum()} ({near_voxels.sum()/len(coords)*100:.1f}%)")
#                 print(f"      - Mid (10~30m): {mid_voxels.sum()} ({mid_voxels.sum()/len(coords)*100:.1f}%)")
#                 print(f"      - Far (>30m): {far_voxels.sum()} ({far_voxels.sum()/len(coords)*100:.1f}%)")
                
#                 # 차이 계산
#                 near_diff = abs(point_near/len(batch_points)*100 - near_voxels.sum()/len(coords)*100)
#                 mid_diff = abs(point_mid/len(batch_points)*100 - mid_voxels.sum()/len(coords)*100)
#                 far_diff = abs(point_far/len(batch_points)*100 - far_voxels.sum()/len(coords)*100)
                
#                 print(f"   📊 Distribution Difference:")
#                 print(f"      - Near difference: {near_diff:.1f}%")
#                 print(f"      - Mid difference: {mid_diff:.1f}%")
#                 print(f"      - Far difference: {far_diff:.1f}%")
#                 print(f"      - Total difference: {(near_diff + mid_diff + far_diff):.1f}%")
                
#                 print(f"   📊 Matching Info:")
#                 print(f"      - Matched voxels: {matched_voxels}/{len(coords)} ({matched_voxels/len(coords)*100:.1f}%)")
#                 print(f"      - Final kept voxels: {len(final_coords)} (target: {(1-self.masked_ratio)*100:.1f}%)")
            
#             break  # 첫 번째 배치만 처리 (디버깅용)
        
#         # 반환
#         if masked_coords:
#             return torch.cat(masked_coords, dim=0), torch.cat(masked_features, dim=0)
#         else:
#             return torch.empty((0, 4), dtype=voxel_coords.dtype, device=voxel_coords.device), \
#                 torch.empty((0, voxel_features.shape[1]), dtype=voxel_features.dtype, device=voxel_features.device)

#     def radial_masking(self, voxel_coords, voxel_features):
#         """기존 간단한 방식 (fallback)"""
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
            
#             # Angular masking (기존 방식)
#             num_sectors = int(360 / self.angular_group_size)
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
#         """
#         ✅ 수정된 forward: 점군 직접 사용한 거리 계산
#         """
#         voxel_features = batch_dict['voxel_features']
#         voxel_coords = batch_dict['voxel_coords']
#         batch_size = batch_dict['batch_size']
        
#         # R-MAE masking 적용 (training 시에만)
#         if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
#             batch_dict['original_voxel_coords'] = voxel_coords.clone()
#             batch_dict['original_voxel_features'] = voxel_features.clone()
            
#             # 🎯 점군 데이터 직접 사용
#             if 'points' in batch_dict:
#                 points = batch_dict['points']  # [N, 4] (batch_idx, x, y, z)
                
#                 # 배치별로 처리
#                 for batch_idx in range(batch_size):
#                     batch_mask = points[:, 0] == batch_idx
#                     batch_points = points[batch_mask]
                    
#                     if len(batch_points) == 0:
#                         continue
                    
#                     # 🎯 점군에서 했던 것과 동일한 거리 계산
#                     point_x = batch_points[:, 1]
#                     point_y = batch_points[:, 2] 
#                     point_z = batch_points[:, 3]
#                     point_distances = torch.sqrt(point_x**2 + point_y**2 + point_z**2)
                    
#                     print(f"🎯 Direct Point Cloud Distance (Batch {batch_idx}):")
#                     print(f"   - Point count: {len(batch_points)}")
#                     print(f"   - Distance range: {point_distances.min():.2f} ~ {point_distances.max():.2f}m")
                    
#                     # 거리별 분포 확인
#                     near_points = (point_distances <= 10.0).sum()
#                     mid_points = ((point_distances > 10.0) & (point_distances <= 30.0)).sum()
#                     far_points = (point_distances > 30.0).sum()
                    
#                     print(f"   - Near points (≤10m): {near_points} ({near_points/len(batch_points)*100:.1f}%)")
#                     print(f"   - Mid points (10~30m): {mid_points} ({mid_points/len(batch_points)*100:.1f}%)")
#                     print(f"   - Far points (>30m): {far_points} ({far_points/len(batch_points)*100:.1f}%)")
#                     print(f"   - 🎯 Your analysis: Near ~64%, Mid ~36%, Far ~1%")
                    
#                     break  # 첫 배치만 출력
            
#             # 📄 R-MAE 논문 정확한 2-Stage masking 사용
#             if self.enable_2stage_masking:
#                 voxel_coords, voxel_features = self.classify_voxels_accurate_distance(voxel_coords, voxel_features, batch_dict)
                
#                 original_count = len(batch_dict['original_voxel_coords'])
#                 current_count = len(voxel_coords)
#                 actual_mask_ratio = 1.0 - (current_count / original_count) if original_count > 0 else 0.0
                
#                 print(f"🎯 R-MAE Masking: Target {self.masked_ratio:.1%} → Actual {actual_mask_ratio:.1%}")
            
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