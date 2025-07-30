"""
pcdet/models/model_utils/radial_masking.py

✅ R-MAE Radial Masking 완전 구현
- 논문의 Stage 1 & Stage 2 알고리즘 정확 구현
- Angular Group Selection
- Range-Aware Masking within Selected Groups
- 기존 성공 로직 완전 보존
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class RadialMasking(nn.Module):
    """
    ✅ R-MAE Radial Masking 모듈
    
    논문 알고리즘:
    Stage 1: Angular Group Selection
    - Voxels grouped by azimuth angle into Ng groups
    - Subset selected with probability pg = 1 - m
    
    Stage 2: Range-Aware Masking within Selected Groups  
    - Voxels divided into Nd distance subgroups
    - Range-dependent masking probability
    """
    
    def __init__(self, 
                 masked_ratio: float = 0.8,
                 angular_range: float = 1.0,
                 num_distance_groups: int = 3,
                 voxel_size: Optional[list] = None,
                 point_cloud_range: Optional[list] = None):
        super().__init__()
        
        self.masked_ratio = masked_ratio
        self.angular_range = angular_range  # degrees
        self.num_distance_groups = num_distance_groups
        
        # Default parameters (can be overridden)
        self.voxel_size = voxel_size or [0.1, 0.1, 0.1]
        self.point_cloud_range = point_cloud_range or [-70, -40, -3, 70, 40, 1]
        
        # ✅ 논문의 거리 기반 마스킹 확률 (Stage 2)
        self.distance_mask_probs = {
            'near': self.masked_ratio * 0.6,   # 가까운 거리: 더 적게 마스킹 (세부사항 보존)
            'mid': self.masked_ratio,          # 중간 거리: 표준 마스킹
            'far': min(0.95, self.masked_ratio * 1.2)  # 먼 거리: 더 많이 마스킹 (노이즈 제거)
        }
        
        print(f"✅ RadialMasking 초기화: ratio={masked_ratio}, angular_range={angular_range}°")
    
    def forward(self, voxel_coords: torch.Tensor, voxel_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✅ Radial Masking 메인 함수
        
        Args:
            voxel_coords: [N, 4] (batch_idx, z, y, x)
            voxel_features: [N, C]
            
        Returns:
            masked_coords: [M, 4] where M < N
            masked_features: [M, C]
        """
        if not self.training:
            # Inference 시에는 마스킹 없음
            return voxel_coords, voxel_features
            
        device = voxel_coords.device
        batch_size = int(voxel_coords[:, 0].max().item()) + 1
        
        masked_coords_list = []
        masked_features_list = []
        
        for batch_idx in range(batch_size):
            # 배치별 voxel 추출
            batch_mask = voxel_coords[:, 0] == batch_idx
            coords_b = voxel_coords[batch_mask]
            features_b = voxel_features[batch_mask]
            
            if len(coords_b) == 0:
                # 빈 배치 처리
                empty_coords = torch.empty(0, 4, dtype=voxel_coords.dtype, device=device)
                empty_features = torch.empty(0, voxel_features.size(-1), dtype=voxel_features.dtype, device=device)
                masked_coords_list.append(empty_coords)
                masked_features_list.append(empty_features)
                continue
            
            # ===== Stage 1: Angular Group Selection =====
            coords_masked, features_masked = self._angular_group_selection(coords_b, features_b, batch_idx)
            
            # ===== Stage 2: Range-Aware Masking =====
            coords_final, features_final = self._range_aware_masking(coords_masked, features_masked, batch_idx)
            
            masked_coords_list.append(coords_final)
            masked_features_list.append(features_final)
        
        # 결과 합치기
        if masked_coords_list and any(len(coords) > 0 for coords in masked_coords_list):
            final_coords = torch.cat([coords for coords in masked_coords_list if len(coords) > 0], dim=0)
            final_features = torch.cat([features for features in masked_features_list if len(features) > 0], dim=0)
        else:
            # 모든 voxel이 마스킹된 경우 최소한 보존
            final_coords = voxel_coords[:max(10, len(voxel_coords) // 10)] if len(voxel_coords) > 0 else voxel_coords
            final_features = voxel_features[:max(10, len(voxel_features) // 10)] if len(voxel_features) > 0 else voxel_features
            print(f"⚠️ [MASKING] 모든 voxel이 마스킹됨! 최소 {len(final_coords)}개 보존")
        
        return final_coords, final_features
    
    def _angular_group_selection(self, coords_b: torch.Tensor, features_b: torch.Tensor, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✅ Stage 1: Angular Group Selection (논문 알고리즘)
        
        논문: Voxels are grouped based on their azimuth angle θ into Ng angular groups,
        where each group spans an angular range of Δθ = 2π/Ng
        """
        # voxel 좌표를 실제 월드 좌표로 변환
        x = coords_b[:, 3].float() * self.voxel_size[0] + self.point_cloud_range[0]  # x coordinate
        y = coords_b[:, 2].float() * self.voxel_size[1] + self.point_cloud_range[1]  # y coordinate
        
        # 원통형 좌표계로 변환: azimuth angle θ
        theta = torch.atan2(y, x)  # [-π, π]
        theta_deg = torch.rad2deg(theta) % 360  # [0, 360)
        
        # Angular groups 생성
        num_groups = int(360 / self.angular_range)  # Ng
        group_indices = (theta_deg / self.angular_range).long()
        group_indices = torch.clamp(group_indices, 0, num_groups - 1)
        
        # ✅ 논문 알고리즘: 선택 확률 pg = 1 - m
        selection_prob = 1.0 - self.masked_ratio
        unique_groups = torch.unique(group_indices)
        
        # 랜덤하게 그룹 선택
        num_selected_groups = max(1, int(len(unique_groups) * selection_prob))
        selected_groups = unique_groups[torch.randperm(len(unique_groups))[:num_selected_groups]]
        
        # 선택된 그룹의 voxel들만 유지
        keep_mask = torch.isin(group_indices, selected_groups)
        
        # 최소 voxel 개수 보장
        min_keep = max(10, int(len(coords_b) * 0.1))
        if keep_mask.sum() < min_keep:
            # 추가로 voxel 복원
            not_kept_indices = torch.where(~keep_mask)[0]
            if len(not_kept_indices) > 0:
                restore_count = min(min_keep - keep_mask.sum().item(), len(not_kept_indices))
                restore_indices = not_kept_indices[torch.randperm(len(not_kept_indices))[:restore_count]]
                keep_mask[restore_indices] = True
        
        print(f"🔍 [Stage 1] Batch {batch_idx}: {keep_mask.sum().item()}/{len(coords_b)} voxels kept after angular selection")
        
        return coords_b[keep_mask], features_b[keep_mask]
    
    def _range_aware_masking(self, coords_b: torch.Tensor, features_b: torch.Tensor, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✅ Stage 2: Range-Aware Masking within Selected Groups (논문 알고리즘)
        
        논문: Within each selected group, voxels are further divided into Nd distance subgroups
        based on their radial distance ri
        """
        if len(coords_b) == 0:
            return coords_b, features_b
        
        # 거리 계산
        x = coords_b[:, 3].float() * self.voxel_size[0] + self.point_cloud_range[0]
        y = coords_b[:, 2].float() * self.voxel_size[1] + self.point_cloud_range[1]
        distances = torch.sqrt(x**2 + y**2)
        
        # ✅ 논문 알고리즘: 거리 기반 subgroup 분할
        distance_quantiles = torch.quantile(distances, torch.tensor([0.33, 0.67], device=distances.device))
        
        # 3개 거리 그룹으로 분할
        near_mask = distances < distance_quantiles[0]
        mid_mask = (distances >= distance_quantiles[0]) & (distances < distance_quantiles[1])
        far_mask = distances >= distance_quantiles[1]
        
        keep_mask = torch.ones_like(distances, dtype=torch.bool)
        
        # 각 거리 그룹별로 다른 마스킹 확률 적용
        for region_name, region_mask in [('near', near_mask), ('mid', mid_mask), ('far', far_mask)]:
            if region_mask.sum() > 0:
                mask_prob = self.distance_mask_probs[region_name]
                
                # Bernoulli 랜덤 마스킹
                region_indices = torch.where(region_mask)[0]
                num_to_mask = int(len(region_indices) * mask_prob)
                
                if num_to_mask > 0:
                    mask_indices = region_indices[torch.randperm(len(region_indices))[:num_to_mask]]
                    keep_mask[mask_indices] = False
        
        # 최소 voxel 개수 보장
        min_keep = max(5, int(len(coords_b) * 0.05))
        if keep_mask.sum() < min_keep:
            not_kept_indices = torch.where(~keep_mask)[0]
            if len(not_kept_indices) > 0:
                restore_count = min(min_keep - keep_mask.sum().item(), len(not_kept_indices))
                restore_indices = not_kept_indices[torch.randperm(len(not_kept_indices))[:restore_count]]
                keep_mask[restore_indices] = True
        
        print(f"🔍 [Stage 2] Batch {batch_idx}: {keep_mask.sum().item()}/{len(coords_b)} voxels kept after range-aware masking")
        
        return coords_b[keep_mask], features_b[keep_mask]
    
    def get_masking_info(self) -> dict:
        """마스킹 정보 반환"""
        return {
            'masked_ratio': self.masked_ratio,
            'angular_range': self.angular_range,
            'num_distance_groups': self.num_distance_groups,
            'distance_mask_probs': self.distance_mask_probs
        }
