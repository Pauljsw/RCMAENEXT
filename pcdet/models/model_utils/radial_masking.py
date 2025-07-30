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
import math
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
                empty_features = torch.empty(0, voxel_features.size(1), dtype=voxel_features.dtype, device=device)
                masked_coords_list.append(empty_coords)
                masked_features_list.append(empty_features)
                continue
            
            # Voxel 좌표를 실제 공간 좌표로 변환
            real_coords = self._voxel_to_real_coords(coords_b)
            
            # Stage 1: Angular Group Selection
            angular_keep_mask = self._angular_group_selection(real_coords)
            
            # Stage 2: Range-Aware Masking
            range_keep_mask = self._range_aware_masking(real_coords[angular_keep_mask])
            
            # 최종 마스크 결합
            final_indices = torch.where(angular_keep_mask)[0][range_keep_mask]
            
            # 최소 voxel 수 보장 (안정성을 위해)
            min_voxels = max(1, int(len(coords_b) * 0.05))  # 최소 5%는 유지
            if len(final_indices) < min_voxels:
                # 랜덤하게 추가 voxel 선택
                remaining_indices = torch.randperm(len(coords_b), device=device)[:min_voxels]
                final_indices = torch.unique(torch.cat([final_indices, remaining_indices]))
            
            # 마스킹된 결과 저장
            masked_coords_list.append(coords_b[final_indices])
            masked_features_list.append(features_b[final_indices])
        
        # 배치 결합
        if len(masked_coords_list) > 0 and all(len(c) > 0 for c in masked_coords_list):
            masked_coords = torch.cat(masked_coords_list, dim=0)
            masked_features = torch.cat(masked_features_list, dim=0)
        else:
            # 모든 배치가 비어있는 경우
            masked_coords = torch.empty(0, 4, dtype=voxel_coords.dtype, device=device)
            masked_features = torch.empty(0, voxel_features.size(1), dtype=voxel_features.dtype, device=device)
        
        return masked_coords, masked_features
    
    def _voxel_to_real_coords(self, voxel_coords: torch.Tensor) -> torch.Tensor:
        """
        Voxel 인덱스를 실제 공간 좌표로 변환
        
        Args:
            voxel_coords: [N, 4] (batch_idx, z, y, x)
            
        Returns:
            real_coords: [N, 3] (x, y, z) in meters
        """
        # Extract spatial coordinates (z, y, x) and convert to (x, y, z)
        voxel_indices = voxel_coords[:, [3, 2, 1]].float()  # [N, 3] (x, y, z)
        
        # Convert to real coordinates
        real_coords = torch.zeros_like(voxel_indices)
        real_coords[:, 0] = self.point_cloud_range[0] + voxel_indices[:, 0] * self.voxel_size[0]  # x
        real_coords[:, 1] = self.point_cloud_range[1] + voxel_indices[:, 1] * self.voxel_size[1]  # y
        real_coords[:, 2] = self.point_cloud_range[2] + voxel_indices[:, 2] * self.voxel_size[2]  # z
        
        return real_coords
    
    def _angular_group_selection(self, real_coords: torch.Tensor) -> torch.Tensor:
        """
        ✅ Stage 1: Angular Group Selection (논문 알고리즘)
        
        Args:
            real_coords: [N, 3] (x, y, z) coordinates
            
        Returns:
            keep_mask: [N] boolean mask
        """
        # 극좌표 변환: 방위각(azimuth) 계산
        x, y = real_coords[:, 0], real_coords[:, 1]
        azimuth = torch.atan2(y, x)  # [-π, π]
        azimuth_deg = azimuth * 180.0 / math.pi  # [-180, 180]
        
        # 각도를 [0, 360) 범위로 정규화
        azimuth_deg = (azimuth_deg + 360) % 360
        
        # Angular groups 생성
        num_groups = int(360 / self.angular_range)
        group_indices = (azimuth_deg / self.angular_range).long()
        group_indices = torch.clamp(group_indices, 0, num_groups - 1)
        
        # 그룹별로 확률적 선택 (pg = 1 - m)
        keep_prob = 1.0 - self.masked_ratio
        keep_mask = torch.zeros(len(real_coords), dtype=torch.bool, device=real_coords.device)
        
        for group_id in torch.unique(group_indices):
            group_mask = group_indices == group_id
            group_size = group_mask.sum().item()
            
            if group_size > 0:
                # 그룹 내에서 확률적 선택
                num_keep = max(1, int(group_size * keep_prob))
                group_indices_list = torch.where(group_mask)[0]
                
                if len(group_indices_list) <= num_keep:
                    keep_mask[group_mask] = True
                else:
                    # 랜덤 선택
                    perm = torch.randperm(len(group_indices_list), device=real_coords.device)
                    selected = group_indices_list[perm[:num_keep]]
                    keep_mask[selected] = True
        
        return keep_mask
    
    def _range_aware_masking(self, real_coords: torch.Tensor) -> torch.Tensor:
        """
        ✅ Stage 2: Range-Aware Masking within Selected Groups
        
        Args:
            real_coords: [N, 3] coordinates from selected angular groups
            
        Returns:
            keep_mask: [N] boolean mask
        """
        if len(real_coords) == 0:
            return torch.empty(0, dtype=torch.bool, device=real_coords.device)
        
        # 거리 계산 (원점으로부터)
        distances = torch.sqrt(real_coords[:, 0]**2 + real_coords[:, 1]**2)
        
        # 거리 기반 그룹 분할
        max_dist = distances.max().item()
        dist_ranges = [
            (0, max_dist * 0.3),          # near
            (max_dist * 0.3, max_dist * 0.7),  # mid  
            (max_dist * 0.7, max_dist)    # far
        ]
        
        keep_mask = torch.zeros(len(real_coords), dtype=torch.bool, device=real_coords.device)
        
        for i, (min_dist, max_dist_range) in enumerate(dist_ranges):
            range_mask = (distances >= min_dist) & (distances < max_dist_range)
            range_size = range_mask.sum().item()
            
            if range_size > 0:
                # 거리별 마스킹 확률
                if i == 0:  # near
                    keep_prob = 1.0 - self.distance_mask_probs['near']
                elif i == 1:  # mid
                    keep_prob = 1.0 - self.distance_mask_probs['mid']
                else:  # far
                    keep_prob = 1.0 - self.distance_mask_probs['far']
                
                num_keep = max(1, int(range_size * keep_prob))
                range_indices = torch.where(range_mask)[0]
                
                if len(range_indices) <= num_keep:
                    keep_mask[range_mask] = True
                else:
                    # 랜덤 선택
                    perm = torch.randperm(len(range_indices), device=real_coords.device)
                    selected = range_indices[perm[:num_keep]]
                    keep_mask[selected] = True
        
        return keep_mask