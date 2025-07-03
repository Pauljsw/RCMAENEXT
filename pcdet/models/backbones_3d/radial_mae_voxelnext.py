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
