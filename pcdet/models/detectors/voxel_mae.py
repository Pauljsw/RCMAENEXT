import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random  # 추가
from .detector3d_template import Detector3DTemplate

class VoxelMAE(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        
        # R-MAE 스타일 파라미터 추가
        self.mask_ratio = model_cfg.get('MASK_RATIO', 0.35)  # 보수적 설정
        self.angular_range = model_cfg.get('ANGULAR_RANGE', 20)  # 새로 추가
        self.range_aware = model_cfg.get('RANGE_AWARE', False)  # 단순화
        self.adaptive_masking = model_cfg.get('ADAPTIVE_MASKING', True)
        
        # Dataset 정보 추가 (R-MAE 마스킹용)
        self.point_cloud_range = dataset.point_cloud_range
        self.voxel_size = dataset.voxel_size
        
        # 기존 코드는 그대로...
        if not hasattr(self, 'module_list') or self.module_list is None:
            self.module_list = self.build_networks()
        
        self.decoder = self._build_decoder()
        loss_cfg = model_cfg.get('LOSS_CONFIG', {})
        pos_weight = loss_cfg.get('POS_WEIGHT', 1.0)
        
        self.occupancy_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight).cuda()
        )
        
        print(f"✅ VoxelMAE Loss: pos_weight={pos_weight}")
        self.chamfer_loss = nn.MSELoss()
    
    def _build_decoder(self):
        """MAE 디코더 구현 - VoxelNeXt용 2D Sparse 디코더"""
        # VoxelNeXt는 2D sparse tensor를 출력하므로 2D sparse decoder 사용
        import spconv.pytorch as spconv
        from torch.nn import GroupNorm  # BatchNorm 문제 해결
        
        # 2D Sparse decoder - occupancy prediction
        decoder_layers = spconv.SparseSequential(
            # 128 -> 64 channels (2D)
            spconv.SubMConv2d(128, 64, 3, padding=1, bias=False, indice_key='decoder1'),
            GroupNorm(1, 64),  # GroupNorm 사용 (BatchNorm1d 문제 해결)
            nn.ReLU(),
            
            # 64 -> 32 channels (2D)
            spconv.SubMConv2d(64, 32, 3, padding=1, bias=False, indice_key='decoder2'),
            GroupNorm(1, 32),
            nn.ReLU(),
            
            # 32 -> 1 channel (occupancy prediction) (2D)
            spconv.SubMConv2d(32, 1, 1, padding=0, bias=True, indice_key='decoder_out'),
        )
        
        return decoder_layers
        
    def adaptive_radial_masking(self, voxel_features, voxel_coords):
        """적응적 R-MAE 스타일 마스킹"""
        total_voxels = len(voxel_coords)
        
        # 데이터 밀도에 따른 마스킹 비율 조정
        if self.adaptive_masking:
            if total_voxels < 500:  # 매우 sparse
                adaptive_mask_ratio = 0.25
            elif total_voxels < 1000:  # 중간 밀도
                adaptive_mask_ratio = 0.35
            else:  # 충분한 밀도
                adaptive_mask_ratio = self.mask_ratio
        else:
            adaptive_mask_ratio = self.mask_ratio
        
        select_ratio = 1 - adaptive_mask_ratio
        
        # voxel 좌표를 실제 좌표로 변환
        real_x = (voxel_coords[:, 3].float() * self.voxel_size[0]) + self.point_cloud_range[0]
        real_y = (voxel_coords[:, 2].float() * self.voxel_size[1]) + self.point_cloud_range[1]
        
        # 각도 계산
        angles = torch.atan2(real_y, real_x)
        angles_deg = torch.rad2deg(angles) % 360
        
        # Angular groups 생성
        radial_groups = {}
        for angle in range(0, 360, self.angular_range):
            mask = (angles_deg >= angle) & (angles_deg < angle + self.angular_range)
            group_indices = torch.where(mask)[0]
            if len(group_indices) > 0:
                radial_groups[angle] = group_indices
        
        if len(radial_groups) == 0:
            # Fallback: 모든 인덱스 반환
            return voxel_features, voxel_coords, torch.arange(len(voxel_coords), dtype=torch.long, device=voxel_coords.device)
        
        # 랜덤하게 그룹 선택
        num_groups_to_select = max(1, int(select_ratio * len(radial_groups)))
        selected_group_angles = random.sample(list(radial_groups.keys()), num_groups_to_select)
        
        # 선택된 그룹의 인덱스 수집
        selected_indices = []
        for angle in selected_group_angles:
            selected_indices.append(radial_groups[angle])
        
        # 최종 선택된 인덱스들
        if selected_indices:
            final_indices = torch.cat(selected_indices)
        else:
            final_indices = torch.arange(len(voxel_coords), dtype=torch.long, device=voxel_coords.device)
        
        # 선택된 voxel만 반환
        masked_features = voxel_features[final_indices]
        masked_coords = voxel_coords[final_indices]
        
        return masked_features, masked_coords, final_indices
    
    def forward(self, batch_dict):
        """R-MAE 스타일 Forward pass"""
        # 1. VFE 적용 (기존 코드 그대로)
        if hasattr(self, 'module_list') and self.module_list:
            for cur_module in self.module_list:
                if hasattr(cur_module, '__class__') and 'VFE' in cur_module.__class__.__name__:
                    batch_dict = cur_module(batch_dict)
                    break
        else:
            from pcdet.models.backbones_3d.vfe import MeanVFE
            if not hasattr(self, '_vfe'):
                self._vfe = MeanVFE(self.model_cfg.VFE, num_point_features=3)
            batch_dict = self._vfe(batch_dict)
        
        voxel_features = batch_dict['voxel_features'] 
        voxel_coords = batch_dict['voxel_coords']
        
        # 원본 데이터 저장 (loss 계산용)
        batch_dict['original_voxel_coords'] = voxel_coords.clone()
        batch_dict['original_voxel_features'] = voxel_features.clone()
        
        # 2. R-MAE 스타일 적응적 마스킹
        if self.training:
            masked_features, masked_coords, selected_indices = self.adaptive_radial_masking(
                voxel_features, voxel_coords
            )
            batch_dict['voxel_features'] = masked_features
            batch_dict['voxel_coords'] = masked_coords
            batch_dict['selected_indices'] = selected_indices
            batch_dict['mask_ratio'] = 1.0 - (len(selected_indices) / len(voxel_coords))
        
        # 3. Backbone 적용 (기존 코드 그대로)
        if hasattr(self, 'module_list') and self.module_list:
            for cur_module in self.module_list:
                if hasattr(cur_module, '__class__') and 'BackBone' in cur_module.__class__.__name__:
                    batch_dict = cur_module(batch_dict)
                    break
        else:
            from pcdet.models.backbones_3d import VoxelResBackBone8xVoxelNeXt
            if not hasattr(self, '_backbone'):
                self._backbone = VoxelResBackBone8xVoxelNeXt(
                    self.model_cfg.BACKBONE_3D, 
                    input_channels=3,
                    grid_size=self.dataset.grid_size
                )
            batch_dict = self._backbone(batch_dict)
        
        # 4. Decoder 및 Loss
        if 'encoded_spconv_tensor' in batch_dict:
            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
            reconstructed_sparse = self.decoder(encoded_spconv_tensor)
            
            if self.training:
                loss_dict = self.compute_rmae_loss(batch_dict, reconstructed_sparse)
                return {'loss': loss_dict['total_loss']}, loss_dict, loss_dict
            else:
                return batch_dict
        else:
            print("Available keys in batch_dict:", list(batch_dict.keys()))
            raise KeyError("No encoded_spconv_tensor found in batch_dict")
    
    def compute_rmae_loss(self, batch_dict, reconstructed_sparse):
        """R-MAE 스타일 Loss 계산"""
        reconstructed = reconstructed_sparse.dense()
        target_occupancy = self.create_rmae_occupancy_target(batch_dict, reconstructed_sparse)
        occupancy_loss = self.occupancy_loss(reconstructed, target_occupancy)
        
        return {
            'total_loss': occupancy_loss,
            'occupancy_loss': occupancy_loss,
        }
    
    def create_rmae_occupancy_target(self, batch_dict, reconstructed_sparse):
        """R-MAE 스타일 occupancy target 생성"""
        reconstructed_dense = reconstructed_sparse.dense()
        batch_size, _, height, width = reconstructed_dense.shape
        
        occupancy_target = torch.zeros(
            (batch_size, 1, height, width),
            device=batch_dict['voxel_features'].device
        )
        
        # 원본 좌표 사용
        original_coords = batch_dict['original_voxel_coords']
        for i in range(len(original_coords)):
            b, z, y, x = original_coords[i]
            b, z, y, x = int(b), int(z), int(y), int(x)
            
            stride = 8
            feat_y = y // stride
            feat_x = x // stride
            
            if 0 <= feat_x < width and 0 <= feat_y < height and 0 <= b < batch_size:
                occupancy_target[b, 0, feat_y, feat_x] = 1.0
                
        return occupancy_target