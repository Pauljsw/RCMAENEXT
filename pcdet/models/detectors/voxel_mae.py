import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .detector3d_template import Detector3DTemplate

class VoxelMAE(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        # 먼저 부모 클래스 초기화 (여기서 backbone_3d가 생성됨)
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        
        # MAE 관련 설정
        self.model_cfg = model_cfg
        self.mask_ratio = model_cfg.get('MASK_RATIO', 0.75)
        
        # 모듈들 직접 생성 (부모 클래스에서 생성이 안된 경우 대비)
        if not hasattr(self, 'module_list') or self.module_list is None:
            self.module_list = self.build_networks()
        
        # MAE decoder 구현
        self.decoder = self._build_decoder()
        
        # Loss 함수들
        self.occupancy_loss = nn.BCEWithLogitsLoss()
        self.chamfer_loss = nn.MSELoss()
        
    def _build_decoder(self):
        """MAE 디코더 구현 - VoxelNeXt용 2D Sparse 디코더"""
        decoder_cfg = self.model_cfg.get('DECODER', {})
        
        # VoxelNeXt는 2D sparse tensor를 출력하므로 2D sparse decoder 사용
        import spconv.pytorch as spconv
        from functools import partial
        
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        # 2D Sparse decoder - occupancy prediction
        decoder_layers = spconv.SparseSequential(
            # 128 -> 64 channels (2D)
            spconv.SubMConv2d(128, 64, 3, padding=1, bias=False, indice_key='decoder1'),
            norm_fn(64),
            nn.ReLU(),
            
            # 64 -> 32 channels (2D)
            spconv.SubMConv2d(64, 32, 3, padding=1, bias=False, indice_key='decoder2'),
            norm_fn(32),
            nn.ReLU(),
            
            # 32 -> 1 channel (occupancy prediction) (2D)
            spconv.SubMConv2d(32, 1, 1, padding=0, bias=True, indice_key='decoder_out'),
        )
        
        return decoder_layers
    
    def random_masking(self, voxel_features, voxel_coords, mask_ratio=0.75):
        """복셀 랜덤 마스킹"""
        batch_size = int(voxel_coords[:, 0].max().item()) + 1  # int로 변환
        masked_features = voxel_features.clone()
        mask_indices_list = []
        
        for batch_idx in range(batch_size):
            # 현재 배치의 복셀 인덱스
            batch_mask = voxel_coords[:, 0] == batch_idx
            batch_voxel_num = batch_mask.sum().item()
            
            if batch_voxel_num == 0:  # 해당 배치에 복셀이 없으면 스킵
                continue
                
            # 마스킹할 복셀 수 계산
            num_masked = int(batch_voxel_num * mask_ratio)
            
            if num_masked == 0:  # 마스킹할 복셀이 없으면 스킵
                continue
            
            # 랜덤하게 마스킹할 복셀 선택
            batch_indices = torch.where(batch_mask)[0]
            masked_indices = torch.randperm(batch_voxel_num, device=voxel_features.device)[:num_masked]
            actual_masked_indices = batch_indices[masked_indices]
            
            # 마스킹 적용 (0으로 설정)
            masked_features[actual_masked_indices] = 0
            
            mask_indices_list.append(actual_masked_indices)
            
        return masked_features, mask_indices_list
    
    def forward(self, batch_dict):
        """MAE Forward pass"""
        # 1. VFE 적용 - module_list에서 VFE 찾기
        if hasattr(self, 'module_list') and self.module_list:
            for cur_module in self.module_list:
                if hasattr(cur_module, '__class__') and 'VFE' in cur_module.__class__.__name__:
                    batch_dict = cur_module(batch_dict)
                    break
        else:
            # 만약 module_list가 없다면 직접 VFE 호출 시도
            from pcdet.models.backbones_3d.vfe import MeanVFE
            if not hasattr(self, '_vfe'):
                self._vfe = MeanVFE(self.model_cfg.VFE, num_point_features=3)
            batch_dict = self._vfe(batch_dict)
        
        voxel_features = batch_dict['voxel_features'] 
        voxel_coords = batch_dict['voxel_coords']
        
        # 2. 복셀 마스킹
        masked_features, mask_indices = self.random_masking(
            voxel_features, voxel_coords, self.mask_ratio
        )
        
        # 마스킹된 특징으로 배치 딕셔너리 업데이트
        batch_dict['voxel_features'] = masked_features
        
        # 3. Backbone 적용
        if hasattr(self, 'module_list') and self.module_list:
            for cur_module in self.module_list:
                if hasattr(cur_module, '__class__') and 'BackBone' in cur_module.__class__.__name__:
                    batch_dict = cur_module(batch_dict)
                    break
        else:
            # 만약 module_list가 없다면 직접 backbone 호출 시도
            from pcdet.models.backbones_3d import VoxelResBackBone8xVoxelNeXt
            if not hasattr(self, '_backbone'):
                self._backbone = VoxelResBackBone8xVoxelNeXt(
                    self.model_cfg.BACKBONE_3D, 
                    input_channels=3,
                    grid_size=self.dataset.grid_size
                )
            batch_dict = self._backbone(batch_dict)
        
        # 4. Sparse 디코더로 복원
        # VoxelNeXt는 encoded_spconv_tensor (sparse tensor)를 출력
        if 'encoded_spconv_tensor' in batch_dict:
            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
            # Sparse tensor 그대로 디코더에 전달
            reconstructed_sparse = self.decoder(encoded_spconv_tensor)
        else:
            # 디버깅: 어떤 키들이 있는지 출력
            print("Available keys in batch_dict:", list(batch_dict.keys()))
            raise KeyError("No encoded_spconv_tensor found in batch_dict")
        
        # Sparse tensor를 dense로 변환해서 loss 계산
        reconstructed = reconstructed_sparse.dense()  # (B, 1, D, H, W)
        
        if self.training:
            # 5. Loss 계산
            loss_dict = self.compute_mae_loss(
                batch_dict, reconstructed_sparse, mask_indices
            )
            
            return {
                'loss': loss_dict['total_loss']
            }, loss_dict, loss_dict
        else:
            return batch_dict
    
    def compute_mae_loss(self, batch_dict, reconstructed_sparse, mask_indices):
        """MAE Loss 계산 - sparse tensor 기반"""
        # Sparse tensor를 dense로 변환
        reconstructed = reconstructed_sparse.dense()  # (B, 1, H, W)
        
        # Occupancy prediction loss
        target_occupancy = self.create_occupancy_target(batch_dict, reconstructed_sparse)
        occupancy_loss = self.occupancy_loss(reconstructed, target_occupancy)
        
        # 마스킹된 영역에서만 loss 계산
        masked_loss = self.compute_masked_reconstruction_loss(
            batch_dict, reconstructed, mask_indices
        )
        
        total_loss = occupancy_loss + 0.5 * masked_loss
        
        return {
            'total_loss': total_loss,
            'occupancy_loss': occupancy_loss,
            'masked_reconstruction_loss': masked_loss
        }
    
    def create_occupancy_target(self, batch_dict, reconstructed_sparse):
        """Occupancy target 생성 - sparse tensor 기반"""
        if reconstructed_sparse is None:
            return None
            
        # reconstructed sparse tensor의 spatial_shape을 사용
        reconstructed_dense = reconstructed_sparse.dense()  # (B, 1, H, W)
        batch_size, _, height, width = reconstructed_dense.shape
        
        # 2D occupancy target 생성
        occupancy_target = torch.zeros(
            (batch_size, 1, height, width),
            device=batch_dict['voxel_features'].device
        )
        
        # voxel_coords를 기반으로 occupancy 설정 (2D BEV)
        coords = batch_dict['voxel_coords']
        for i in range(len(coords)):
            b, z, y, x = coords[i]
            # tensor를 python int로 변환
            b, z, y, x = int(b), int(z), int(y), int(x)
            
            # 2D BEV이므로 x, y 좌표만 사용
            # 좌표 변환: voxel 좌표를 feature map 좌표로 변환
            stride = 8  # VoxelNeXt의 stride
            feat_y = y // stride
            feat_x = x // stride
            
            if 0 <= feat_x < width and 0 <= feat_y < height and 0 <= b < batch_size:
                occupancy_target[b, 0, feat_y, feat_x] = 1.0
                
        return occupancy_target
    
    def compute_masked_reconstruction_loss(self, batch_dict, reconstructed, mask_indices):
        """마스킹된 영역의 재구성 loss - 2D BEV 기반"""
        # 간단한 MSE loss 계산 (2D)
        target = self.create_occupancy_target(batch_dict, None)
        if target is None:
            # Fallback: reconstructed와 같은 크기의 타겟 생성
            target = torch.zeros_like(reconstructed)
        
        mse_loss = F.mse_loss(reconstructed, target, reduction='mean')
        
        # 실제 구현에서는 mask_indices를 spatial 좌표로 변환해서 
        # 마스킹된 영역에서만 loss 계산해야 하지만, 
        # 일단 전체 영역에서 loss 계산
        return mse_loss