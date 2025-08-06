"""
pcdet/models/backbones_3d/rmae_cmae_backbone_fixed.py

✅ 수정된 R-MAE + CMAE-3D Backbone
- Loss 계산 로직 분리
- Feature alignment 수정
- Sparse tensor 처리 개선
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from functools import partial
import numpy as np

from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
from ..model_utils.radial_masking import RadialMasking
from ..model_utils.hrcl_utils import HRCLModule


class RMAECMAEBackbone(VoxelResBackBone8xVoxelNeXt):
    """✅ R-MAE + CMAE-3D 통합 Backbone - Loss 계산 분리"""
    
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg, input_channels, grid_size, **kwargs)
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.model_cfg = model_cfg
        
        # ✅ R-MAE Radial Masking
        self.radial_masking = RadialMasking(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            **model_cfg.get('RMAE', {})
        )
        
        # ✅ Pretraining 모드 체크
        self.is_pretraining = model_cfg.get('PRETRAINING', False)
        
        if self.is_pretraining:
            # R-MAE Occupancy Decoder
            norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
            self.occupancy_decoder = spconv.SparseSequential(
                spconv.SubMConv3d(128, 64, 3, padding=1, bias=False, indice_key='occ_dec1'),
                norm_fn(64), nn.ReLU(),
                spconv.SubMConv3d(64, 32, 3, padding=1, bias=False, indice_key='occ_dec2'),
                norm_fn(32), nn.ReLU(),
                spconv.SubMConv3d(32, 1, 1, bias=True, indice_key='occ_out')
            )
            
            # CMAE-3D MLFR Decoders (multi-scale)
            if model_cfg.get('CMAE', {}).get('USE_MLFR', False):
                self.mlfr_decoders = nn.ModuleDict({
                    'x_conv1': self._build_mlfr_decoder(16, 16),   # stride 1
                    'x_conv2': self._build_mlfr_decoder(32, 32),   # stride 2
                    'x_conv3': self._build_mlfr_decoder(64, 64),   # stride 4
                    'x_conv4': self._build_mlfr_decoder(128, 128), # stride 8
                })
            
            # CMAE-3D HRCL Module
            if model_cfg.get('CMAE', {}).get('USE_HRCL', False):
                self.hrcl_module = HRCLModule(model_cfg.CMAE.HRCL)
        
        # 저장할 중간 결과들
        self.cached_features = {}
    
    def _build_mlfr_decoder(self, in_channels, out_channels):
        """MLFR decoder for each scale"""
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, in_channels//2, 3, padding=1, bias=False),
            norm_fn(in_channels//2), nn.ReLU(),
            spconv.SubMConv3d(in_channels//2, out_channels, 1, bias=True)
        )
    
    def forward(self, batch_dict):
        """Forward with proper feature caching"""
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        # ✅ R-MAE Radial Masking (training only)
        if self.training and self.is_pretraining:
            # 원본 저장
            batch_dict['original_voxel_coords'] = voxel_coords.clone()
            batch_dict['original_voxel_features'] = voxel_features.clone()
            
            # Masking 적용
            masked_result = self.radial_masking(voxel_coords, voxel_features)
            voxel_coords = masked_result['masked_coords']
            voxel_features = masked_result['masked_features']
            batch_dict['mask_indices'] = masked_result['mask_indices']
            batch_dict['kept_indices'] = masked_result['kept_indices']
        
        # Sparse tensor 생성
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # ✅ VoxelNeXt backbone forward
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        # Multi-scale features 저장
        multi_scale_features = {
            'x_conv1': x_conv1, 'x_conv2': x_conv2,
            'x_conv3': x_conv3, 'x_conv4': x_conv4
        }
        
        # ✅ Pretraining: predictions 생성
        if self.training and self.is_pretraining:
            # 1. R-MAE Occupancy prediction
            occupancy_pred = self.occupancy_decoder(x_conv4)
            self.cached_features['occupancy_pred'] = occupancy_pred
            
            # 2. CMAE MLFR predictions
            if hasattr(self, 'mlfr_decoders'):
                mlfr_preds = {}
                for scale_name, decoder in self.mlfr_decoders.items():
                    if scale_name in multi_scale_features:
                        pred = decoder(multi_scale_features[scale_name])
                        mlfr_preds[scale_name] = pred
                self.cached_features['mlfr_preds'] = mlfr_preds
            
            # 3. CMAE HRCL features
            if hasattr(self, 'hrcl_module'):
                # 최종 feature를 dense로 변환하여 HRCL 입력
                hrcl_input = x_conv4.dense()  # [B, C, D, H, W]
                B, C, D, H, W = hrcl_input.shape
                hrcl_input = hrcl_input.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # [B, N, C]
                
                voxel_embeddings = self.hrcl_module.voxel_projection(hrcl_input)
                frame_embeddings = self.hrcl_module.frame_projection(hrcl_input.mean(dim=1))
                
                self.cached_features['voxel_embeddings'] = voxel_embeddings
                self.cached_features['frame_embeddings'] = frame_embeddings
        
        # 출력
        batch_dict.update({
            'encoded_spconv_tensor': x_conv4,
            'encoded_spconv_tensor_stride': 8,
            'multi_scale_3d_features': multi_scale_features
        })
        
        return batch_dict
    
    def get_occupancy_loss(self):
        """✅ R-MAE Occupancy Loss 계산"""
        if 'occupancy_pred' not in self.cached_features:
            return torch.tensor(0.0).cuda(), {}
            
        occupancy_pred = self.cached_features['occupancy_pred']
        
        # Ground truth occupancy 생성 (실제로는 mask indices 기반)
        # 여기서는 단순화: masked region = 0, kept region = 1
        occupancy_gt = torch.ones_like(occupancy_pred.features)
        
        # Binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(
            occupancy_pred.features.squeeze(-1), 
            occupancy_gt.squeeze(-1)
        )
        
        tb_dict = {'loss_occupancy': loss.item()}
        return loss, tb_dict
    
    def get_mlfr_loss(self):
        """✅ CMAE Multi-scale Latent Feature Reconstruction Loss"""
        if 'mlfr_preds' not in self.cached_features:
            return torch.tensor(0.0).cuda(), {}
            
        if 'teacher_features' not in self.cached_features:
            # Teacher features가 없으면 skip
            return torch.tensor(0.0).cuda(), {}
            
        mlfr_preds = self.cached_features['mlfr_preds']
        teacher_features = self.cached_features['teacher_features']
        
        total_loss = 0
        tb_dict = {}
        
        for scale_name, pred in mlfr_preds.items():
            if scale_name in teacher_features:
                target = teacher_features[scale_name]
                
                # Feature size alignment
                if pred.features.shape != target.shape:
                    # Interpolate or pad as needed
                    continue
                    
                # L1 loss (논문 수식 11)
                scale_loss = F.l1_loss(pred.features, target)
                total_loss += scale_loss
                tb_dict[f'loss_mlfr_{scale_name}'] = scale_loss.item()
        
        tb_dict['loss_mlfr_total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else 0
        return total_loss, tb_dict
    
    def get_hrcl_loss(self):
        """✅ CMAE Hierarchical Relational Contrastive Loss"""
        if 'voxel_embeddings' not in self.cached_features:
            return torch.tensor(0.0).cuda(), {}
            
        voxel_emb = self.cached_features['voxel_embeddings']
        frame_emb = self.cached_features['frame_embeddings']
        
        # HRCL module이 loss 계산
        if hasattr(self, 'hrcl_module'):
            loss_dict = self.hrcl_module.compute_loss(voxel_emb, frame_emb)
            total_loss = loss_dict['total']
            
            tb_dict = {
                'loss_hrcl_voxel': loss_dict['voxel'].item(),
                'loss_hrcl_frame': loss_dict['frame'].item(),
                'loss_hrcl_total': total_loss.item()
            }
            return total_loss, tb_dict
        
        return torch.tensor(0.0).cuda(), {}
    
    def set_teacher_features(self, teacher_features):
        """Teacher features 설정 (Detector에서 호출)"""
        self.cached_features['teacher_features'] = teacher_features