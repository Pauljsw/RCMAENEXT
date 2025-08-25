# pcdet/models/backbones_3d/rmae_cmae_backbone_phase2.py
"""
CMAE-3D Phase 2: Multi-scale Latent Feature Reconstruction Backbone

Phase 1 기능 + Phase 2 MLFR 추가:
- Teacher-Student Architecture (Phase 1에서 구축됨)
- Multi-scale Latent Feature Reconstruction (새로 추가)
- 기존 R-MAE 완전 호환 유지
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from functools import partial
import numpy as np

# Phase 1 backbone 상속
from .rmae_cmae_backbone_phase1 import RMAECMAEBackbonePhase1
from .components.multiscale_feature_decoder import MultiScaleFeatureDecoder
from .components.voxel_contrastive_loss import VoxelContrastiveLoss


class RMAECMAEBackbonePhase2(RMAECMAEBackbonePhase1):
    """
    🔥 Phase 2: Multi-scale Latent Feature Reconstruction Backbone
    
    Phase 1 기능 완전 유지 + CMAE-3D MLFR 추가:
    
    Phase 1 (기존):
    - Teacher-Student Architecture ✅
    - R-MAE occupancy prediction ✅
    - Basic feature extraction ✅
    
    Phase 2 (새로 추가):
    - Multi-scale Latent Feature Reconstruction 🔥
    - Semantic feature reconstruction 🔥  
    - Enhanced loss function 🔥
    """
    
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        # Phase 1 초기화
        super().__init__(model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs)
        
        # 📍 Phase 2 MLFR 설정
        self.enable_mlfr = model_cfg.get('ENABLE_MLFR', True)
        self.mlfr_weight = model_cfg.get('MLFR_WEIGHT', 1.0)
        
        # 📍 Phase 2 Voxel Contrastive Learning 설정
        self.enable_voxel_contrastive = model_cfg.get('ENABLE_VOXEL_CONTRASTIVE', True)
        self.voxel_contrastive_weight = model_cfg.get('VOXEL_CONTRASTIVE_WEIGHT', 0.6)  # CMAE-3D 논문 기본값
        
        # MLFR 활성화 시에만 decoder 생성
        if self.enable_mlfr and self.enable_teacher_student:
            self.mlfr_decoder = MultiScaleFeatureDecoder(model_cfg.get('MLFR_CONFIG', {}))
            print(f"🔥 Phase 2 MLFR enabled with weight: {self.mlfr_weight}")
        else:
            self.mlfr_decoder = None
            print(f"🎯 Phase 2 MLFR disabled (enable_mlfr: {self.enable_mlfr}, teacher_student: {self.enable_teacher_student})")
        
        # Voxel Contrastive Learning 활성화 시에만 loss module 생성
        if self.enable_voxel_contrastive and self.enable_teacher_student:
            voxel_contrastive_cfg = model_cfg.get('VOXEL_CONTRASTIVE_CONFIG', {})
            voxel_contrastive_cfg.update({
                'FEATURE_DIM': 128,  # Final feature dimension
                'PROJECTION_DIM': 128,  # Contrastive projection dimension
                'CONTRASTIVE_TEMPERATURE': 0.07  # CMAE-3D paper default
            })
            self.voxel_contrastive_loss = VoxelContrastiveLoss(voxel_contrastive_cfg)
            print(f"🔥 Phase 2 Voxel Contrastive enabled with weight: {self.voxel_contrastive_weight}")
        else:
            self.voxel_contrastive_loss = None
            print(f"🎯 Phase 2 Voxel Contrastive disabled (enable: {self.enable_voxel_contrastive}, teacher_student: {self.enable_teacher_student})")
        
        # Phase 2 forward return dict
        self.forward_ret_dict = {}
        
        print(f"🚀 Phase 2 Backbone initialized:")
        print(f"   - Phase 1 features: ✅ Teacher-Student, R-MAE")
        print(f"   - Phase 2 features: {'✅' if self.enable_mlfr else '❌'} MLFR, {'✅' if self.enable_voxel_contrastive else '❌'} Voxel Contrastive")
    
    def teacher_forward(self, batch_dict):
        """
        🔥 Phase 2 Enhanced Teacher Forward
        
        Phase 1 teacher_forward 확장:
        - 기존: Multi-scale feature 추출
        - 추가: MLFR을 위한 더 정확한 feature alignment
        """
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        # VoxelNeXt backbone forward (완전한 view)
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # Multi-scale feature extraction
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)      # Scale 1: 16 channels
        x_conv2 = self.conv2(x_conv1) # Scale 2: 32 channels  
        x_conv3 = self.conv3(x_conv2) # Scale 3: 64 channels
        x_conv4 = self.conv4(x_conv3) # Scale 4: 128 channels
        
        # 📍 Phase 2: MLFR을 위한 정확한 multi-scale features
        teacher_features = {
            'scale_1': x_conv1,  # SparseConvTensor
            'scale_2': x_conv2,  # SparseConvTensor
            'scale_3': x_conv3,  # SparseConvTensor
            'scale_4': x_conv4,  # SparseConvTensor
            'final_tensor': x_conv4,  # Backward compatibility
            
            # 추가: MLFR debugging을 위한 정보
            'scale_info': {
                'scale_1': {'channels': 16, 'num_voxels': x_conv1.features.size(0)},
                'scale_2': {'channels': 32, 'num_voxels': x_conv2.features.size(0)},
                'scale_3': {'channels': 64, 'num_voxels': x_conv3.features.size(0)},
                'scale_4': {'channels': 128, 'num_voxels': x_conv4.features.size(0)}
            }
        }
        
        return teacher_features
    
    def student_forward(self, batch_dict):
        """
        🔥 Phase 2 Enhanced Student Forward
        
        Phase 1 student_forward 확장:
        - 기존: R-MAE masked forward
        - 추가: MLFR을 위한 multi-scale feature 추출
        """
        # Phase 1의 기본 student forward 실행
        result = super().student_forward(batch_dict)
        
        # 📍 Phase 2: Student의 multi-scale features 추가 추출
        if self.enable_mlfr and 'encoded_spconv_tensor' in result:
            
            # Student branch의 multi-scale features 추출
            # (masked input에서 생성된 features)
            voxel_features = batch_dict['voxel_features']
            voxel_coords = batch_dict['voxel_coords'] 
            batch_size = batch_dict['batch_size']
            
            # Masked input으로 multi-scale 재추출
            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            
            x = self.conv_input(input_sp_tensor)
            x_conv1 = self.conv1(x)      # Scale 1: 16 channels
            x_conv2 = self.conv2(x_conv1) # Scale 2: 32 channels
            x_conv3 = self.conv3(x_conv2) # Scale 3: 64 channels
            x_conv4 = self.conv4(x_conv3) # Scale 4: 128 channels
            
            # Student multi-scale features 저장
            result['student_multiscale_features'] = {
                'scale_1': x_conv1,
                'scale_2': x_conv2, 
                'scale_3': x_conv3,
                'scale_4': x_conv4
            }
            
            # Scale 정보 저장 (debugging)
            result['student_scale_info'] = {
                'scale_1': {'channels': 16, 'num_voxels': x_conv1.features.size(0)},
                'scale_2': {'channels': 32, 'num_voxels': x_conv2.features.size(0)},
                'scale_3': {'channels': 64, 'num_voxels': x_conv3.features.size(0)},
                'scale_4': {'channels': 128, 'num_voxels': x_conv4.features.size(0)}
            }
        
        return result
    
    def forward(self, batch_dict):
        """
        🚀 Phase 2 Forward: Teacher-Student + MLFR 통합
        
        Phase 1 forward 확장:
        1. Teacher-Student forward (Phase 1) ✅
        2. Multi-scale Feature Reconstruction (Phase 2) 🔥
        3. Loss 계산 및 저장
        """
        
        if not self.enable_teacher_student or not self.training:
            # Teacher-Student 비활성화 시: 기존 R-MAE 동작 (Phase 1과 동일)
            return super().forward(batch_dict)
        
        # 📍 Phase 2: Enhanced Teacher-Student + MLFR Training Mode
        
        # 1. Teacher Branch: Complete view 처리
        teacher_batch = {
            'voxel_features': batch_dict['voxel_features'].clone(),
            'voxel_coords': batch_dict['voxel_coords'].clone(),
            'batch_size': batch_dict['batch_size']
        }
        teacher_features = self.teacher_forward(teacher_batch)
        
        # 2. Student Branch: Masked view 처리 + multi-scale features
        student_result = self.student_forward(batch_dict)
        
        # 3. 📍 Phase 2: Multi-scale Latent Feature Reconstruction
        mlfr_results = {}
        if self.enable_mlfr and self.mlfr_decoder is not None:
            if 'student_multiscale_features' in student_result:
                
                student_multiscale = student_result['student_multiscale_features']
                mlfr_results = self.mlfr_decoder(student_multiscale, teacher_features)
                
                print(f"🔥 MLFR executed: {len(mlfr_results['mlfr_losses'])} scales, " +
                      f"total_loss={mlfr_results['total_mlfr_loss']:.6f}")
            else:
                print("⚠️ MLFR skipped: student_multiscale_features not found")
                mlfr_results = {
                    'reconstructed_features': {},
                    'mlfr_losses': {},
                    'total_mlfr_loss': torch.tensor(0.0, device='cuda', requires_grad=True)
                }
        
        # 📍 4. Phase 2: Voxel-level Contrastive Learning
        voxel_contrastive_results = {}
        if self.enable_voxel_contrastive and self.voxel_contrastive_loss is not None:
            # Teacher-Student final features 사용 (scale_4)
            if 'scale_4' in teacher_features and 'student_multiscale_features' in student_result:
                teacher_final = teacher_features['scale_4']  # SparseConvTensor
                student_final = student_result['student_multiscale_features']['scale_4']  # SparseConvTensor
                
                # Voxel contrastive learning 수행
                voxel_contrastive_results = self.voxel_contrastive_loss(
                    teacher_features=teacher_final.features,      # [N_t, 128]
                    student_features=student_final.features,      # [N_s, 128] 
                    teacher_coords=teacher_final.indices,         # [N_t, 4] (batch, z, y, x)
                    student_coords=student_final.indices          # [N_s, 4] (batch, z, y, x)
                )
                
                print(f"🔥 Voxel Contrastive executed: {voxel_contrastive_results['num_positive_pairs']} pos pairs, " +
                      f"acc={voxel_contrastive_results['contrastive_acc']:.4f}, " +
                      f"loss={voxel_contrastive_results['voxel_contrastive_loss']:.6f}")
            else:
                print("⚠️ Voxel Contrastive skipped: required features not found")
                voxel_contrastive_results = {
                    'voxel_contrastive_loss': torch.tensor(0.0, device='cuda', requires_grad=True),
                    'num_positive_pairs': 0,
                    'num_negative_pairs': 0,
                    'contrastive_acc': 0.0,
                    'avg_positive_sim': 0.0,
                    'avg_negative_sim': 0.0
                }
        
        # 5. Feature Projection (Phase 1 호환성)
        # 5. Feature Projection (Phase 1 호환성)
        teacher_embed = None
        student_embed = None
        if hasattr(self, 'teacher_projector') and hasattr(self, 'student_projector'):
            # Global average pooling for projection
            teacher_global = torch.mean(teacher_features['scale_4'].features, dim=0, keepdim=True)
            student_global = torch.mean(student_result['encoded_spconv_tensor'].features, dim=0, keepdim=True)
            
            teacher_embed = self.teacher_projector(teacher_global)
            student_embed = self.student_projector(student_global)
        
        # 6. Result 통합 (Phase 1 + Phase 2)
        result = student_result.copy()  # 기본은 student result (Phase 1 호환성)
        
        # Phase 1 features
        result.update({
            'teacher_features': teacher_features,
            'teacher_embed': teacher_embed,
            'student_embed': student_embed,
            'phase1_enabled': True
        })
        
        # 📍 Phase 2 features 추가
        result.update({
            'mlfr_results': mlfr_results,
            'mlfr_total_loss': mlfr_results.get('total_mlfr_loss', 0.0),
            'voxel_contrastive_results': voxel_contrastive_results,
            'voxel_contrastive_loss': voxel_contrastive_results.get('voxel_contrastive_loss', 0.0),
            'phase2_enabled': True,
            'phase2_mlfr': self.enable_mlfr,
            'phase2_voxel_contrastive': self.enable_voxel_contrastive
        })
        
        # Forward return dict에 저장 (loss 계산용)
        self.forward_ret_dict.update(result)
        
        return result
    
    def get_loss(self, tb_dict=None):
        """
        Phase 2 Loss: Phase 1 loss + MLFR loss + Voxel Contrastive loss
        
        Loss 구성:
        1. R-MAE occupancy loss (Phase 1) ✅
        2. Multi-scale feature reconstruction loss (Phase 2) 🔥
        3. Voxel-level contrastive loss (Phase 2) 🔥
        """
        tb_dict = {} if tb_dict is None else tb_dict
        
        # 📍 1. Phase 1 R-MAE Loss (기존 유지)
        rmae_loss, rmae_tb_dict = self._get_rmae_loss()
        tb_dict.update(rmae_tb_dict)
        
        # 📍 2. Phase 2 MLFR Loss
        mlfr_loss = 0.0
        if self.enable_mlfr and 'mlfr_results' in self.forward_ret_dict:
            mlfr_results = self.forward_ret_dict['mlfr_results']
            mlfr_loss = mlfr_results.get('total_mlfr_loss', 0.0)
            
            # Scale별 loss 기록
            mlfr_losses = mlfr_results.get('mlfr_losses', {})
            for scale, scale_loss in mlfr_losses.items():
                tb_dict[f'mlfr_{scale}_loss'] = scale_loss.item() if torch.is_tensor(scale_loss) else scale_loss
            
            tb_dict['mlfr_total_loss'] = mlfr_loss.item() if torch.is_tensor(mlfr_loss) else mlfr_loss
        
        # 📍 3. Phase 2 Voxel Contrastive Loss (새로 추가)
        voxel_contrastive_loss = 0.0
        if self.enable_voxel_contrastive and 'voxel_contrastive_results' in self.forward_ret_dict:
            voxel_results = self.forward_ret_dict['voxel_contrastive_results']
            voxel_contrastive_loss = voxel_results.get('voxel_contrastive_loss', 0.0)
            
            # Contrastive learning statistics 기록
            tb_dict['voxel_contrastive_loss'] = voxel_contrastive_loss.item() if torch.is_tensor(voxel_contrastive_loss) else voxel_contrastive_loss
            tb_dict['voxel_contrastive_acc'] = voxel_results.get('contrastive_acc', 0.0)
            tb_dict['voxel_positive_pairs'] = voxel_results.get('num_positive_pairs', 0)
            tb_dict['voxel_negative_pairs'] = voxel_results.get('num_negative_pairs', 0)
            tb_dict['voxel_avg_pos_sim'] = voxel_results.get('avg_positive_sim', 0.0)
            tb_dict['voxel_avg_neg_sim'] = voxel_results.get('avg_negative_sim', 0.0)
        
        # 📍 4. Total Loss (weighted combination)
        total_loss = (rmae_loss + 
                     self.mlfr_weight * mlfr_loss + 
                     self.voxel_contrastive_weight * voxel_contrastive_loss)
        
        tb_dict.update({
            'phase2_total_loss': total_loss.item(),
            'phase2_rmae_loss': rmae_loss.item() if torch.is_tensor(rmae_loss) else rmae_loss,
            'phase2_mlfr_loss': mlfr_loss.item() if torch.is_tensor(mlfr_loss) else mlfr_loss,
            'phase2_voxel_contrastive_loss': voxel_contrastive_loss.item() if torch.is_tensor(voxel_contrastive_loss) else voxel_contrastive_loss,
            'mlfr_weight': self.mlfr_weight,
            'voxel_contrastive_weight': self.voxel_contrastive_weight,
            'phase2_enabled': True
        })
        
        return total_loss, tb_dict
    
    def _get_rmae_loss(self):
        """Phase 1 R-MAE occupancy loss (동일)"""
        tb_dict = {}
        
        if 'occupancy_pred' in self.forward_ret_dict and 'occupancy_target' in self.forward_ret_dict:
            occupancy_pred = self.forward_ret_dict['occupancy_pred']
            occupancy_target = self.forward_ret_dict['occupancy_target']
            
            if hasattr(self, 'criterion'):
                rmae_loss = self.criterion(occupancy_pred, occupancy_target)
            else:
                criterion = torch.nn.BCEWithLogitsLoss()
                rmae_loss = criterion(occupancy_pred, occupancy_target)
            
            tb_dict['rmae_occupancy_loss'] = rmae_loss.item()
            return rmae_loss, tb_dict
        else:
            return torch.tensor(0.0, device='cuda', requires_grad=True), tb_dict


def build_rmae_cmae_backbone_phase2(model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
    """Factory function for creating RMAECMAEBackbonePhase2"""
    return RMAECMAEBackbonePhase2(
        model_cfg=model_cfg,
        input_channels=input_channels,
        grid_size=grid_size,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        **kwargs
    )