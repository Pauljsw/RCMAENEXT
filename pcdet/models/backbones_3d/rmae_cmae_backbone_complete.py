"""
pcdet/models/backbones_3d/rmae_cmae_backbone_complete.py

✅ R-MAE + CMAE-3D 완전 통합 백본
- 기존 성공한 R-MAE 구조 100% 유지
- CMAE-3D Teacher-Student 논리 완벽 통합
- Sparse 3D CNN 구조에 최적화
- 유기적 연결과 안정성 보장
"""

import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from functools import partial

from .spconv_backbone import post_act_block, SparseBasicBlock
from ..model_utils.radial_masking import RadialMasking


class RMAECMAEBackboneComplete(spconv.SparseModule):
    """
    ✅ R-MAE + CMAE-3D 완전 통합 백본
    
    핵심 설계 원칙:
    1. 기존 성공한 R-MAE 로직 100% 보존
    2. CMAE-3D Teacher-Student 자연스러운 통합
    3. Momentum 기반 Teacher 업데이트
    4. 유기적 특징 추출 및 재구성
    """
    
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.input_channels = input_channels
        
        # ✅ CMAE-3D 핵심 파라미터
        self.momentum = model_cfg.get('MOMENTUM', 0.999)  # Teacher EMA momentum
        self.temperature = model_cfg.get('TEMPERATURE', 0.2)
        
        # ✅ R-MAE 파라미터 (기존 성공 로직)
        self.masked_ratio = model_cfg.get('MASKED_RATIO', 0.8)
        self.angular_range = model_cfg.get('ANGULAR_RANGE', 1)
        
        # 초기화 플래그
        self._teacher_initialized = False
        
        # ===== 1. Student Network (기존 VoxelNeXt) =====
        self._build_student_network()
        
        # ===== 2. Teacher Network (Student와 동일 구조) =====
        self._build_teacher_network()
        
        # ===== 3. R-MAE Components =====
        self._build_rmae_components()
        
        # ===== 4. CMAE-3D Components =====
        self._build_cmae_components()
        
        print("✅ R-MAE + CMAE-3D 완전 통합 백본 초기화 완료")
    
    def _build_student_network(self):
        """✅ Student network 구축 (기존 성공 VoxelNeXt 구조)"""
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(self.input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16), nn.ReLU()
        )
        
        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1_1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1_2'),
        )
        
        self.conv2 = spconv.SparseSequential(
            post_act_block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, 
                          indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2_1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2_2'),
        )
        
        self.conv3 = spconv.SparseSequential(
            post_act_block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1,
                          indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3_1'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3_2'),
        )
        
        self.conv4 = spconv.SparseSequential(
            post_act_block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1),
                          indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4_1'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4_2'),
        )
        
        print("✅ Student network (VoxelNeXt) 구축 완료")
    
    def _build_teacher_network(self):
        """✅ Teacher network 구축 (Student와 동일 구조, 분리된 파라미터)"""
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        # Teacher는 Student와 완전히 동일한 구조
        self.teacher_conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(self.input_channels, 16, 3, padding=1, bias=False, indice_key='teacher_subm1'),
            norm_fn(16), nn.ReLU()
        )
        
        self.teacher_conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='teacher_res1_1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='teacher_res1_2'),
        )
        
        self.teacher_conv2 = spconv.SparseSequential(
            post_act_block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, 
                          indice_key='teacher_spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='teacher_res2_1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='teacher_res2_2'),
        )
        
        self.teacher_conv3 = spconv.SparseSequential(
            post_act_block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1,
                          indice_key='teacher_spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='teacher_res3_1'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='teacher_res3_2'),
        )
        
        self.teacher_conv4 = spconv.SparseSequential(
            post_act_block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1),
                          indice_key='teacher_spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='teacher_res4_1'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='teacher_res4_2'),
        )
        
        print("✅ Teacher network 구축 완료")
    
    def _build_rmae_components(self):
        """✅ R-MAE 구성요소 구축 (기존 성공 로직)"""
        # Radial masking 모듈
        self.radial_masking = RadialMasking(
            masked_ratio=self.masked_ratio,
            angular_range=self.angular_range
        )
        
        # Occupancy decoder (R-MAE 핵심)
        self.occupancy_decoder = spconv.SparseSequential(
            spconv.SubMConv3d(128, 64, 3, padding=1, bias=False, indice_key='occupancy_1'),
            nn.BatchNorm1d(64), nn.ReLU(),
            spconv.SubMConv3d(64, 32, 3, padding=1, bias=False, indice_key='occupancy_2'),
            nn.BatchNorm1d(32), nn.ReLU(),
            spconv.SubMConv3d(32, 1, 1, bias=True, indice_key='occupancy_final')
        )
        
        print("✅ R-MAE 구성요소 구축 완료")
    
    def _build_cmae_components(self):
        """✅ CMAE-3D 구성요소 구축"""
        # 1. Multi-scale Latent Feature Reconstruction (MLFR)
        self.feature_decoder_conv4 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )
        
        self.feature_decoder_conv3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64), 
            nn.ReLU(inplace=True),
            nn.Linear(64, 64)
        )
        
        # 2. Hierarchical Relational Contrastive Learning (HRCL)
        # Voxel-level projection heads
        self.student_voxel_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        self.teacher_voxel_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        # Frame-level projection heads  
        self.student_frame_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        self.teacher_frame_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        print("✅ CMAE-3D 구성요소 구축 완료")
    
    @torch.no_grad()
    def _initialize_teacher_from_student(self):
        """✅ Teacher 초기화 (Student 파라미터로 복사)"""
        if self._teacher_initialized:
            return
            
        print("🔄 Teacher network 초기화 중...")
        
        # Student -> Teacher 파라미터 복사
        student_modules = [self.conv_input, self.conv1, self.conv2, self.conv3, self.conv4]
        teacher_modules = [self.teacher_conv_input, self.teacher_conv1, self.teacher_conv2, 
                          self.teacher_conv3, self.teacher_conv4]
        
        for student_module, teacher_module in zip(student_modules, teacher_modules):
            for s_param, t_param in zip(student_module.parameters(), teacher_module.parameters()):
                t_param.data.copy_(s_param.data)
        
        # Projection head 초기화
        for s_param, t_param in zip(self.student_voxel_proj.parameters(), self.teacher_voxel_proj.parameters()):
            t_param.data.copy_(s_param.data)
        for s_param, t_param in zip(self.student_frame_proj.parameters(), self.teacher_frame_proj.parameters()):
            t_param.data.copy_(s_param.data)
        
        self._teacher_initialized = True
        print("✅ Teacher 초기화 완료")
    
    @torch.no_grad()
    def _momentum_update_teacher(self):
        """✅ Teacher momentum 업데이트 (CMAE-3D 논문 방식)"""
        if not self._teacher_initialized:
            return
        
        # Backbone momentum update
        student_modules = [self.conv_input, self.conv1, self.conv2, self.conv3, self.conv4]
        teacher_modules = [self.teacher_conv_input, self.teacher_conv1, self.teacher_conv2, 
                          self.teacher_conv3, self.teacher_conv4]
        
        for student_module, teacher_module in zip(student_modules, teacher_modules):
            for s_param, t_param in zip(student_module.parameters(), teacher_module.parameters()):
                t_param.data.mul_(self.momentum).add_(s_param.data, alpha=1.0 - self.momentum)
        
        # Projection head momentum update
        for s_param, t_param in zip(self.student_voxel_proj.parameters(), self.teacher_voxel_proj.parameters()):
            t_param.data.mul_(self.momentum).add_(s_param.data, alpha=1.0 - self.momentum)
        for s_param, t_param in zip(self.student_frame_proj.parameters(), self.teacher_frame_proj.parameters()):
            t_param.data.mul_(self.momentum).add_(s_param.data, alpha=1.0 - self.momentum)
    
    def _student_forward(self, voxel_coords, voxel_features, batch_size):
        """✅ Student network forward (masked input)"""
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # VoxelNeXt forward
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        return {
            'conv1': x_conv1, 'conv2': x_conv2,
            'conv3': x_conv3, 'conv4': x_conv4,
            'final_tensor': x_conv4
        }
    
    @torch.no_grad()
    def _teacher_forward(self, voxel_coords, voxel_features, batch_size):
        """✅ Teacher network forward (complete input)"""
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # Teacher forward
        x = self.teacher_conv_input(input_sp_tensor)
        x_conv1 = self.teacher_conv1(x)
        x_conv2 = self.teacher_conv2(x_conv1)
        x_conv3 = self.teacher_conv3(x_conv2)
        x_conv4 = self.teacher_conv4(x_conv3)
        
        return {
            'conv1': x_conv1, 'conv2': x_conv2,
            'conv3': x_conv3, 'conv4': x_conv4,
            'final_tensor': x_conv4
        }
    
    def forward(self, batch_dict):
        """✅ 메인 forward pass - R-MAE + CMAE-3D 통합"""
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        # 원본 데이터 저장
        batch_dict['original_voxel_coords'] = voxel_coords.clone()
        batch_dict['original_voxel_features'] = voxel_features.clone()
        
        # Pretraining 모드
        if self.training and self.model_cfg.get('PRETRAINING', False):
            # Teacher 초기화 (첫 번째 forward에서만)
            if not self._teacher_initialized:
                self._initialize_teacher_from_student()
            
            # ===== 1. R-MAE Radial Masking (기존 성공 로직) =====
            masked_coords, masked_features = self.radial_masking(voxel_coords, voxel_features)
            
            # ===== 2. Teacher Forward (완전한 입력) =====
            teacher_outputs = self._teacher_forward(voxel_coords, voxel_features, batch_size)
            
            # ===== 3. Student Forward (마스킹된 입력) =====
            student_outputs = self._student_forward(masked_coords, masked_features, batch_size)
            
            # ===== 4. R-MAE Occupancy Prediction =====
            occupancy_pred = self.occupancy_decoder(student_outputs['final_tensor'])
            
            # ===== 5. CMAE-3D Feature Extraction =====
            # Student features
            student_conv4_features = student_outputs['final_tensor'].features
            student_conv3_features = student_outputs['conv3'].features if student_outputs['conv3'].features.size(0) > 0 else None
            
            # Teacher features  
            teacher_conv4_features = teacher_outputs['final_tensor'].features
            teacher_conv3_features = teacher_outputs['conv3'].features if teacher_outputs['conv3'].features.size(0) > 0 else None
            
            # MLFR reconstruction targets
            if student_conv4_features.size(0) > 0:
                student_recon_conv4 = self.feature_decoder_conv4(student_conv4_features)
            else:
                student_recon_conv4 = torch.empty(0, 128, device=student_conv4_features.device)
                
            if student_conv3_features is not None and student_conv3_features.size(0) > 0:
                student_recon_conv3 = self.feature_decoder_conv3(student_conv3_features)
            else:
                student_recon_conv3 = torch.empty(0, 64, device=voxel_features.device)
            
            # Contrastive projections
            if student_conv4_features.size(0) > 0:
                student_voxel_proj_features = self.student_voxel_proj(student_conv4_features)
                student_frame_proj_features = self.student_frame_proj(student_conv4_features.mean(dim=0, keepdim=True))
            else:
                student_voxel_proj_features = torch.empty(0, 128, device=voxel_features.device)
                student_frame_proj_features = torch.empty(1, 128, device=voxel_features.device)
                
            if teacher_conv4_features.size(0) > 0:
                teacher_voxel_proj_features = self.teacher_voxel_proj(teacher_conv4_features)
                teacher_frame_proj_features = self.teacher_frame_proj(teacher_conv4_features.mean(dim=0, keepdim=True))
            else:
                teacher_voxel_proj_features = torch.empty(0, 128, device=voxel_features.device)
                teacher_frame_proj_features = torch.empty(1, 128, device=voxel_features.device)
            
            # ===== 6. Teacher Momentum Update =====
            self._momentum_update_teacher()
            
            # ===== 7. 결과 정리 =====
            batch_dict.update({
                # Standard VoxelNeXt outputs
                'encoded_spconv_tensor': student_outputs['final_tensor'],
                'encoded_spconv_tensor_stride': 8,
                'multi_scale_3d_features': {
                    'x_conv1': student_outputs['conv1'],
                    'x_conv2': student_outputs['conv2'], 
                    'x_conv3': student_outputs['conv3'],
                    'x_conv4': student_outputs['final_tensor'],
                },
                
                # R-MAE outputs
                'occupancy_pred': occupancy_pred.features,
                'occupancy_coords': occupancy_pred.indices,
                
                # CMAE-3D outputs
                'teacher_conv4_features': teacher_conv4_features,
                'teacher_conv3_features': teacher_conv3_features,
                'student_recon_conv4': student_recon_conv4,
                'student_recon_conv3': student_recon_conv3,
                'student_voxel_proj': student_voxel_proj_features,
                'teacher_voxel_proj': teacher_voxel_proj_features,
                'student_frame_proj': student_frame_proj_features,
                'teacher_frame_proj': teacher_frame_proj_features,
                
                # Coordinates for loss computation
                'masked_coords': masked_coords,
                'masked_features': masked_features,
            })
            
        else:
            # ===== Fine-tuning/Inference 모드 =====
            student_outputs = self._student_forward(voxel_coords, voxel_features, batch_size)
            batch_dict.update({
                'encoded_spconv_tensor': student_outputs['final_tensor'],
                'encoded_spconv_tensor_stride': 8,
                'multi_scale_3d_features': {
                    'x_conv1': student_outputs['conv1'],
                    'x_conv2': student_outputs['conv2'],
                    'x_conv3': student_outputs['conv3'], 
                    'x_conv4': student_outputs['final_tensor'],
                }
            })
        
        return batch_dict


# ✅ 모델 등록을 위한 alias
RMAECMAEBackbone = RMAECMAEBackboneComplete
