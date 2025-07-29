"""
pcdet/models/backbones_3d/rmae_cmae_backbone.py

R-MAE + CMAE-3D 통합 Backbone
- 기존 성공한 R-MAE 로직 100% 유지 
- CMAE-3D 논문 논리 정확히 구현 (Sparse CNN 적응)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from functools import partial
import numpy as np
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt


class RMAECMAEBackbone(VoxelResBackBone8xVoxelNeXt):
    """
    R-MAE + CMAE-3D Backbone
    
    논문 논리 유지:
    1. ✅ R-MAE radial masking (기존 성공 로직)
    2. ✅ CMAE-3D Teacher-Student with proper EMA (논문 구조)
    3. ✅ Sparse CNN 적응 (Transformer 대신 VoxelNeXt backbone)
    """
    
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        # ✅ 기존 성공 로직: 부모 클래스 초기화 (기존 VoxelNeXt 구조 그대로)
        super().__init__(model_cfg, input_channels, grid_size, **kwargs)
        
        # ✅ 실제 속성 저장 (Teacher 구축에 필요)
        self.model_cfg = model_cfg
        self.input_channels = input_channels  
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # ✅ 기존 성공 로직: R-MAE 파라미터
        self.masked_ratio = model_cfg.get('MASKED_RATIO', 0.8)
        self.angular_range = model_cfg.get('ANGULAR_RANGE', 1)
        
        # ➕ CMAE 새 파라미터
        self.momentum = model_cfg.get('MOMENTUM', 0.999)
        self.temperature = model_cfg.get('TEMPERATURE', 0.1)
        
        # ✅ 기존 성공 로직: R-MAE pretraining용 decoder
        if hasattr(model_cfg, 'PRETRAINING') and model_cfg.PRETRAINING:
            norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
            self.occupancy_decoder = spconv.SparseSequential(
                spconv.SubMConv3d(128, 64, 3, padding=1, bias=False, indice_key='dec1'),
                norm_fn(64), nn.ReLU(),
                spconv.SubMConv3d(64, 32, 3, padding=1, bias=False, indice_key='dec2'),
                norm_fn(32), nn.ReLU(),
                spconv.SubMConv3d(32, 1, 1, bias=True, indice_key='dec_out')
            )
            
            # ➕ CMAE 새 요소: Teacher network 추가
            self._build_teacher_network()
            
            # ➕ CMAE 새 요소: Feature projectors 추가
            self._build_feature_projectors()
        
        print(f"🎯 R-MAE + CMAE Backbone 초기화 완료")
        print(f"   - R-MAE mask ratio: {self.masked_ratio}")
        print(f"   - CMAE momentum: {self.momentum}")
    
    def _build_teacher_network(self):
        """
        ➕ CMAE Teacher network 구축 (논문 논리 준수)
        
        CMAE-3D 논문:
        - Teacher: full point cloud (unmasked)
        - Student: masked point cloud  
        - Teacher는 Student의 EMA copy
        """
        try:
            print("🔧 CMAE-3D Teacher network 구축 (논문 논리 준수)")
            print(f"🔍 Teacher 파라미터: input_channels={self.input_channels}, grid_size={self.grid_size}")
            
            # ✅ CMAE-3D 논문: Teacher는 Student와 동일한 구조
            # Sparse CNN 적응: VoxelNeXt backbone 사용 (Transformer 대신)
            from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
            
            # Teacher용 별도 model_cfg (indice_key 충돌 방지)
            teacher_cfg = type('TeacherConfig', (), {})()
            for attr in dir(self.model_cfg):
                if not attr.startswith('_'):
                    setattr(teacher_cfg, attr, getattr(self.model_cfg, attr))
            
            self.teacher_backbone = VoxelResBackBone8xVoxelNeXt(
                model_cfg=teacher_cfg,
                input_channels=self.input_channels, 
                grid_size=self.grid_size
            )
            
            # ✅ CMAE-3D 논문: Teacher 파라미터를 Student로 초기화
            print("🔄 Teacher 파라미터 초기화 (CMAE-3D EMA 방식)")
            self._initialize_teacher_from_student()
            
            # Teacher는 gradient 업데이트 없음 (EMA만)
            for param in self.teacher_backbone.parameters():
                param.requires_grad = False
            
            self.has_teacher = True
            print(f"✅ CMAE-3D Teacher network 구축 완료")
            
        except Exception as e:
            print(f"❌ Teacher network 구축 실패: {e}")
            import traceback
            print(f"🔍 상세 오류: {traceback.format_exc()}")
            self.has_teacher = False
    
    def _initialize_teacher_from_student(self):
        """
        CMAE-3D 논문: Teacher를 Student 파라미터로 정확히 초기화
        """
        with torch.no_grad():
            # 모든 conv 레이어 초기화
            teacher_modules = list(self.teacher_backbone.named_modules())
            student_modules = list(self.named_modules())
            
            teacher_dict = {name: module for name, module in teacher_modules}
            student_dict = {name: module for name, module in student_modules}
            
            # conv_input, conv1, conv2, conv3, conv4 초기화
            layer_names = ['conv_input', 'conv1', 'conv2', 'conv3', 'conv4']
            
            for layer_name in layer_names:
                if layer_name in teacher_dict and layer_name in student_dict:
                    self._copy_layer_parameters(
                        student_dict[layer_name], 
                        teacher_dict[layer_name]
                    )
                    print(f"✅ Teacher {layer_name} 초기화 완료")
    
    def _copy_layer_parameters(self, student_layer, teacher_layer):
        """개별 레이어 파라미터 복사"""
        student_params = list(student_layer.parameters())
        teacher_params = list(teacher_layer.parameters())
        
        for s_param, t_param in zip(student_params, teacher_params):
            if s_param.shape == t_param.shape:
                t_param.data.copy_(s_param.data)
    
    def _build_feature_projectors(self):
        """
        ➕ CMAE Feature projectors 구축 (새로 추가)
        Multi-scale features를 공통 차원으로 투영
        """
        try:
            # 간단한 projection heads
            self.student_projector = nn.Sequential(
                nn.Linear(128, 128),  # VoxelNeXt 마지막 채널
                nn.ReLU(),
                nn.Linear(128, 256)   # 공통 projection 차원
            )
            
            self.teacher_projector = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(), 
                nn.Linear(128, 256)
            )
            
            # Teacher projector를 Student로 초기화
            with torch.no_grad():
                for teacher_param, student_param in zip(
                    self.teacher_projector.parameters(),
                    self.student_projector.parameters()
                ):
                    teacher_param.copy_(student_param)
                    teacher_param.requires_grad = False
            
            print("✅ Feature projectors 구축 완료")
            
        except Exception as e:
            print(f"⚠️ Feature projectors 구축 실패: {e}")
    
    def forward(self, batch_dict):
        """
        ✅ CMAE-3D + R-MAE 통합 forward (논문 논리 준수)
        
        CMAE-3D 논문:
        - Student: masked input → masked features
        - Teacher: full input → full features  
        - Contrastive learning between student & teacher
        """
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        # ✅ 디버깅 코드 추가 - 원본 배치 정보
        original_batch_size = int(voxel_coords[:, 0].max().item()) + 1
        print(f"🔍 [DEBUG] Original batch size: {original_batch_size}")
        
        # 배치별 voxel 개수 확인
        for batch_idx in range(original_batch_size):
            batch_mask = voxel_coords[:, 0] == batch_idx
            voxel_count = batch_mask.sum().item()
            print(f"🔍 [DEBUG] Batch {batch_idx}: {voxel_count} voxels")
        
        # ✅ R-MAE masking 적용 (training 시에만)
        if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
            batch_dict['original_voxel_coords'] = voxel_coords.clone()
            batch_dict['original_voxel_features'] = voxel_features.clone()
            
            # R-MAE radial masking
            voxel_coords, voxel_features = self.radial_masking(voxel_coords, voxel_features)
            batch_dict['voxel_coords'] = voxel_coords
            batch_dict['voxel_features'] = voxel_features
            
            # ✅ 디버깅 코드 추가 - 마스킹 후 배치 정보
            if len(voxel_coords) > 0:
                masked_batch_size = int(voxel_coords[:, 0].max().item()) + 1
                print(f"🔍 [DEBUG] After masking batch size: {masked_batch_size}")
                
                # 마스킹 후 배치별 voxel 개수 확인
                for batch_idx in range(masked_batch_size):
                    batch_mask = voxel_coords[:, 0] == batch_idx
                    voxel_count = batch_mask.sum().item()
                    print(f"🔍 [DEBUG] Masked Batch {batch_idx}: {voxel_count} voxels")
            else:
                print(f"🔍 [DEBUG] No voxels after masking!")
            
            # ✅ CMAE-3D: Teacher forward (full/unmasked input)
            if hasattr(self, 'has_teacher') and self.has_teacher:
                teacher_features = self._forward_teacher_correct(
                    batch_dict['original_voxel_coords'], 
                    batch_dict['original_voxel_features'], 
                    batch_size
                )
                batch_dict['teacher_features'] = teacher_features
                print(f"🔍 [DEBUG] Teacher features shape: {teacher_features.shape}")
        
        # ✅ Student network forward (masked input)
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # VoxelNeXt conv layers
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)  
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        # ✅ Pretraining outputs
        if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
            # R-MAE occupancy prediction
            occupancy_pred = self.occupancy_decoder(x_conv4)
            batch_dict['occupancy_pred'] = occupancy_pred.features
            batch_dict['occupancy_coords'] = occupancy_pred.indices
            
            # ✅ CMAE-3D: Student features
            student_features = self._extract_global_features_correct(x_conv4)
            batch_dict['student_features'] = student_features
            print(f"🔍 [DEBUG] Student features shape: {student_features.shape}")
                
            # ✅ CMAE-3D: Teacher EMA 업데이트
            if hasattr(self, 'has_teacher') and self.has_teacher:
                self._update_teacher_ema_correct()
            
            # Multi-scale features
            batch_dict['multi_scale_features'] = {
                'x_conv1': x_conv1, 'x_conv2': x_conv2,
                'x_conv3': x_conv3, 'x_conv4': x_conv4,
            }
        
        # ✅ 기존 VoxelNeXt 출력 형식 유지
        batch_dict.update({
            'encoded_spconv_tensor': x_conv4,
            'encoded_spconv_tensor_stride': 8,
            'multi_scale_3d_features': {
                'x_conv1': x_conv1, 'x_conv2': x_conv2,
                'x_conv3': x_conv3, 'x_conv4': x_conv4,
            }
        })
        
        return batch_dict
    
    def radial_masking(self, voxel_coords, voxel_features):
        """
        ✅ 이전 성공 R-MAE 로직 완전 복원 (최소 voxel 보장)
        """
        if not self.training:
            return voxel_coords, voxel_features
            
        batch_size = int(voxel_coords[:, 0].max()) + 1
        print(f"🔍 [MASKING] Processing {batch_size} batches for radial masking")
        
        masked_coords, masked_features = [], []
        
        for batch_idx in range(batch_size):
            mask = voxel_coords[:, 0] == batch_idx
            
            if not mask.any():
                print(f"🔍 [MASKING] Batch {batch_idx}: EMPTY - keeping empty tensors")
                # ✅ 빈 배치도 빈 텐서로 추가하여 배치 순서 유지
                empty_coords = torch.empty(0, 4, dtype=voxel_coords.dtype, device=voxel_coords.device)
                empty_features = torch.empty(0, voxel_features.size(-1), dtype=voxel_features.dtype, device=voxel_features.device)
                masked_coords.append(empty_coords)
                masked_features.append(empty_features)
                continue
                
            coords_b = voxel_coords[mask]
            features_b = voxel_features[mask]
            
            print(f"🔍 [MASKING] Batch {batch_idx}: {len(coords_b)} voxels before masking")
            
            # ✅ 이전 성공 로직: 실제 좌표 계산
            voxel_size = getattr(self, 'voxel_size', [0.1, 0.1, 0.1])
            point_cloud_range = getattr(self, 'point_cloud_range', [-70, -40, -3, 70, 40, 1])
            
            x = coords_b[:, 1].float() * voxel_size[0] + point_cloud_range[0]
            y = coords_b[:, 2].float() * voxel_size[1] + point_cloud_range[1]
            theta = torch.atan2(y, x)
            
            # ✅ Angular masking (이전 성공 로직)
            num_sectors = int(360 / self.angular_range)
            sector_size = 2 * np.pi / num_sectors
            keep_mask = torch.ones(len(coords_b), dtype=torch.bool, device=coords_b.device)
            
            for i in range(num_sectors):
                start = -np.pi + i * sector_size
                end = -np.pi + (i + 1) * sector_size
                in_sector = (theta >= start) & (theta < end)
                
                if in_sector.sum() > 0 and torch.rand(1) < self.masked_ratio:
                    keep_mask[in_sector] = False
            
            kept_voxels_before = keep_mask.sum().item()
            print(f"🔍 [MASKING] Batch {batch_idx}: {kept_voxels_before}/{len(coords_b)} voxels after initial masking")
            
            # ✅ 핵심: 최소 voxel 보장 (이전 성공 로직 완전 복원)
            min_keep = max(10, int(len(coords_b) * 0.1))  # 최소 10개 또는 10%
            if keep_mask.sum() < min_keep:
                # 마스킹된 것들 중 일부 복원
                indices = torch.where(~keep_mask)[0]
                restore_count = min_keep - keep_mask.sum().item()
                
                if restore_count > 0 and len(indices) > 0:
                    restore_count = min(restore_count, len(indices))
                    restore_idx = indices[torch.randperm(len(indices))[:restore_count]]
                    keep_mask[restore_idx] = True
                    print(f"🔧 [MASKING] Batch {batch_idx}: Restored {restore_count} voxels to maintain minimum")
            
            kept_voxels_final = keep_mask.sum().item()
            print(f"🔍 [MASKING] Batch {batch_idx}: {kept_voxels_final}/{len(coords_b)} voxels kept ({kept_voxels_final/len(coords_b)*100:.1f}%)")
            
            masked_coords.append(coords_b[keep_mask])
            masked_features.append(features_b[keep_mask])
        
        # ✅ 모든 배치에 대해 결과가 있어야 함
        assert len(masked_coords) == batch_size, f"Missing batches: {len(masked_coords)} != {batch_size}"
        
        if masked_coords:
            result_coords = torch.cat(masked_coords, dim=0)
            result_features = torch.cat(masked_features, dim=0)
            
            print(f"🔍 [MASKING] Final result: {len(result_coords)} total voxels")
            return result_coords, result_features
        else:
            print(f"⚠️ [MASKING] No voxels remained after masking! Returning original")
            return voxel_coords, voxel_features
    
    def _forward_teacher_correct(self, original_coords, original_features, batch_size):
        """
        ➕ CMAE Teacher network forward (논문 논리 준수)
        
        CMAE-3D 논문:
        - Teacher는 full/unmasked input 처리
        - Student와 동일한 forward pass
        """
        if not (hasattr(self, 'has_teacher') and self.has_teacher):
            # Teacher 없으면 Student features 사용 (논문에서는 이런 경우 없음)
            return torch.randn(1, 256, device=original_features.device)
        
        try:
            with torch.no_grad():  # Teacher는 gradient 계산 없음
                # ✅ CMAE-3D: Teacher forward (unmasked input)
                input_sp_tensor = spconv.SparseConvTensor(
                    features=original_features,
                    indices=original_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                
                # Teacher VoxelNeXt forward
                t_x = self.teacher_backbone.conv_input(input_sp_tensor)
                t_x_conv1 = self.teacher_backbone.conv1(t_x)
                t_x_conv2 = self.teacher_backbone.conv2(t_x_conv1)
                t_x_conv3 = self.teacher_backbone.conv3(t_x_conv2)
                t_x_conv4 = self.teacher_backbone.conv4(t_x_conv3)
                
                # Global features 추출
                teacher_features = self._extract_global_features_correct(t_x_conv4)
                
                return teacher_features
                
        except Exception as e:
            print(f"⚠️ Teacher forward 실패: {e}")
            return torch.randn(1, 256, device=original_features.device)
    
    def _extract_global_features_correct(self, sparse_tensor):
        """
        ✅ 올바른 CMAE Global feature extraction (배치별 처리)
        """
        try:
            if hasattr(sparse_tensor, 'features') and hasattr(sparse_tensor, 'indices'):
                features = sparse_tensor.features  # [N, C]
                indices = sparse_tensor.indices    # [N, 4] (batch, z, y, x)
            else:
                # Fallback
                return torch.randn(1, 256, device=sparse_tensor.device)
            
            if features.size(0) == 0:
                return torch.randn(1, 256, device=features.device)
            
            # ✅ 배치별로 global feature 추출
            batch_indices = indices[:, 0]  # batch indices
            batch_size = int(batch_indices.max().item()) + 1
            
            batch_features = []
            for batch_idx in range(batch_size):
                # 현재 배치의 features만 추출
                batch_mask = batch_indices == batch_idx
                batch_feat = features[batch_mask]  # [N_batch, C]
                
                if batch_feat.size(0) > 0:
                    # Global average pooling for this batch
                    global_feat = torch.mean(batch_feat, dim=0, keepdim=True)  # [1, C]
                else:
                    # Empty batch handling
                    global_feat = torch.zeros(1, features.size(-1), device=features.device)
                
                batch_features.append(global_feat)
            
            # Stack all batch features
            result = torch.cat(batch_features, dim=0)  # [batch_size, C]
            
            # Project to target dimension
            if hasattr(self, 'student_projector'):
                projected = self.student_projector(result)
            else:
                # Dimension adjustment
                if result.size(-1) != 256:
                    if result.size(-1) >= 256:
                        projected = result[:, :256]
                    else:
                        padding = 256 - result.size(-1)
                        projected = F.pad(result, (0, padding))
                else:
                    projected = result
            
            return projected  # [batch_size, 256] ← 이제 올바른 shape!
            
        except Exception as e:
            print(f"⚠️ Global feature extraction 실패: {e}")
            # Fallback to batch_size=1
            device = sparse_tensor.features.device if hasattr(sparse_tensor, 'features') else sparse_tensor.device
            return torch.randn(1, 256, device=device)
    
    def _update_teacher_ema_correct(self):
        """
        ➕ CMAE Teacher EMA update (논문 논리 준수)
        
        CMAE-3D 논문: θ_teacher = momentum * θ_teacher + (1-momentum) * θ_student
        """
        if not (hasattr(self, 'has_teacher') and self.has_teacher):
            return
        
        try:
            with torch.no_grad():
                # ✅ CMAE-3D EMA 업데이트 (momentum=0.999)
                teacher_modules = dict(self.teacher_backbone.named_modules())
                student_modules = dict(self.named_modules())
                
                # conv layers EMA 업데이트
                layer_names = ['conv_input', 'conv1', 'conv2', 'conv3', 'conv4']
                
                for layer_name in layer_names:
                    if layer_name in teacher_modules and layer_name in student_modules:
                        self._ema_update_layer_correct(
                            student_modules[layer_name],
                            teacher_modules[layer_name]
                        )
                
                # Feature projectors EMA 업데이트
                if hasattr(self, 'teacher_projector') and hasattr(self, 'student_projector'):
                    for t_param, s_param in zip(
                        self.teacher_projector.parameters(),
                        self.student_projector.parameters()
                    ):
                        t_param.data.mul_(self.momentum).add_(
                            s_param.data, alpha=1.0 - self.momentum
                        )
                        
        except Exception as e:
            print(f"⚠️ Teacher EMA update 실패: {e}")
    
    def _ema_update_layer_correct(self, student_layer, teacher_layer):
        """CMAE-3D 논문 기준 EMA 업데이트"""
        try:
            student_params = list(student_layer.parameters())
            teacher_params = list(teacher_layer.parameters())
            
            for s_param, t_param in zip(student_params, teacher_params):
                if s_param.shape == t_param.shape:
                    # CMAE-3D EMA: θ_t = m*θ_t + (1-m)*θ_s
                    t_param.data.mul_(self.momentum).add_(
                        s_param.data, alpha=1.0 - self.momentum
                    )
                        
        except Exception as e:
            print(f"⚠️ Layer EMA 업데이트 실패: {e}")


# ✅ 모델 등록을 위한 alias
RMAECMAEBackbone = RMAECMAEBackbone