"""
pcdet/models/detectors/rmae_cmae_voxelnext_fixed.py

✅ 수정된 R-MAE + CMAE-3D VoxelNeXt Detector
- Teacher-Student 구조 완벽 구현
- EMA 업데이트 로직 수정
- Loss 계산 안정화
"""

import torch
import torch.nn as nn
import copy
from .detector3d_template import Detector3DTemplate
from ..model_utils.model_nms_utils import class_agnostic_nms
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class RMAECMAEVoxelNeXt(Detector3DTemplate):
    """✅ R-MAE + CMAE-3D 통합 Detector - Teacher-Student 구조 포함"""
    
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        
        # 기본 모듈 빌드
        self.module_list = self.build_networks()
        
        # ✅ CMAE-3D: Teacher-Student 네트워크 초기화
        self.has_teacher = hasattr(model_cfg, 'CMAE') and model_cfg.CMAE.get('USE_TEACHER_STUDENT', False)
        
        if self.has_teacher:
            # Teacher 네트워크 생성 (Student의 깊은 복사)
            self.teacher_backbone = self._create_teacher_network()
            
            # Teacher는 gradient 계산 안함
            for param in self.teacher_backbone.parameters():
                param.requires_grad = False
                
            # EMA 파라미터
            self.ema_momentum = model_cfg.CMAE.get('EMA_MOMENTUM', 0.999)
            
            print("✅ Teacher-Student Network initialized with EMA momentum:", self.ema_momentum)
    
    def _create_teacher_network(self):
        """Teacher 네트워크 생성 - Student backbone의 깊은 복사"""
        # Student backbone 찾기
        student_backbone = None
        for module in self.module_list:
            if hasattr(module, '__class__') and 'backbone' in module.__class__.__name__.lower():
                student_backbone = module
                break
        
        if student_backbone is None:
            raise ValueError("No backbone found in module_list")
            
        # Teacher는 Student의 깊은 복사
        teacher_backbone = copy.deepcopy(student_backbone)
        
        # Teacher는 eval 모드
        teacher_backbone.eval()
        
        return teacher_backbone
    
    def _update_teacher_ema(self):
        """✅ Teacher 네트워크 EMA 업데이트"""
        if not self.has_teacher or not self.training:
            return
            
        # Student backbone 찾기
        student_backbone = None
        for module in self.module_list:
            if hasattr(module, '__class__') and 'backbone' in module.__class__.__name__.lower():
                student_backbone = module
                break
                
        if student_backbone is None:
            return
            
        # EMA 업데이트: teacher = momentum * teacher + (1-momentum) * student
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_backbone.parameters(), 
                student_backbone.parameters()
            ):
                teacher_param.data.mul_(self.ema_momentum).add_(
                    student_param.data, alpha=1 - self.ema_momentum
                )
    
    def forward(self, batch_dict):
        """✅ 수정된 Forward - Teacher-Student 로직 포함"""
        
        # 1. VFE 처리
        for cur_module in self.module_list:
            if 'vfe' in cur_module.__class__.__name__.lower():
                batch_dict = cur_module(batch_dict)
                break
        
        # 2. Teacher 처리 (training & has_teacher)
        if self.training and self.has_teacher:
            # Teacher는 마스킹 없이 전체 데이터 처리
            with torch.no_grad():
                teacher_batch = {
                    'voxel_features': batch_dict['voxel_features'].clone(),
                    'voxel_coords': batch_dict['voxel_coords'].clone(),
                    'batch_size': batch_dict['batch_size']
                }
                teacher_batch = self.teacher_backbone(teacher_batch)
                
                # Teacher features 저장
                batch_dict['teacher_features'] = {}
                if 'multi_scale_3d_features' in teacher_batch:
                    for key, feat in teacher_batch['multi_scale_3d_features'].items():
                        batch_dict['teacher_features'][key] = feat.features.detach()
        
        # 3. Student backbone 처리 (R-MAE masking 포함)
        for cur_module in self.module_list:
            if hasattr(cur_module, '__class__') and 'backbone' in cur_module.__class__.__name__.lower():
                batch_dict = cur_module(batch_dict)
                break
        
        # 4. Detection head 처리
        if not self.training or not hasattr(self.model_cfg, 'PRETRAINING') or not self.model_cfg.PRETRAINING:
            for cur_module in self.module_list:
                if 'head' in cur_module.__class__.__name__.lower():
                    batch_dict = cur_module(batch_dict)
        
        # 5. Loss 계산
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            
            # EMA 업데이트
            if self.has_teacher:
                self._update_teacher_ema()
            
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
    
    def get_training_loss(self):
        """✅ 통합 손실 함수"""
        disp_dict = {}
        tb_dict = {}
        
        # 기본 손실들
        loss_rpn, tb_dict_rpn = self.dense_head.get_loss() if hasattr(self, 'dense_head') else (0, {})
        loss_rcnn, tb_dict_rcnn = self.roi_head.get_loss(tb_dict) if hasattr(self, 'roi_head') else (0, {})
        
        # RMAE + CMAE 손실들 계산
        loss_rmae = loss_mlfr = loss_hrcl = 0
        
        # 1. R-MAE Occupancy Loss
        if hasattr(self.backbone_3d, 'get_occupancy_loss'):
            loss_rmae, tb_dict_rmae = self.backbone_3d.get_occupancy_loss()
            tb_dict.update(tb_dict_rmae)
        
        # 2. CMAE MLFR Loss
        if hasattr(self.backbone_3d, 'get_mlfr_loss'):
            loss_mlfr, tb_dict_mlfr = self.backbone_3d.get_mlfr_loss()
            tb_dict.update(tb_dict_mlfr)
        
        # 3. CMAE HRCL Loss
        if hasattr(self.backbone_3d, 'get_hrcl_loss'):
            loss_hrcl, tb_dict_hrcl = self.backbone_3d.get_hrcl_loss()
            tb_dict.update(tb_dict_hrcl)
        
        # 총 손실 (논문 수식 기준)
        if hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
            # Pretraining: MAE 손실들만
            loss = (self.model_cfg.LOSS_CONFIG.get('OCCUPANCY_WEIGHT', 1.0) * loss_rmae + 
                   self.model_cfg.LOSS_CONFIG.get('FEATURE_WEIGHT', 0.5) * loss_mlfr + 
                   self.model_cfg.LOSS_CONFIG.get('CONTRASTIVE_WEIGHT', 0.6) * loss_hrcl)
        else:
            # Fine-tuning: Detection + MAE 손실들
            loss = loss_rpn + loss_rcnn + 0.1 * (loss_rmae + loss_mlfr + loss_hrcl)
        
        tb_dict.update({
            'loss_rpn': loss_rpn.item() if isinstance(loss_rpn, torch.Tensor) else loss_rpn,
            'loss_rcnn': loss_rcnn.item() if isinstance(loss_rcnn, torch.Tensor) else loss_rcnn,
            'loss_rmae': loss_rmae.item() if isinstance(loss_rmae, torch.Tensor) else loss_rmae,
            'loss_mlfr': loss_mlfr.item() if isinstance(loss_mlfr, torch.Tensor) else loss_mlfr,
            'loss_hrcl': loss_hrcl.item() if isinstance(loss_hrcl, torch.Tensor) else loss_hrcl,
            'loss_total': loss.item() if isinstance(loss, torch.Tensor) else loss
        })
        
        return loss, tb_dict, disp_dict