# pcdet/models/detectors/rmae_cmae_detector_phase2.py
"""
CMAE-3D Phase 2: Multi-scale Latent Feature Reconstruction Detector

Phase 1 기능 + Phase 2 MLFR 추가:
- Teacher-Student loss 관리 (Phase 1)
- Multi-scale Feature Reconstruction loss 추가 (Phase 2)
- 통합 loss balancing
"""

import torch
import torch.nn.functional as F
from .rmae_cmae_detector_phase1 import RMAECMAEDetectorPhase1


class RMAECMAEDetectorPhase2(RMAECMAEDetectorPhase1):
    """
    🔥 Phase 2: Multi-scale Latent Feature Reconstruction Detector
    
    Phase 1 기능 완전 유지 + CMAE-3D MLFR 추가:
    
    Phase 1 (기존):
    - Teacher-Student Architecture ✅
    - R-MAE loss management ✅
    - Detection loss ✅
    
    Phase 2 (새로 추가):
    - Multi-scale Latent Feature Reconstruction loss 🔥
    - Enhanced loss balancing 🔥
    - Performance monitoring 🔥
    """
    
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        
        # Phase 2 MLFR 설정
        self.enable_mlfr = getattr(model_cfg.BACKBONE_3D, 'ENABLE_MLFR', True)
        
        # 📍 Phase 2 Voxel Contrastive Learning 설정
        self.enable_voxel_contrastive = getattr(model_cfg.BACKBONE_3D, 'ENABLE_VOXEL_CONTRASTIVE', True)
        
        self.phase2_loss_weights = {
            'rmae_weight': model_cfg.get('RMAE_WEIGHT', 1.0),                    # R-MAE occupancy
            'mlfr_weight': model_cfg.get('MLFR_WEIGHT', 1.0),                    # Multi-scale reconstruction  
            'voxel_contrastive_weight': model_cfg.get('VOXEL_CONTRASTIVE_WEIGHT', 0.6),  # Voxel contrastive (CMAE-3D paper)
            'teacher_student_weight': model_cfg.get('TEACHER_STUDENT_WEIGHT', 0.0),     # Phase 3에서 활성화 예정
            'detection_weight': model_cfg.get('DETECTION_WEIGHT', 1.0)          # Detection loss
        }
        
        # Loss storage
        self.forward_ret_dict = {}
        
        print(f"🚀 Phase 2 Detector initialized:")
        print(f"   - Phase 1 features: ✅ Teacher-Student, R-MAE, Detection")
        print(f"   - Phase 2 features: {'✅' if self.enable_mlfr else '❌'} MLFR, {'✅' if self.enable_voxel_contrastive else '❌'} Voxel Contrastive")
        print(f"   - Loss weights: {self.phase2_loss_weights}")
    
    def forward(self, batch_dict):
        """
        Phase 2 Forward:
        1. 모든 모듈 순차 실행 (Phase 1과 동일)
        2. MLFR features 저장 및 관리 (Phase 2 추가)
        3. Enhanced loss 계산
        """
        
        # 기존 모듈 순차 실행 (Phase 1과 동일)
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        
        # Training mode에서만 loss 계산
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # Inference mode (Phase 1과 동일)
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
    
    def get_training_loss(self):
        """
        Phase 2 Training Loss:
        1. R-MAE occupancy loss (Phase 1) ✅
        2. Multi-scale feature reconstruction loss (Phase 2) 🔥
        3. Voxel-level contrastive learning loss (Phase 2) 🔥
        4. Detection loss (fine-tuning 시) ✅
        5. Teacher-Student contrastive loss (Phase 3에서 추가 예정) 🔄
        """
        disp_dict = {}
        loss = 0
        
        # 📍 1. R-MAE Occupancy Loss (Phase 1 유지)
        rmae_loss, rmae_tb_dict = self._get_rmae_loss()
        weighted_rmae_loss = self.phase2_loss_weights['rmae_weight'] * rmae_loss
        loss += weighted_rmae_loss
        disp_dict.update(rmae_tb_dict)
        
        # 📍 2. Multi-scale Latent Feature Reconstruction Loss (Phase 2)
        mlfr_loss, mlfr_tb_dict = self._get_mlfr_loss()
        weighted_mlfr_loss = self.phase2_loss_weights['mlfr_weight'] * mlfr_loss
        loss += weighted_mlfr_loss
        disp_dict.update(mlfr_tb_dict)
        
        # 📍 3. Voxel-level Contrastive Learning Loss (Phase 2 새로 추가)
        voxel_contrastive_loss, voxel_contrastive_tb_dict = self._get_voxel_contrastive_loss()
        weighted_voxel_contrastive_loss = self.phase2_loss_weights['voxel_contrastive_weight'] * voxel_contrastive_loss
        loss += weighted_voxel_contrastive_loss
        disp_dict.update(voxel_contrastive_tb_dict)
        
        # 📍 4. Detection Loss (Fine-tuning 시)
        if not self._is_pretraining_mode():
            det_loss, det_tb_dict = self._get_detection_loss()
            weighted_det_loss = self.phase2_loss_weights['detection_weight'] * det_loss
            loss += weighted_det_loss
            disp_dict.update(det_tb_dict)
        
        # 📍 5. Teacher-Student Contrastive Loss (Phase 2에서는 아직 비활성화)
        if self.enable_teacher_student and self.phase2_loss_weights['teacher_student_weight'] > 0:
            ts_loss, ts_tb_dict = self._get_teacher_student_loss()
            weighted_ts_loss = self.phase2_loss_weights['teacher_student_weight'] * ts_loss
            loss += weighted_ts_loss
            disp_dict.update(ts_tb_dict)
        
        # 📍 6. Phase 2 통합 정보
        disp_dict.update({
            'total_loss': loss.item(),
            'rmae_loss': rmae_loss.item(),
            'mlfr_loss': mlfr_loss.item(),
            'voxel_contrastive_loss': voxel_contrastive_loss.item(),
            'weighted_rmae': weighted_rmae_loss.item(),
            'weighted_mlfr': weighted_mlfr_loss.item(),
            'weighted_voxel_contrastive': weighted_voxel_contrastive_loss.item(),
            'phase2_active': True,
            'mlfr_enabled': self.enable_mlfr,
            'voxel_contrastive_enabled': self.enable_voxel_contrastive
        })
        
        # Tensorboard logging을 위한 tb_dict
        tb_dict = {
            'loss': loss.item(),
            'phase2_total_loss': loss.item(),
            'phase2_rmae_loss': rmae_loss.item(),
            'phase2_mlfr_loss': mlfr_loss.item(),
            'phase2_voxel_contrastive_loss': voxel_contrastive_loss.item(),
            'phase2_rmae_weighted': weighted_rmae_loss.item(),
            'phase2_mlfr_weighted': weighted_mlfr_loss.item(),
            'phase2_voxel_contrastive_weighted': weighted_voxel_contrastive_loss.item(),
            **disp_dict
        }
        
        return loss, tb_dict, disp_dict
    
    def _get_mlfr_loss(self):
        """
        📍 Phase 2: Multi-scale Latent Feature Reconstruction Loss
        
        Backbone에서 계산된 MLFR loss를 가져와서 detector level에서 통합 관리
        """
        tb_dict = {}
        
        # Backbone module에서 MLFR loss 가져오기
        backbone_module = self._get_backbone_module()
        
        if backbone_module is not None and hasattr(backbone_module, 'get_loss'):
            # Backbone의 get_loss에서 MLFR loss 포함된 결과 가져오기
            total_backbone_loss, backbone_tb_dict = backbone_module.get_loss()
            
            # MLFR 관련 정보 추출
            mlfr_loss = backbone_tb_dict.get('mlfr_total_loss', 0.0)
            
            # Scale별 MLFR loss 정보 추가
            for key, value in backbone_tb_dict.items():
                if 'mlfr' in key:
                    tb_dict[key] = value
            
            # MLFR 활성화 상태 기록
            tb_dict['mlfr_backbone_enabled'] = backbone_tb_dict.get('phase2_mlfr', False)
            
            return torch.tensor(mlfr_loss, device='cuda', requires_grad=True), tb_dict
            
        else:
            # Backbone에서 MLFR loss를 가져올 수 없는 경우
            tb_dict['mlfr_error'] = 'backbone_not_found'
            return torch.tensor(0.0, device='cuda', requires_grad=True), tb_dict
    
    def _get_backbone_module(self):
        """Backbone module 찾기"""
        for module in self.module_list:
            if hasattr(module, '__class__') and 'Backbone' in module.__class__.__name__:
                return module
        return None
    
    def _get_voxel_contrastive_loss(self):
        """
        📍 Phase 2: Voxel-level Contrastive Learning Loss
        
        Backbone에서 계산된 voxel contrastive loss를 가져와서 detector level에서 통합 관리
        """
        tb_dict = {}
        
        # Backbone module에서 voxel contrastive loss 가져오기
        backbone_module = self._get_backbone_module()
        
        if backbone_module is not None and hasattr(backbone_module, 'get_loss'):
            # Backbone의 get_loss에서 voxel contrastive loss 포함된 결과 가져오기
            total_backbone_loss, backbone_tb_dict = backbone_module.get_loss()
            
            # Voxel contrastive 관련 정보 추출
            voxel_contrastive_loss = backbone_tb_dict.get('phase2_voxel_contrastive_loss', 0.0)
            
            # Voxel contrastive statistics 추가
            for key, value in backbone_tb_dict.items():
                if 'voxel_contrastive' in key or 'voxel_positive' in key or 'voxel_negative' in key or 'voxel_avg' in key:
                    tb_dict[key] = value
            
            # Voxel contrastive 활성화 상태 기록
            tb_dict['voxel_contrastive_backbone_enabled'] = backbone_tb_dict.get('phase2_voxel_contrastive', False)
            
            return torch.tensor(voxel_contrastive_loss, device='cuda', requires_grad=True), tb_dict
            
        else:
            # Backbone에서 voxel contrastive loss를 가져올 수 없는 경우
            tb_dict['voxel_contrastive_error'] = 'backbone_not_found'
            return torch.tensor(0.0, device='cuda', requires_grad=True), tb_dict
    
    def _get_rmae_loss(self):
        """Phase 1 R-MAE occupancy loss (동일)"""
        tb_dict = {}
        
        # Backbone에서 R-MAE loss 가져오기  
        backbone_module = self._get_backbone_module()
        
        if backbone_module is not None and hasattr(backbone_module, 'get_loss'):
            total_backbone_loss, backbone_tb_dict = backbone_module.get_loss()
            
            # R-MAE 관련 정보 추출
            rmae_loss = backbone_tb_dict.get('phase2_rmae_loss', 0.0)
            
            # R-MAE 관련 tb_dict 추가
            for key, value in backbone_tb_dict.items():
                if 'rmae' in key or 'occupancy' in key:
                    tb_dict[key] = value
                    
            return torch.tensor(rmae_loss, device='cuda', requires_grad=True), tb_dict
        else:
            return torch.tensor(0.0, device='cuda', requires_grad=True), tb_dict
    
    def _get_detection_loss(self):
        """표준 detection loss (Phase 1과 동일)"""
        tb_dict = {}
        loss = 0
        
        # Dense head에서 loss 계산
        for module in self.module_list:
            if hasattr(module, '__class__') and 'Head' in module.__class__.__name__:
                if hasattr(module, 'get_loss'):
                    head_loss, head_tb_dict = module.get_loss()
                    loss += head_loss
                    tb_dict.update(head_tb_dict)
        
        return loss, tb_dict
    
    def _get_teacher_student_loss(self):
        """
        Teacher-Student contrastive loss
        Phase 2에서는 여전히 placeholder (Phase 3에서 구현 예정)
        """
        tb_dict = {
            'teacher_student_loss': 0.0,
            'phase2_ts_placeholder': True,
            'phase3_preview': 'contrastive_learning_coming_soon'
        }
        
        # Phase 3에서 여기에 실제 contrastive loss 구현 예정
        loss = torch.tensor(0.0, device='cuda', requires_grad=True)
        
        return loss, tb_dict
    
    def _is_pretraining_mode(self):
        """현재 pretraining 모드인지 확인"""
        backbone_module = self._get_backbone_module()
        if backbone_module and hasattr(backbone_module, 'model_cfg') and hasattr(backbone_module.model_cfg, 'PRETRAINING'):
            return backbone_module.model_cfg.PRETRAINING
        return False
    
    def get_phase2_status(self):
        """Phase 2 상태 정보 반환 (debugging용)"""
        return {
            'phase2_enabled': True,
            'mlfr_enabled': self.enable_mlfr,
            'voxel_contrastive_enabled': self.enable_voxel_contrastive,
            'loss_weights': self.phase2_loss_weights,
            'backbone_type': type(self._get_backbone_module()).__name__ if self._get_backbone_module() else 'Unknown'
        }
    
    def get_mlfr_features(self):
        """MLFR features 접근용 메서드 (Phase 3 준비)"""
        features = {}
        
        backbone_module = self._get_backbone_module()
        if backbone_module and hasattr(backbone_module, 'forward_ret_dict'):
            ret_dict = backbone_module.forward_ret_dict
            if 'mlfr_results' in ret_dict:
                features['mlfr_results'] = ret_dict['mlfr_results']
            if 'teacher_features' in ret_dict:
                features['teacher_features'] = ret_dict['teacher_features']
            if 'student_multiscale_features' in ret_dict:
                features['student_multiscale_features'] = ret_dict['student_multiscale_features']
            if 'voxel_contrastive_results' in ret_dict:
                features['voxel_contrastive_results'] = ret_dict['voxel_contrastive_results']
        
        return features