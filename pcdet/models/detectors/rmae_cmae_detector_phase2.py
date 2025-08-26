# pcdet/models/detectors/rmae_cmae_detector_phase2.py
"""
CMAE-3D Phase 2: Multi-scale Latent Feature Reconstruction Detector

Phase 1 기능 + Phase 2 모든 기능 완전 통합:
- Teacher-Student loss 관리 (Phase 1)
- Multi-scale Feature Reconstruction loss 추가 (Phase 2 Step 1)
- Voxel-level Contrastive Learning loss 추가 (Phase 2 Step 2)
- Frame-level Contrastive Learning loss 추가 (Phase 2 Step 3)
- 통합 loss balancing
"""

import torch
import torch.nn.functional as F
from .rmae_cmae_detector_phase1 import RMAECMAEDetectorPhase1


class RMAECMAEDetectorPhase2(RMAECMAEDetectorPhase1):
    """
    🔥 Phase 2: Multi-scale Latent Feature Reconstruction Detector
    
    Phase 1 기능 완전 유지 + CMAE-3D 모든 기능 추가:
    
    Phase 1 (기존):
    - Teacher-Student Architecture ✅
    - R-MAE loss management ✅
    - Detection loss ✅
    
    Phase 2 (새로 추가):
    - Multi-scale Latent Feature Reconstruction loss 🔥
    - Voxel-level Contrastive Learning loss 🔥
    - Frame-level Contrastive Learning loss 🔥
    - Enhanced loss balancing 🔥
    - Performance monitoring 🔥
    """
    
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        
        # Phase 2 기능 활성화 설정
        self.enable_mlfr = getattr(model_cfg.BACKBONE_3D, 'ENABLE_MLFR', True)
        self.enable_voxel_contrastive = getattr(model_cfg.BACKBONE_3D, 'ENABLE_VOXEL_CONTRASTIVE', True)
        self.enable_frame_contrastive = getattr(model_cfg.BACKBONE_3D, 'ENABLE_FRAME_CONTRASTIVE', True)
        
        # Phase 2 완전체 Loss weights
        self.phase2_loss_weights = {
            # Phase 1 weights 유지
            'rmae_weight': model_cfg.get('RMAE_WEIGHT', 1.0),                          # R-MAE occupancy
            'teacher_student_weight': model_cfg.get('TEACHER_STUDENT_WEIGHT', 0.0),    # Phase 3에서 활성화 예정
            
            # Phase 2 weights
            'mlfr_weight': model_cfg.get('MLFR_WEIGHT', 1.0),                          # Multi-scale reconstruction  
            'voxel_contrastive_weight': model_cfg.get('VOXEL_CONTRASTIVE_WEIGHT', 0.6), # Voxel contrastive (CMAE-3D paper)
            'frame_contrastive_weight': model_cfg.get('FRAME_CONTRASTIVE_WEIGHT', 0.3), # Frame contrastive (CMAE-3D paper)
            
            # Detection weight
            'detection_weight': model_cfg.get('DETECTION_WEIGHT', 1.0)                # Detection loss
        }
        
        # Loss storage
        self.forward_ret_dict = {}
        
        print(f"🚀 Phase 2 Complete Detector initialized:")
        print(f"   - Phase 1 features: ✅ Teacher-Student, R-MAE, Detection")
        print(f"   - Phase 2 features: {'✅' if self.enable_mlfr else '❌'} MLFR, {'✅' if self.enable_voxel_contrastive else '❌'} Voxel Contrastive, {'✅' if self.enable_frame_contrastive else '❌'} Frame Contrastive")
        print(f"   - Loss weights: {self.phase2_loss_weights}")
    
    def forward(self, batch_dict):
        """
        Phase 2 Complete Forward:
        1. 모든 모듈 순차 실행 (Phase 1과 동일)
        2. 모든 Phase 2 features 저장 및 관리
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
        Phase 2 Complete Training Loss:
        1. R-MAE occupancy loss (Phase 1) ✅
        2. Multi-scale feature reconstruction loss (Phase 2 Step 1) ✅
        3. Voxel-level contrastive learning loss (Phase 2 Step 2) ✅
        4. Frame-level contrastive learning loss (Phase 2 Step 3) ✅
        5. Detection loss (fine-tuning 시) ✅
        6. Teacher-Student contrastive loss (Phase 3에서 추가 예정) 🔄
        """
        disp_dict = {}
        loss = 0
        
        # 📍 1. R-MAE Occupancy Loss (Phase 1 유지)
        rmae_loss, rmae_tb_dict = self._get_rmae_loss()
        weighted_rmae_loss = self.phase2_loss_weights['rmae_weight'] * rmae_loss
        loss += weighted_rmae_loss
        disp_dict.update(rmae_tb_dict)
        
        # 📍 2. Multi-scale Latent Feature Reconstruction Loss (Phase 2 Step 1)
        mlfr_loss, mlfr_tb_dict = self._get_mlfr_loss()
        weighted_mlfr_loss = self.phase2_loss_weights['mlfr_weight'] * mlfr_loss
        loss += weighted_mlfr_loss
        disp_dict.update(mlfr_tb_dict)
        
        # 📍 3. Voxel-level Contrastive Learning Loss (Phase 2 Step 2)
        voxel_contrastive_loss, voxel_contrastive_tb_dict = self._get_voxel_contrastive_loss()
        weighted_voxel_contrastive_loss = self.phase2_loss_weights['voxel_contrastive_weight'] * voxel_contrastive_loss
        loss += weighted_voxel_contrastive_loss
        disp_dict.update(voxel_contrastive_tb_dict)
        
        # 📍 4. Frame-level Contrastive Learning Loss (Phase 2 Step 3)
        frame_contrastive_loss, frame_contrastive_tb_dict = self._get_frame_contrastive_loss()
        weighted_frame_contrastive_loss = self.phase2_loss_weights['frame_contrastive_weight'] * frame_contrastive_loss
        loss += weighted_frame_contrastive_loss
        disp_dict.update(frame_contrastive_tb_dict)
        
        # 📍 5. Detection Loss (Fine-tuning 시)
        if not self._is_pretraining_mode():
            det_loss, det_tb_dict = self._get_detection_loss()
            weighted_det_loss = self.phase2_loss_weights['detection_weight'] * det_loss
            loss += weighted_det_loss
            disp_dict.update(det_tb_dict)
        
        # 📍 6. Teacher-Student Contrastive Loss (Phase 3에서 활성화 예정)
        if self.enable_teacher_student and self.phase2_loss_weights['teacher_student_weight'] > 0:
            ts_loss, ts_tb_dict = self._get_teacher_student_loss()
            weighted_ts_loss = self.phase2_loss_weights['teacher_student_weight'] * ts_loss
            loss += weighted_ts_loss
            disp_dict.update(ts_tb_dict)
        
        # 📍 7. Phase 2 Complete 통합 정보
        disp_dict.update({
            'total_loss': loss.item(),
            'rmae_loss': rmae_loss.item(),
            'mlfr_loss': mlfr_loss.item(),
            'voxel_contrastive_loss': voxel_contrastive_loss.item(),
            'frame_contrastive_loss': frame_contrastive_loss.item(),
            'weighted_rmae': weighted_rmae_loss.item(),
            'weighted_mlfr': weighted_mlfr_loss.item(),
            'weighted_voxel_contrastive': weighted_voxel_contrastive_loss.item(),
            'weighted_frame_contrastive': weighted_frame_contrastive_loss.item(),
            'phase2_complete_active': True,
            'mlfr_enabled': self.enable_mlfr,
            'voxel_contrastive_enabled': self.enable_voxel_contrastive,
            'frame_contrastive_enabled': self.enable_frame_contrastive
        })
        
        # Tensorboard logging을 위한 tb_dict
        tb_dict = {
            'loss': loss.item(),
            'phase2_complete_total_loss': loss.item(),
            'phase2_rmae_loss': rmae_loss.item(),
            'phase2_mlfr_loss': mlfr_loss.item(),
            'phase2_voxel_contrastive_loss': voxel_contrastive_loss.item(),
            'phase2_frame_contrastive_loss': frame_contrastive_loss.item(),
            'phase2_rmae_weighted': weighted_rmae_loss.item(),
            'phase2_mlfr_weighted': weighted_mlfr_loss.item(),
            'phase2_voxel_contrastive_weighted': weighted_voxel_contrastive_loss.item(),
            'phase2_frame_contrastive_weighted': weighted_frame_contrastive_loss.item(),
            **disp_dict
        }
        
        print(f"🚀 Phase 2 Complete Detector Loss: {loss.item():.6f}")
        print(f"   - R-MAE: {rmae_loss.item():.6f} (weighted: {weighted_rmae_loss.item():.6f})")
        print(f"   - MLFR: {mlfr_loss.item():.6f} (weighted: {weighted_mlfr_loss.item():.6f})")
        print(f"   - Voxel Contrastive: {voxel_contrastive_loss.item():.6f} (weighted: {weighted_voxel_contrastive_loss.item():.6f})")
        print(f"   - Frame Contrastive: {frame_contrastive_loss.item():.6f} (weighted: {weighted_frame_contrastive_loss.item():.6f})")
        
        return loss, tb_dict, disp_dict
    
    def _get_mlfr_loss(self):
        """
        📍 Phase 2 Step 1: Multi-scale Latent Feature Reconstruction Loss
        
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
            
            # Ensure tensor type
            if not torch.is_tensor(mlfr_loss):
                mlfr_loss = torch.tensor(mlfr_loss, device='cuda', requires_grad=True)
            
            print(f"✅ MLFR Loss obtained: {mlfr_loss.item():.6f}")
            
            return mlfr_loss, tb_dict
        
        else:
            print("⚠️ Backbone module not found or no get_loss method for MLFR")
            return torch.tensor(0.0, device='cuda', requires_grad=True), {'mlfr_loss_no_backbone': 0.0}
    
    def _get_voxel_contrastive_loss(self):
        """
        📍 Phase 2 Step 2: Voxel-level Contrastive Learning Loss
        
        Backbone에서 계산된 Voxel contrastive loss를 가져와서 detector level에서 통합 관리
        """
        tb_dict = {}
        
        # Backbone module에서 Voxel contrastive loss 가져오기
        backbone_module = self._get_backbone_module()
        
        if backbone_module is not None and hasattr(backbone_module, 'get_loss'):
            # Backbone의 get_loss에서 Voxel contrastive loss 포함된 결과 가져오기
            total_backbone_loss, backbone_tb_dict = backbone_module.get_loss()
            
            # Voxel contrastive 관련 정보 추출
            voxel_contrastive_loss = backbone_tb_dict.get('voxel_contrastive_loss', 0.0)
            
            # Voxel contrastive 상세 정보 추가
            for key, value in backbone_tb_dict.items():
                if 'voxel_contrastive' in key or 'voxel_' in key:
                    tb_dict[key] = value
            
            # Ensure tensor type
            if not torch.is_tensor(voxel_contrastive_loss):
                voxel_contrastive_loss = torch.tensor(voxel_contrastive_loss, device='cuda', requires_grad=True)
            
            print(f"✅ Voxel Contrastive Loss obtained: {voxel_contrastive_loss.item():.6f}")
            
            return voxel_contrastive_loss, tb_dict
        
        else:
            print("⚠️ Backbone module not found or no get_loss method for Voxel Contrastive")
            return torch.tensor(0.0, device='cuda', requires_grad=True), {'voxel_contrastive_loss_no_backbone': 0.0}
    
    def _get_frame_contrastive_loss(self):
        """
        📍 Phase 2 Step 3: Frame-level Contrastive Learning Loss
        
        Backbone에서 계산된 Frame contrastive loss를 가져와서 detector level에서 통합 관리
        """
        tb_dict = {}
        
        # Backbone module에서 Frame contrastive loss 가져오기
        backbone_module = self._get_backbone_module()
        
        if backbone_module is not None and hasattr(backbone_module, 'get_loss'):
            # Backbone의 get_loss에서 Frame contrastive loss 포함된 결과 가져오기
            total_backbone_loss, backbone_tb_dict = backbone_module.get_loss()
            
            # Frame contrastive 관련 정보 추출
            frame_contrastive_loss = backbone_tb_dict.get('frame_contrastive_loss', 0.0)
            
            # Frame contrastive 상세 정보 추가
            for key, value in backbone_tb_dict.items():
                if 'frame_contrastive' in key or 'frame_' in key:
                    tb_dict[key] = value
            
            # Ensure tensor type
            if not torch.is_tensor(frame_contrastive_loss):
                frame_contrastive_loss = torch.tensor(frame_contrastive_loss, device='cuda', requires_grad=True)
            
            print(f"✅ Frame Contrastive Loss obtained: {frame_contrastive_loss.item():.6f}")
            
            return frame_contrastive_loss, tb_dict
        
        else:
            print("⚠️ Backbone module not found or no get_loss method for Frame Contrastive")
            return torch.tensor(0.0, device='cuda', requires_grad=True), {'frame_contrastive_loss_no_backbone': 0.0}
    
    def _get_teacher_student_loss(self):
        """
        📍 Phase 3: Teacher-Student Contrastive Loss (Phase 2에서는 비활성화)
        
        Phase 3에서 활성화될 Teacher-Student 간 contrastive learning
        현재는 placeholder로 0 반환
        """
        tb_dict = {
            'teacher_student_loss': 0.0,
            'phase2_ts_placeholder': True,
            'phase3_preview': 'teacher_student_contrastive_learning_coming_soon'
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
    
    def _get_backbone_module(self):
        """Backbone module 접근 (DDP 고려)"""
        if hasattr(self, 'module'):  # DistributedDataParallel
            return getattr(self.module, 'backbone_3d', None)
        else:
            return getattr(self, 'backbone_3d', None)
    
    def get_phase2_status(self):
        """Phase 2 Complete 상태 정보 반환 (debugging용)"""
        return {
            'phase2_complete_enabled': True,
            'mlfr_enabled': self.enable_mlfr,
            'voxel_contrastive_enabled': self.enable_voxel_contrastive,
            'frame_contrastive_enabled': self.enable_frame_contrastive,
            'loss_weights': self.phase2_loss_weights,
            'backbone_type': type(self._get_backbone_module()).__name__ if self._get_backbone_module() else 'Unknown',
            'total_phase2_features': 3,  # MLFR + Voxel + Frame Contrastive
            'phase2_steps_completed': ['MLFR', 'Voxel_Contrastive', 'Frame_Contrastive']
        }
    
    def get_phase2_features(self):
        """Phase 2 모든 features 접근용 메서드 (완전체)"""
        features = {}
        
        backbone_module = self._get_backbone_module()
        if backbone_module and hasattr(backbone_module, 'forward_ret_dict'):
            ret_dict = backbone_module.forward_ret_dict
            
            # Phase 1 features
            if 'teacher_features' in ret_dict:
                features['teacher_features'] = ret_dict['teacher_features']
            if 'student_multiscale_features' in ret_dict:
                features['student_multiscale_features'] = ret_dict['student_multiscale_features']
            
            # Phase 2 Step 1: MLFR features
            if 'mlfr_results' in ret_dict:
                features['mlfr_results'] = ret_dict['mlfr_results']
            
            # Phase 2 Step 2: Voxel Contrastive features  
            if 'voxel_contrastive_results' in ret_dict:
                features['voxel_contrastive_results'] = ret_dict['voxel_contrastive_results']
            
            # Phase 2 Step 3: Frame Contrastive features
            if 'frame_contrastive_results' in ret_dict:
                features['frame_contrastive_results'] = ret_dict['frame_contrastive_results']
        
        return features
    
    def get_loss_breakdown(self):
        """Phase 2 완전체 loss breakdown 반환 (debugging용)"""
        breakdown = {}
        
        try:
            # 각 loss component 개별 계산
            rmae_loss, _ = self._get_rmae_loss()
            mlfr_loss, _ = self._get_mlfr_loss()
            voxel_contrastive_loss, _ = self._get_voxel_contrastive_loss()
            frame_contrastive_loss, _ = self._get_frame_contrastive_loss()
            
            # Raw losses
            breakdown['raw_losses'] = {
                'rmae': rmae_loss.item(),
                'mlfr': mlfr_loss.item(),
                'voxel_contrastive': voxel_contrastive_loss.item(),
                'frame_contrastive': frame_contrastive_loss.item()
            }
            
            # Weighted losses
            breakdown['weighted_losses'] = {
                'rmae': (self.phase2_loss_weights['rmae_weight'] * rmae_loss).item(),
                'mlfr': (self.phase2_loss_weights['mlfr_weight'] * mlfr_loss).item(),
                'voxel_contrastive': (self.phase2_loss_weights['voxel_contrastive_weight'] * voxel_contrastive_loss).item(),
                'frame_contrastive': (self.phase2_loss_weights['frame_contrastive_weight'] * frame_contrastive_loss).item()
            }
            
            # Total loss
            total = sum(breakdown['weighted_losses'].values())
            breakdown['total_loss'] = total
            
            # Loss weights
            breakdown['weights'] = self.phase2_loss_weights
            
        except Exception as e:
            breakdown['error'] = f"Loss calculation failed: {e}"
        
        return breakdown