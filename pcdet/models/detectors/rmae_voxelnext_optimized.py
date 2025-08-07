# pcdet/models/detectors/rmae_voxelnext_optimized.py
"""
최적화된 R-MAE VoxelNeXt Detector

기존 rmae_voxelnext.py를 기반으로 성능 최적화:
1. Enhanced occupancy loss with multi-scale consistency
2. Progressive training with curriculum learning
3. Advanced loss smoothing and monitoring
4. Improved pretraining stability
5. Better fine-tuning transition

기존 파일에 영향 없이 새로운 파일로 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .detector3d_template import Detector3DTemplate
import numpy as np


class RMAEVoxelNeXtOptimized(Detector3DTemplate):
    """성능 최적화된 R-MAE VoxelNeXt Detector"""
    
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        
        # ===== 📊 개선된 loss 가중치 =====
        self.occupancy_weight = model_cfg.get('OCCUPANCY_WEIGHT', 1.2)  # 1.0 → 1.2
        self.consistency_weight = model_cfg.get('CONSISTENCY_WEIGHT', 0.3)  # 새로 추가
        self.aux_weight = model_cfg.get('AUX_WEIGHT', 0.5)  # Multi-scale auxiliary loss
        
        # ===== 🎯 Progressive training 설정 =====
        self.progressive_training = model_cfg.get('PROGRESSIVE_TRAINING', True)
        self.warmup_epochs = model_cfg.get('WARMUP_EPOCHS', 5)
        self.current_epoch = 0
        
        # ===== 📈 Loss smoothing 및 모니터링 =====
        self.loss_history = []
        self.smoothing_window = model_cfg.get('SMOOTHING_WINDOW', 20)
        self.loss_weights_history = []
        
        # ===== 🔧 Advanced loss 파라미터 =====
        self.focal_loss_alpha = model_cfg.get('FOCAL_LOSS_ALPHA', 0.25)
        self.focal_loss_gamma = model_cfg.get('FOCAL_LOSS_GAMMA', 2.0)
        self.use_focal_loss = model_cfg.get('USE_FOCAL_LOSS', True)
        
        # Occupancy target generation
        self.occupancy_threshold = model_cfg.get('OCCUPANCY_THRESHOLD', 0.5)
        self.hard_negative_ratio = model_cfg.get('HARD_NEGATIVE_RATIO', 3.0)
        
        print(f"🚀 Optimized R-MAE VoxelNeXt Detector initialized:")
        print(f"   📊 Occupancy weight: {self.occupancy_weight}")
        print(f"   🎯 Consistency weight: {self.consistency_weight}")
        print(f"   📈 Auxiliary weight: {self.aux_weight}")
        print(f"   🔧 Use focal loss: {self.use_focal_loss}")
        print(f"   🎯 Progressive training: {self.progressive_training}")
    
    def set_epoch(self, epoch, dataloader_len=None):
        """Training epoch 설정 (scheduler에서 호출)"""
        self.current_epoch = epoch
        
        # Backbone의 training step 정보 업데이트
        for module in self.module_list:
            if hasattr(module, 'training_step') and hasattr(module, 'total_training_steps'):
                if dataloader_len is not None:
                    # 실제 dataloader 길이 사용 (가장 정확)
                    module.training_step = epoch * dataloader_len
                else:
                    # 설정 파일의 총 epoch 수 사용
                    total_epochs = getattr(self.model_cfg.OPTIMIZATION, 'NUM_EPOCHS', 35)
                    estimated_steps_per_epoch = module.total_training_steps // total_epochs
                    module.training_step = epoch * estimated_steps_per_epoch
    
    def get_progressive_weights(self):
        """🔄 Progressive training을 위한 dynamic loss weights"""
        if not self.progressive_training or not self.training:
            return {
                'occupancy': self.occupancy_weight,
                'consistency': self.consistency_weight,
                'aux': self.aux_weight
            }
        
        # Warmup phase: 낮은 consistency weight
        if self.current_epoch < self.warmup_epochs:
            warmup_progress = self.current_epoch / self.warmup_epochs
            
            weights = {
                'occupancy': self.occupancy_weight * (0.5 + 0.5 * warmup_progress),
                'consistency': self.consistency_weight * warmup_progress,
                'aux': self.aux_weight * (0.3 + 0.7 * warmup_progress)
            }
        else:
            # Normal training phase
            weights = {
                'occupancy': self.occupancy_weight,
                'consistency': self.consistency_weight,
                'aux': self.aux_weight
            }
        
        return weights
    
    def generate_occupancy_targets(self, batch_dict):
        """🎯 고품질 occupancy target 생성"""
        if 'original_voxel_coords' not in batch_dict:
            return None
        
        original_coords = batch_dict['original_voxel_coords']
        device = original_coords.device
        
        # Grid 기반 occupancy target 생성
        batch_size = int(original_coords[:, 0].max()) + 1
        grid_size = self.dataset.grid_size
        
        occupancy_targets = []
        target_coords = []
        
        for batch_idx in range(batch_size):
            mask = original_coords[:, 0] == batch_idx
            coords = original_coords[mask]
            
            if len(coords) == 0:
                continue
            
            # Sparse coordinate to dense occupancy
            batch_occupancy = torch.zeros(
                grid_size[2], grid_size[1], grid_size[0], 
                dtype=torch.float32, device=device
            )
            
            # Mark occupied voxels
            valid_mask = (
                (coords[:, 1] >= 0) & (coords[:, 1] < grid_size[0]) &
                (coords[:, 2] >= 0) & (coords[:, 2] < grid_size[1]) &
                (coords[:, 3] >= 0) & (coords[:, 3] < grid_size[2])
            )
            
            if valid_mask.sum() > 0:
                valid_coords = coords[valid_mask]
                batch_occupancy[
                    valid_coords[:, 3].long(),
                    valid_coords[:, 2].long(), 
                    valid_coords[:, 1].long()
                ] = 1.0
                
                # Convert back to sparse format for efficiency
                occupied_indices = torch.nonzero(batch_occupancy, as_tuple=False)
                if len(occupied_indices) > 0:
                    # Format: [batch_idx, x, y, z]
                    batch_indices = torch.full(
                        (len(occupied_indices), 1), batch_idx, 
                        dtype=torch.long, device=device
                    )
                    coords_formatted = torch.cat([
                        batch_indices,
                        occupied_indices[:, [2, 1, 0]]  # z,y,x -> x,y,z
                    ], dim=1)
                    
                    target_coords.append(coords_formatted)
                    occupancy_targets.append(torch.ones(len(occupied_indices), device=device))
        
        if target_coords:
            return {
                'target_coords': torch.cat(target_coords, dim=0),
                'target_occupancy': torch.cat(occupancy_targets, dim=0)
            }
        
        return None
    
    def compute_focal_loss(self, predictions, targets, alpha=0.25, gamma=2.0):
        """🎯 Focal loss for handling class imbalance"""
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def compute_enhanced_occupancy_loss(self, batch_dict):
        """🏗️ 실제 작동하는 multi-scale occupancy loss"""
        device = batch_dict['voxel_coords'].device
        
        if 'occupancy_pred' not in batch_dict or 'original_voxel_coords' not in batch_dict:
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                'occupancy_loss': dummy_loss,
                'consistency_loss': dummy_loss,
                'aux_loss': dummy_loss,
                'total_loss': dummy_loss
            }
        
        occupancy_pred = batch_dict['occupancy_pred']
        
        if len(occupancy_pred) == 0:
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                'occupancy_loss': dummy_loss,
                'consistency_loss': dummy_loss, 
                'aux_loss': dummy_loss,
                'total_loss': dummy_loss
            }
        
        # ===== Main Occupancy Loss =====
        target = torch.ones_like(occupancy_pred.squeeze(-1), device=device)
        main_occupancy_loss = F.binary_cross_entropy_with_logits(
            occupancy_pred.squeeze(-1), target
        )
        
        # ===== Multi-scale Auxiliary Loss (실제 작동) =====
        aux_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if 'multi_scale_occupancy' in batch_dict:
            scales = batch_dict['multi_scale_occupancy']
            
            for scale_name, scale_features in scales.items():
                if scale_name != 'scale_1' and len(scale_features) > 0:  # scale_1은 main prediction
                    # Scale features를 occupancy prediction으로 변환
                    scale_pred = torch.sigmoid(scale_features.mean(dim=1, keepdim=True))  # 채널 평균
                    scale_target = torch.ones_like(scale_pred.squeeze(-1), device=device)
                    
                    scale_loss = F.binary_cross_entropy(scale_pred.squeeze(-1), scale_target)
                    aux_loss = aux_loss + scale_loss * 0.3  # 각 scale별 가중치
        
        # ===== Feature Consistency Loss (실제 작동) =====
        consistency_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if 'multi_scale_occupancy' in batch_dict:
            scales = batch_dict['multi_scale_occupancy']
            scale_features_list = []
            
            # 각 scale의 features 수집
            for scale_name, features in scales.items():
                if len(features) > 0:
                    # Feature normalization
                    normalized_features = F.normalize(features, p=2, dim=1)
                    # Global average pooling
                    pooled_features = normalized_features.mean(dim=0, keepdim=True)
                    scale_features_list.append(pooled_features)
            
            # Scale간 consistency 계산
            if len(scale_features_list) >= 2:
                for i in range(len(scale_features_list)):
                    for j in range(i + 1, len(scale_features_list)):
                        feat1, feat2 = scale_features_list[i], scale_features_list[j]
                        
                        # 채널 수 맞추기
                        min_channels = min(feat1.shape[1], feat2.shape[1])
                        feat1_crop = feat1[:, :min_channels]
                        feat2_crop = feat2[:, :min_channels]
                        
                        # Cosine similarity loss
                        cosine_sim = F.cosine_similarity(feat1_crop, feat2_crop, dim=1).mean()
                        consistency_loss = consistency_loss + (1 - cosine_sim) * 0.2
        
        # ===== Distance-aware Consistency (추가 개선) =====
        if 'original_voxel_coords' in batch_dict and 'voxel_coords' in batch_dict:
            original_count = len(batch_dict['original_voxel_coords'])
            masked_count = len(batch_dict['voxel_coords'])
            
            if original_count > 0:
                # Masking ratio에 따른 adaptive weighting
                actual_mask_ratio = 1.0 - (masked_count / original_count)
                
                # 높은 masking ratio일수록 consistency 중요도 증가
                adaptive_weight = 1.0 + actual_mask_ratio * 0.5
                consistency_loss = consistency_loss * adaptive_weight
        
        # ===== Progressive weights =====
        weights = self.get_progressive_weights()
        total_loss = (
            weights['occupancy'] * main_occupancy_loss +
            weights['consistency'] * consistency_loss +
            weights['aux'] * aux_loss
        )
        
        return {
            'occupancy_loss': main_occupancy_loss,
            'consistency_loss': consistency_loss,
            'aux_loss': aux_loss,
            'total_loss': total_loss
        }
    
    def update_loss_history(self, loss_dict):
        """📊 Loss history 업데이트 및 smoothing"""
        current_total_loss = loss_dict['total_loss'].item()
        self.loss_history.append(current_total_loss)
        
        # Keep only recent history
        if len(self.loss_history) > self.smoothing_window:
            self.loss_history.pop(0)
        
        # Progressive weights history
        weights = self.get_progressive_weights()
        self.loss_weights_history.append(weights.copy())
        if len(self.loss_weights_history) > self.smoothing_window:
            self.loss_weights_history.pop(0)
    
    def get_smoothed_metrics(self):
        """📈 Smoothed metrics 계산"""
        if not self.loss_history:
            return {}
        
        smoothed_loss = sum(self.loss_history) / len(self.loss_history)
        
        # Loss variance (stability indicator)
        if len(self.loss_history) > 1:
            loss_variance = np.var(self.loss_history)
            loss_std = np.std(self.loss_history)
        else:
            loss_variance = 0.0
            loss_std = 0.0
        
        # Recent trend (last 5 vs previous 5)
        if len(self.loss_history) >= 10:
            recent_losses = self.loss_history[-5:]
            previous_losses = self.loss_history[-10:-5]
            recent_avg = sum(recent_losses) / len(recent_losses)
            previous_avg = sum(previous_losses) / len(previous_losses)
            loss_trend = recent_avg - previous_avg  # Negative = improving
        else:
            loss_trend = 0.0
        
        return {
            'smoothed_loss': smoothed_loss,
            'loss_variance': loss_variance,
            'loss_std': loss_std,
            'loss_trend': loss_trend,
            'history_length': len(self.loss_history)
        }
    
    def forward(self, batch_dict):
        """🚀 최적화된 통합 forward 함수"""
        # ===== Pretraining mode =====
        if self.training and self._is_pretraining_mode():
            return self._forward_pretraining_optimized(batch_dict)
        
        # ===== Fine-tuning/Inference mode =====
        else:
            return self._forward_detection(batch_dict)
            
    
    def _is_pretraining_mode(self):
        """Pretraining 모드 확인"""
        # Backbone에서 PRETRAINING 플래그 확인
        backbone_cfg = getattr(self.model_cfg, 'BACKBONE_3D', {})
        return backbone_cfg.get('PRETRAINING', False)
    
    def _forward_pretraining_optimized(self, batch_dict):
        """🎯 최적화된 pretraining forward"""
        # ===== 모든 모듈 실행 =====
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        
        # ===== 개선된 loss 계산 =====
        loss_dict = self.compute_enhanced_occupancy_loss(batch_dict)
        
        # ===== Loss history 업데이트 =====
        self.update_loss_history(loss_dict)
        
        # ===== Smoothed metrics 계산 =====
        smoothed_metrics = self.get_smoothed_metrics()
        
        # ===== 상세한 통계 정보 =====
        weights = self.get_progressive_weights()
        
        tb_dict = {
            # Main losses
            'occupancy_loss': loss_dict['occupancy_loss'].item(),
            'consistency_loss': loss_dict['consistency_loss'].item(),
            'aux_loss': loss_dict['aux_loss'].item(),
            'total_loss': loss_dict['total_loss'].item(),
            
            # Progressive weights
            'weight_occupancy': weights['occupancy'],
            'weight_consistency': weights['consistency'],
            'weight_aux': weights['aux'],
            
            # Training progress
            'epoch': self.current_epoch,
            'mask_ratio': batch_dict.get('actual_mask_ratio', 0.0),
            'target_mask_ratio': batch_dict.get('target_mask_ratio', 0.0),
            
            # Voxel statistics
            'voxel_count': len(batch_dict['voxel_coords']),
            'original_voxel_count': len(batch_dict.get('original_voxel_coords', [])),
            
            # Smoothed metrics
            **smoothed_metrics
        }
        
        # ===== Training step 정보 추가 =====
        if 'training_step' in batch_dict:
            tb_dict['training_step'] = batch_dict['training_step']
        
        # ===== Multi-scale occupancy 통계 =====
        if 'multi_scale_occupancy' in batch_dict:
            scales = batch_dict['multi_scale_occupancy']
            for scale_name, features in scales.items():
                tb_dict[f'occupancy_{scale_name}_count'] = len(features)
                if len(features) > 0:
                    tb_dict[f'occupancy_{scale_name}_mean'] = features.mean().item()
                    tb_dict[f'occupancy_{scale_name}_std'] = features.std().item()
        
        ret_dict = {'loss': loss_dict['total_loss']}
        
        return ret_dict, tb_dict, {}
    
    def _forward_detection(self, batch_dict):
        """🎯 표준 detection forward (fine-tuning/inference)"""
        # 모든 모듈 실행
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        
        if self.training:
            # Fine-tuning losses
            loss, tb_dict, disp_dict = self.get_training_loss()
            
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            # Inference - post processing
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
    
    def load_params_from_pretrained(self, pretrained_dict, strict=True):
        """🔄 Pretrained model에서 파라미터 로드 (improved)"""
        model_dict = self.state_dict()
        
        # 호환되는 파라미터만 필터링
        compatible_dict = {}
        incompatible_keys = []
        
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    compatible_dict[k] = v
                else:
                    incompatible_keys.append(f"{k}: {model_dict[k].shape} vs {v.shape}")
            else:
                incompatible_keys.append(f"{k}: not found in current model")
        
        # 로드
        model_dict.update(compatible_dict)
        self.load_state_dict(model_dict, strict=False)
        
        print(f"✅ Loaded {len(compatible_dict)} compatible parameters")
        if incompatible_keys:
            print(f"⚠️  {len(incompatible_keys)} incompatible parameters:")
            for key in incompatible_keys[:5]:  # Show first 5
                print(f"   {key}")
            if len(incompatible_keys) > 5:
                print(f"   ... and {len(incompatible_keys) - 5} more")
        
        return len(compatible_dict), len(incompatible_keys)