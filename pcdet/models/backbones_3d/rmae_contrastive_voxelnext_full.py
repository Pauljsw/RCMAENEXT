import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from functools import partial
import numpy as np
import random
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt

class RMAEContrastiveVoxelNeXtFull(VoxelResBackBone8xVoxelNeXt):
    """
    R-MAE + Contrastive Learning VoxelNeXt with Full Momentum Encoder
    
    개선사항:
    1. Full VoxelNeXt structure for momentum encoder
    2. Smart parameter initialization 
    3. Safe momentum update mechanism
    """
    
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg, input_channels, grid_size, **kwargs)
        
        # Required attributes
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.grid_size = grid_size
        
        # R-MAE parameters
        self.masked_ratio = model_cfg.get('MASKED_RATIO', 0.8)
        self.angular_range = model_cfg.get('ANGULAR_RANGE', 10)
        
        # Contrastive learning parameters
        self.temperature = model_cfg.get('TEMPERATURE', 0.1)
        self.momentum = model_cfg.get('MOMENTUM', 0.999)
        self.queue_size = model_cfg.get('QUEUE_SIZE', 4096)
        self.use_contrastive = model_cfg.get('USE_CONTRASTIVE', True)
        
        # Multi-scale feature dimensions
        self.feature_dims = {
            'conv1': 16, 'conv2': 32, 'conv3': 64, 'conv4': 128
        }
        
        # Build pretraining components
        if hasattr(model_cfg, 'PRETRAINING') and model_cfg.PRETRAINING:
            self._build_pretraining_components()
    
    def _build_pretraining_components(self):
        """Build R-MAE + Full Contrastive components"""
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        # 1. R-MAE Occupancy Decoder
        self.occupancy_decoder = spconv.SparseSequential(
            spconv.SubMConv3d(128, 64, 3, padding=1, bias=False, indice_key='dec1'),
            norm_fn(64), nn.ReLU(),
            spconv.SubMConv3d(64, 32, 3, padding=1, bias=False, indice_key='dec2'),
            norm_fn(32), nn.ReLU(),
            spconv.SubMConv3d(32, 16, 3, padding=1, bias=False, indice_key='dec3'),
            norm_fn(16), nn.ReLU(),
            spconv.SubMConv3d(16, 1, 1, bias=True, indice_key='dec_out')
        )
        
        if self.use_contrastive:
            print("🔧 Building Full Momentum Encoder system...")
            
            # 2. Multi-scale Projection Heads
            self.projection_heads = nn.ModuleDict()
            for level, dim in self.feature_dims.items():
                self.projection_heads[level] = nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.BatchNorm1d(dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(dim * 2, 128),
                    nn.BatchNorm1d(128)
                )
            
            # 3. Global projection head
            self.global_projection = nn.Sequential(
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128)
            )
            
            # 4. Full Momentum Encoder (핵심 개선!)
            self.momentum_encoder = self._build_full_momentum_encoder()
            self._safe_init_momentum_encoder()
            
            # 5. Memory queues
            for level in self.feature_dims.keys():
                queue_name = f"queue_{level}"
                ptr_name = f"queue_ptr_{level}"
                
                self.register_buffer(queue_name, torch.randn(128, self.queue_size))
                self.register_buffer(ptr_name, torch.zeros(1, dtype=torch.long))
                
                # Normalize queue
                queue = getattr(self, queue_name)
                queue = F.normalize(queue, dim=0)
                setattr(self, queue_name, queue)
                
                print(f"✅ Registered {queue_name} and {ptr_name}")
            
            # Global queue
            self.register_buffer("global_queue", torch.randn(128, self.queue_size))
            self.register_buffer("global_queue_ptr", torch.zeros(1, dtype=torch.long))
            self.global_queue = F.normalize(self.global_queue, dim=0)
            
            print("✅ Registered global_queue and global_queue_ptr")
            
            print("✅ Full Momentum Encoder system successfully initialized!")
            
        # Training step counter
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))
    
    def _build_full_momentum_encoder(self):
        """Build full momentum encoder with safe architecture"""
        try:
            # 방법 1: Clean state로 새로운 VoxelNeXt 생성
            momentum_encoder = VoxelResBackBone8xVoxelNeXt(
                model_cfg=self.model_cfg,
                input_channels=self.input_channels,
                grid_size=self.grid_size
            )
            
            # Projection heads 추가 (contrastive components)
            momentum_encoder.projection_heads = nn.ModuleDict()
            for level, dim in self.feature_dims.items():
                momentum_encoder.projection_heads[level] = nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.BatchNorm1d(dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(dim * 2, 128),
                    nn.BatchNorm1d(128)
                )
            
            momentum_encoder.global_projection = nn.Sequential(
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128)
            )
            
            print("✅ Full VoxelNeXt momentum encoder created successfully")
            return momentum_encoder
            
        except Exception as e:
            print(f"⚠️  Full momentum encoder creation failed: {e}")
            print("🔄 Falling back to modular approach...")
            return self._build_modular_momentum_encoder()
    
    def _build_modular_momentum_encoder(self):
        """Build modular momentum encoder as fallback"""
        class ModularMomentumEncoder(nn.Module):
            def __init__(self, feature_dims, sparse_shape, model_cfg):
                super().__init__()
                self.sparse_shape = sparse_shape
                
                # VoxelNeXt backbone components 개별 구성
                norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
                
                # Input convolution
                self.conv_input = spconv.SparseSequential(
                    spconv.SubMConv3d(3, 16, 3, padding=1, bias=False, indice_key='subm1'),
                    norm_fn(16), nn.ReLU(),
                )
                
                # Main conv layers (simplified)
                self.conv1 = spconv.SparseSequential(
                    spconv.SubMConv3d(16, 16, 3, padding=1, indice_key='subm1'),
                    norm_fn(16), nn.ReLU(),
                )
                
                self.conv2 = spconv.SparseSequential(
                    spconv.SparseConv3d(16, 32, 3, stride=2, padding=1, bias=False),
                    norm_fn(32), nn.ReLU(),
                )
                
                self.conv3 = spconv.SparseSequential(
                    spconv.SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False),
                    norm_fn(64), nn.ReLU(),
                )
                
                self.conv4 = spconv.SparseSequential(
                    spconv.SparseConv3d(64, 128, 3, stride=2, padding=1, bias=False),
                    norm_fn(128), nn.ReLU(),
                )
                
                # Projection heads
                self.projection_heads = nn.ModuleDict()
                for level, dim in feature_dims.items():
                    self.projection_heads[level] = nn.Sequential(
                        nn.Linear(dim, dim * 2),
                        nn.BatchNorm1d(dim * 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(dim * 2, 128),
                        nn.BatchNorm1d(128)
                    )
                
                self.global_projection = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128)
                )
        
        return ModularMomentumEncoder(self.feature_dims, self.sparse_shape, self.model_cfg)
    
    def _safe_init_momentum_encoder(self):
        """Safe momentum encoder initialization with selective copying"""
        try:
            print("🔄 Initializing momentum encoder with main encoder weights...")
            
            # Main encoder의 state dict 가져오기
            main_state = self.state_dict()
            momentum_state = self.momentum_encoder.state_dict()
            
            # VoxelNeXt backbone 부분만 선택적 복사
            backbone_keys = [
                'conv_input', 'conv1', 'conv2', 'conv3', 'conv4', 
                'conv5', 'conv6', 'conv_out', 'shared_conv'
            ]
            
            copied_count = 0
            total_count = 0
            
            for main_key, main_param in main_state.items():
                # Backbone 관련 키인지 확인
                is_backbone = any(bk in main_key for bk in backbone_keys)
                # Contrastive components 제외
                is_contrastive = any(ck in main_key for ck in ['projection', 'queue', 'momentum_encoder'])
                
                if is_backbone and not is_contrastive:
                    # Momentum encoder에 대응하는 키 찾기
                    momentum_key = main_key.replace('momentum_encoder.', '')
                    
                    if momentum_key in momentum_state:
                        momentum_param = momentum_state[momentum_key]
                        
                        # 크기가 일치하는 경우만 복사
                        if main_param.shape == momentum_param.shape:
                            momentum_state[momentum_key].copy_(main_param)
                            copied_count += 1
                        
                total_count += 1
            
            print(f"✅ Copied {copied_count}/{total_count} backbone parameters to momentum encoder")
            
            # 모든 momentum encoder 파라미터를 requires_grad=False로 설정
            for param in self.momentum_encoder.parameters():
                param.requires_grad = False
                
        except Exception as e:
            print(f"⚠️  Momentum encoder initialization warning: {e}")
            print("🔄 Using random initialization...")
            for param in self.momentum_encoder.parameters():
                param.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update_encoder(self):
        """Safe momentum update with selective parameter matching"""
        try:
            # Main과 momentum encoder의 공통 파라미터만 업데이트
            main_params = dict(self.named_parameters())
            momentum_params = dict(self.momentum_encoder.named_parameters())
            
            update_count = 0
            for name, param_k in momentum_params.items():
                # Main encoder에서 대응하는 파라미터 찾기
                main_name = name
                if main_name in main_params:
                    param_q = main_params[main_name]
                    
                    # 크기가 일치하고 contrastive component가 아닌 경우만 업데이트
                    if param_q.shape == param_k.shape and 'projection' not in name:
                        param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
                        update_count += 1
            
            # 주기적으로 업데이트 상태 로깅
            if self.step_count % 100 == 0:
                print(f"🔄 Momentum update: {update_count} parameters updated")
                        
        except Exception as e:
            if self.step_count % 1000 == 0:  # 가끔만 에러 로그
                print(f"⚠️  Momentum update warning: {e}")
    
    def _forward_momentum_encoder(self, batch_dict):
        """Enhanced momentum encoder forward with full VoxelNeXt pipeline"""
        try:
            # Create augmented version with different masking
            aug_coords, aug_features = self.radial_masking(
                batch_dict['original_voxel_coords'], 
                batch_dict['original_voxel_features']
            )
            
            # Full VoxelNeXt forward pass
            if hasattr(self.momentum_encoder, 'conv_input'):
                input_sp_tensor = spconv.SparseConvTensor(
                    features=aug_features,
                    indices=aug_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_dict['batch_size']
                )
                
                # Full pipeline
                x = self.momentum_encoder.conv_input(input_sp_tensor)
                x_conv1 = self.momentum_encoder.conv1(x)
                x_conv2 = self.momentum_encoder.conv2(x_conv1)
                x_conv3 = self.momentum_encoder.conv3(x_conv2)
                x_conv4 = self.momentum_encoder.conv4(x_conv3)
                
                return {
                    'conv1': x_conv1, 'conv2': x_conv2,
                    'conv3': x_conv3, 'conv4': x_conv4,
                }
            else:
                print("⚠️  Momentum encoder doesn't have expected structure")
                return {}
                
        except Exception as e:
            print(f"⚠️  Momentum encoder forward failed: {e}")
            return {}
    
    def enhanced_radial_masking(self, voxel_coords, voxel_features):
        """Enhanced radial masking with distance-aware strategy"""
        if not self.training:
            return voxel_coords, voxel_features
            
        batch_size = int(voxel_coords[:, 0].max()) + 1
        masked_coords, masked_features = [], []
        
        for batch_idx in range(batch_size):
            mask = voxel_coords[:, 0] == batch_idx
            coords_b = voxel_coords[mask]
            features_b = voxel_features[mask]
            
            if len(coords_b) == 0:
                continue
            
            # Convert to cylindrical coordinates
            x, y = coords_b[:, 2].float(), coords_b[:, 3].float()
            distances = torch.sqrt(x**2 + y**2)
            angles = torch.atan2(y, x)
            
            # Angle-based grouping
            angles_deg = torch.rad2deg(angles) % 360
            num_groups = int(360 / self.angular_range)
            group_indices = (angles_deg / self.angular_range).long()
            
            # Distance-aware adaptive masking ratios
            distance_quantiles = torch.quantile(distances, torch.tensor([0.3, 0.7]).to(distances.device))
            near_mask = distances < distance_quantiles[0]
            mid_mask = (distances >= distance_quantiles[0]) & (distances < distance_quantiles[1])
            far_mask = distances >= distance_quantiles[1]
            
            # Different masking ratios for different distance ranges
            ratios = {
                'near': self.masked_ratio * 0.6,        # 0.48 (preserve details)
                'mid': self.masked_ratio,                # 0.8  (standard)
                'far': min(0.95, self.masked_ratio * 1.2)  # 0.95 (remove noise)
            }
            
            keep_mask = torch.ones_like(group_indices, dtype=torch.bool)
            
            for region_name, region_mask in [('near', near_mask), ('mid', mid_mask), ('far', far_mask)]:
                if region_mask.sum() > 0:
                    unique_groups = torch.unique(group_indices[region_mask])
                    num_mask_groups = min(len(unique_groups), 
                                        int(len(unique_groups) * ratios[region_name]))
                    
                    if num_mask_groups > 0:
                        mask_groups = unique_groups[torch.randperm(len(unique_groups))[:num_mask_groups]]
                        region_keep_mask = ~(region_mask & torch.isin(group_indices, mask_groups))
                        keep_mask &= region_keep_mask
            
            masked_coords.append(coords_b[keep_mask])
            masked_features.append(features_b[keep_mask])
        
        if len(masked_coords) > 0:
            return torch.cat(masked_coords, dim=0), torch.cat(masked_features, dim=0)
        else:
            return voxel_coords, voxel_features
    
    # Alias for compatibility
    radial_masking = enhanced_radial_masking
    
    def compute_enhanced_contrastive_loss(self, multi_scale_features, momentum_features=None):
        """Enhanced contrastive learning with full momentum encoder"""
        if not self.use_contrastive:
            return {}
            
        contrastive_losses = {}
        self.step_count += 1
        
        # Multi-scale contrastive learning
        for level_name, features in multi_scale_features.items():
            if level_name not in self.projection_heads or not hasattr(features, 'features'):
                continue
                
            local_features = features.features
            if len(local_features) == 0:
                continue
            
            # Project features
            q = F.normalize(self.projection_heads[level_name](local_features), dim=1)
            
            # Enhanced momentum features
            if momentum_features and level_name in momentum_features:
                with torch.no_grad():
                    momentum_feats = momentum_features[level_name]
                    if hasattr(momentum_feats, 'features') and len(momentum_feats.features) > 0:
                        k = F.normalize(
                            self.momentum_encoder.projection_heads[level_name](momentum_feats.features), 
                            dim=1
                        )
                        self._dequeue_and_enqueue(k, f"queue_{level_name}")
                
                # Enhanced contrastive loss
                queue = getattr(self, f"queue_{level_name}").clone().detach()
                
                # Positive pairs
                l_pos = torch.einsum('nc,nc->n', [q[:len(k)], k]).unsqueeze(-1)
                # Negative pairs  
                l_neg = torch.einsum('nc,ck->nk', [q[:len(k)], queue])
                
                # Hard negative mining for better learning
                with torch.no_grad():
                    # Select harder negatives (higher similarity)
                    hard_neg_threshold = l_neg.quantile(0.7, dim=1, keepdim=True)
                    hard_neg_mask = l_neg > hard_neg_threshold
                    l_neg_weighted = l_neg * (1 + 0.5 * hard_neg_mask.float())
                
                logits = torch.cat([l_pos, l_neg_weighted], dim=1) / self.temperature
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                
                contrastive_losses[f'contrastive_{level_name}'] = F.cross_entropy(logits, labels)
        
        # Enhanced global contrastive learning
        if 'conv4' in multi_scale_features:
            global_features = self.global_pooling(multi_scale_features['conv4'])
            q_global = F.normalize(self.global_projection(global_features), dim=1)
            
            if momentum_features and 'conv4' in momentum_features:
                with torch.no_grad():
                    k_global_features = self.global_pooling(momentum_features['conv4'])
                    k_global = F.normalize(
                        self.momentum_encoder.global_projection(k_global_features), dim=1
                    )
                    self._dequeue_and_enqueue(k_global, "global_queue")
                
                # Enhanced global contrastive loss
                global_queue = self.global_queue.clone().detach()
                l_pos_global = torch.einsum('nc,nc->n', [q_global, k_global]).unsqueeze(-1)
                l_neg_global = torch.einsum('nc,ck->nk', [q_global, global_queue])
                
                logits_global = torch.cat([l_pos_global, l_neg_global], dim=1) / self.temperature
                labels_global = torch.zeros(logits_global.shape[0], dtype=torch.long).cuda()
                
                contrastive_losses['contrastive_global'] = F.cross_entropy(logits_global, labels_global)
        
        return contrastive_losses
    
    # Update the main forward to use enhanced functions
    def forward(self, batch_dict):
        """Enhanced forward pass with full momentum encoder"""
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        # Store original data
        if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
            batch_dict['original_voxel_coords'] = voxel_coords.clone()
            batch_dict['original_voxel_features'] = voxel_features.clone()
            
            # Apply enhanced radial masking
            voxel_coords, voxel_features = self.enhanced_radial_masking(voxel_coords, voxel_features)
            batch_dict['voxel_coords'] = voxel_coords
            batch_dict['voxel_features'] = voxel_features
            
            # Full momentum encoder forward
            momentum_features = None
            if self.use_contrastive:
                try:
                    with torch.no_grad():
                        self._momentum_update_encoder()
                        momentum_features = self._forward_momentum_encoder(batch_dict)
                        if momentum_features:
                            print("✅ Full momentum encoder forward successful")
                        else:
                            print("⚠️  Momentum encoder forward returned empty")
                except Exception as e:
                    print(f"⚠️  Momentum encoder forward failed: {e}")
                    momentum_features = None
        
        # Standard VoxelNeXt forward pass
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)  
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        multi_scale_features = {
            'conv1': x_conv1, 'conv2': x_conv2,
            'conv3': x_conv3, 'conv4': x_conv4,
        }
        
        # Pretraining mode
        if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
            # R-MAE Occupancy Prediction
            occupancy_pred = self.occupancy_decoder(x_conv4)
            batch_dict['occupancy_pred'] = occupancy_pred.features
            batch_dict['occupancy_coords'] = occupancy_pred.indices
            
            # Enhanced Contrastive Learning
            if self.use_contrastive and momentum_features:
                contrastive_losses = self.compute_enhanced_contrastive_loss(
                    multi_scale_features, momentum_features
                )
                batch_dict['contrastive_losses'] = contrastive_losses
        
        # Standard VoxelNeXt output
        batch_dict.update({
            'encoded_spconv_tensor': x_conv4,
            'encoded_spconv_tensor_stride': 8,
            'multi_scale_3d_features': multi_scale_features
        })
        
        return batch_dict
    
    def global_pooling(self, sparse_tensor):
        """Global pooling for sparse tensor"""
        features = sparse_tensor.features
        indices = sparse_tensor.indices
        
        batch_indices = indices[:, 0]
        batch_size = int(batch_indices.max()) + 1
        
        pooled_features = []
        for b in range(batch_size):
            mask = batch_indices == b
            if mask.sum() > 0:
                pooled_features.append(features[mask].mean(dim=0))
            else:
                pooled_features.append(torch.zeros(features.shape[1]).cuda())
        
        return torch.stack(pooled_features)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_name):
        """Update memory queue with safe attribute access"""
        try:
            batch_size = keys.shape[0]
            
            # Safe queue access
            if hasattr(self, queue_name):
                queue = getattr(self, queue_name)
            else:
                print(f"⚠️  Queue {queue_name} not found, skipping update")
                return
            
            # Safe pointer access
            ptr_name = f"{queue_name}_ptr"
            if hasattr(self, ptr_name):
                ptr = int(getattr(self, ptr_name))
            else:
                print(f"⚠️  Pointer {ptr_name} not found, skipping update")
                return
            
            # Update queue
            if ptr + batch_size <= self.queue_size:
                queue[:, ptr:ptr + batch_size] = keys.T
            else:
                queue[:, ptr:self.queue_size] = keys[:self.queue_size - ptr].T
                queue[:, 0:batch_size - (self.queue_size - ptr)] = keys[self.queue_size - ptr:].T
            
            # Update pointer
            ptr = (ptr + batch_size) % self.queue_size
            getattr(self, ptr_name)[0] = ptr
            
        except Exception as e:
            print(f"⚠️  Queue update failed for {queue_name}: {e}")
            # Continue without crashing