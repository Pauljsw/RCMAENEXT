import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from functools import partial
import numpy as np
import random
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt

class RMAEContrastiveVoxelNeXt(VoxelResBackBone8xVoxelNeXt):
    """
    R-MAE + Contrastive Learning VoxelNeXt Backbone
    
    Combines:
    - R-MAE: Radial geometric masking & reconstruction
    - Contrastive Learning: Multi-scale discriminative learning
    - VoxelNeXt: Efficient sparse convolution backbone
    """
    
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg, input_channels, grid_size, **kwargs)
        
        # Required attributes
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.model_cfg = model_cfg
        
        # R-MAE parameters
        self.masked_ratio = model_cfg.get('MASKED_RATIO', 0.8)
        self.angular_range = model_cfg.get('ANGULAR_RANGE', 10)  # degrees
        
        # Contrastive learning parameters
        self.temperature = model_cfg.get('TEMPERATURE', 0.1)
        self.momentum = model_cfg.get('MOMENTUM', 0.999)
        self.queue_size = model_cfg.get('QUEUE_SIZE', 4096)
        self.use_contrastive = model_cfg.get('USE_CONTRASTIVE', True)
        
        # Multi-scale feature dimensions (from VoxelNeXt)
        self.feature_dims = {
            'conv1': 16, 'conv2': 32, 'conv3': 64, 'conv4': 128
        }
        
        # Build pretraining components
        if hasattr(model_cfg, 'PRETRAINING') and model_cfg.PRETRAINING:
            self._build_pretraining_components()
    
    def _build_pretraining_components(self):
        """Build R-MAE + Contrastive components"""
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
            # 2. Multi-scale Projection Heads
            try:
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
                
                # 4. Momentum encoder (더 단순하게)
                print("Building momentum encoder...")
                self.momentum_encoder = self._build_simple_momentum_encoder()
                
                # 5. Memory queues for each level
                for level in self.feature_dims.keys():
                    self.register_buffer(f"queue_{level}", torch.randn(128, self.queue_size))
                    self.register_buffer(f"queue_ptr_{level}", torch.zeros(1, dtype=torch.long))
                    queue = getattr(self, f"queue_{level}")
                    queue = F.normalize(queue, dim=0)
                    setattr(self, f"queue_{level}", queue)
                
                # Global queue
                self.register_buffer("global_queue", torch.randn(128, self.queue_size))
                self.register_buffer("global_queue_ptr", torch.zeros(1, dtype=torch.long))
                self.global_queue = F.normalize(self.global_queue, dim=0)
                
                print("✅ Contrastive learning components successfully initialized")
                
            except Exception as e:
                print(f"❌ Failed to initialize contrastive learning: {e}")
                print("🔄 Falling back to R-MAE only mode")
                self.use_contrastive = False
        else:
            print("ℹ️  Contrastive learning disabled - R-MAE only mode")
        
        # 6. Training step counter
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))
    
    def _build_simple_momentum_encoder(self):
        """Build simplified momentum encoder for contrastive learning"""
        try:
            # 단순한 복사본 생성 (projection head만 추가)
            momentum_encoder = nn.Module()
            
            # Main encoder의 구조를 직접 복사하지 말고, 필요한 부분만 구현
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
            
            # 모든 파라미터를 requires_grad=False로 설정
            for param in momentum_encoder.parameters():
                param.requires_grad = False
                
            return momentum_encoder
            
        except Exception as e:
            print(f"Warning: Simple momentum encoder creation failed: {e}")
            return nn.Module()  # 더미 모듈 반환
        """Build momentum encoder with same architecture"""
        # 부모 클래스에서 정확한 속성명 사용
        input_channels = getattr(self, 'num_bev_features', 
                                getattr(self, 'input_channels', 
                                       getattr(self, 'num_point_features', 3)))
        
        momentum_encoder = VoxelResBackBone8xVoxelNeXt(
            self.model_cfg, input_channels, self.sparse_shape
        )
        
    def _build_momentum_encoder(self):
        """Build momentum encoder with same architecture"""
    def _build_momentum_encoder(self):
        """Build momentum encoder with identical structure"""
        try:
            # 현재 main encoder와 정확히 동일한 구조로 생성
            # self는 이미 VoxelResBackBone8xVoxelNeXt를 상속받았으므로
            # 동일한 클래스로 momentum encoder 생성
            
            momentum_encoder = self.__class__(
                model_cfg=self.model_cfg,
                input_channels=self.input_channels if hasattr(self, 'input_channels') else 3,
                grid_size=self.grid_size if hasattr(self, 'grid_size') else self.sparse_shape,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            
            # Pretraining components는 생성하지 않도록 임시 설정
            momentum_encoder.use_contrastive = False
            
            # Projection heads만 추가
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
            
            return momentum_encoder
            
        except Exception as e:
            print(f"Failed to create momentum encoder: {e}")
            # 더 간단한 방법으로 시도
            return self._build_simple_momentum_encoder()
    
    def _build_simple_momentum_encoder(self):
        """Build simplified momentum encoder using only necessary parts"""
        # 완전히 독립적인 momentum encoder
        class SimpleMomentumEncoder(nn.Module):
            def __init__(self, feature_dims):
                super().__init__()
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
        
        return SimpleMomentumEncoder(self.feature_dims)
    
    def _init_momentum_encoder(self):
        """Initialize momentum encoder"""
        for param_q, param_k in zip(self.parameters(), self.momentum_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update_encoder(self):
        """Momentum update with safe parameter matching"""
        try:
            main_params = dict(self.named_parameters())
            momentum_params = dict(self.momentum_encoder.named_parameters())
            
            for name, param_k in momentum_params.items():
                if name in main_params:
                    param_q = main_params[name]
                    # 크기가 동일한 경우만 업데이트
                    if param_q.shape == param_k.shape:
                        param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
                        
        except Exception as e:
            print(f"Warning: Momentum update failed: {e}")
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_name):
        """Update memory queue"""
        batch_size = keys.shape[0]
        queue = getattr(self, queue_name)
        ptr = int(getattr(self, f"{queue_name}_ptr"))
        
        # Replace keys at ptr
        if ptr + batch_size <= self.queue_size:
            queue[:, ptr:ptr + batch_size] = keys.T
        else:
            queue[:, ptr:self.queue_size] = keys[:self.queue_size - ptr].T
            queue[:, 0:batch_size - (self.queue_size - ptr)] = keys[self.queue_size - ptr:].T
        
        ptr = (ptr + batch_size) % self.queue_size
        getattr(self, f"{queue_name}_ptr")[0] = ptr
    
    def radial_masking(self, voxel_coords, voxel_features):
        """Enhanced radial masking strategy"""
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
            
            # Distance-aware masking
            distance_quantiles = torch.quantile(distances, torch.tensor([0.3, 0.7]).to(distances.device))
            near_mask = distances < distance_quantiles[0]
            mid_mask = (distances >= distance_quantiles[0]) & (distances < distance_quantiles[1])
            far_mask = distances >= distance_quantiles[1]
            
            # Different masking ratios for different distance ranges
            # 모든 비율이 1.0을 넘지 않도록 보장
            ratios = {
                'near': self.masked_ratio * 0.6,        # 0.8 * 0.6 = 0.48
                'mid': self.masked_ratio,                # 0.8
                'far': min(0.95, self.masked_ratio * 1.2)  # min(0.95, 0.8 * 1.2) = 0.95
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
    
    def compute_contrastive_loss(self, multi_scale_features, momentum_features=None):
        """Multi-scale contrastive learning"""
        if not self.use_contrastive:
            return {}
            
        contrastive_losses = {}
        self.step_count += 1
        
        # Local contrastive learning at each scale
        for level_name, features in multi_scale_features.items():
            if level_name not in self.projection_heads or not hasattr(features, 'features'):
                continue
                
            local_features = features.features
            if len(local_features) == 0:
                continue
            
            # Project features
            q = F.normalize(self.projection_heads[level_name](local_features), dim=1)
            
            # Momentum features
            if momentum_features and level_name in momentum_features:
                with torch.no_grad():
                    momentum_feats = momentum_features[level_name]
                    if hasattr(momentum_feats, 'features') and len(momentum_feats.features) > 0:
                        k = F.normalize(
                            self.momentum_encoder.projection_heads[level_name](momentum_feats.features), 
                            dim=1
                        )
                        self._dequeue_and_enqueue(k, f"queue_{level_name}")
                
                # Contrastive loss computation
                queue = getattr(self, f"queue_{level_name}").clone().detach()
                
                # Positive pairs
                l_pos = torch.einsum('nc,nc->n', [q[:len(k)], k]).unsqueeze(-1)
                # Negative pairs
                l_neg = torch.einsum('nc,ck->nk', [q[:len(k)], queue])
                
                logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                
                contrastive_losses[f'contrastive_{level_name}'] = F.cross_entropy(logits, labels)
        
        # Global contrastive learning
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
                
                # Global contrastive loss
                global_queue = self.global_queue.clone().detach()
                l_pos_global = torch.einsum('nc,nc->n', [q_global, k_global]).unsqueeze(-1)
                l_neg_global = torch.einsum('nc,ck->nk', [q_global, global_queue])
                
                logits_global = torch.cat([l_pos_global, l_neg_global], dim=1) / self.temperature
                labels_global = torch.zeros(logits_global.shape[0], dtype=torch.long).cuda()
                
                contrastive_losses['contrastive_global'] = F.cross_entropy(logits_global, labels_global)
        
        return contrastive_losses
    
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
    
    def forward(self, batch_dict):
        """Enhanced forward pass"""
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        # Store original data for loss computation
        if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
            batch_dict['original_voxel_coords'] = voxel_coords.clone()
            batch_dict['original_voxel_features'] = voxel_features.clone()
            
            # Apply radial masking
            voxel_coords, voxel_features = self.radial_masking(voxel_coords, voxel_features)
            batch_dict['voxel_coords'] = voxel_coords
            batch_dict['voxel_features'] = voxel_features
            
            # Momentum encoder forward (for contrastive learning)
            momentum_features = None
            if self.use_contrastive:
                try:
                    with torch.no_grad():
                        self._momentum_update_encoder()
                        momentum_features = self._forward_momentum_encoder(batch_dict)
                except Exception as e:
                    print(f"Warning: Momentum encoder forward failed: {e}")
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
            
            # Contrastive Learning
            if self.use_contrastive and 'momentum_features' in locals():
                contrastive_losses = self.compute_contrastive_loss(
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
    
    def _forward_momentum_encoder(self, batch_dict):
        """Forward pass through momentum encoder"""
        try:
            # Create different augmented version
            aug_coords, aug_features = self.radial_masking(
                batch_dict['original_voxel_coords'], 
                batch_dict['original_voxel_features']
            )
            
            # Check if momentum encoder has full VoxelNeXt structure
            if hasattr(self.momentum_encoder, 'conv_input'):
                # Full momentum encoder (VoxelNeXt structure)
                input_sp_tensor = spconv.SparseConvTensor(
                    features=aug_features,
                    indices=aug_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_dict['batch_size']
                )
                
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
                # Simple momentum encoder - use main encoder features
                # 이 경우 main encoder의 forward로 feature 생성 후 momentum projection 적용
                print("Using simplified momentum encoding strategy")
                return {}
                
        except Exception as e:
            print(f"Warning: Momentum encoder forward failed: {e}")
            return {}