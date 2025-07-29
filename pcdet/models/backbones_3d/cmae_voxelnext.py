import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from functools import partial
import numpy as np
import random
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt, SparseBasicBlock, post_act_block

class CMAEVoxelNeXtComplete(VoxelResBackBone8xVoxelNeXt):
    """
    Complete CMAE-3D VoxelNeXt Backbone Implementation
    
    Full implementation of all CMAE-3D components:
    1. Geometric-Semantic Hybrid Masking (GSHM)
    2. Multi-scale Latent Feature Reconstruction (MLFR)  
    3. Hierarchical Relational Contrastive Learning (HRCL)
    4. Teacher-Student architecture with momentum updates
    """
    
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        # Store input_channels before calling super().__init__
        self.input_channels = input_channels
        
        super().__init__(model_cfg, input_channels, grid_size, **kwargs)
        
        # Required attributes
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        
        # GSHM parameters
        self.masked_ratio = model_cfg.get('MASKED_RATIO', 0.75)
        self.angular_range = model_cfg.get('ANGULAR_RANGE', 15)  # degrees
        self.distance_ranges = model_cfg.get('DISTANCE_RANGES', [0, 30, 60, 100])  # meters
        self.mask_ratios = model_cfg.get('MASK_RATIOS', [0.9, 0.8, 0.7, 0.6])  # per distance range
        
        # HRCL parameters
        self.temperature = model_cfg.get('TEMPERATURE', 0.1)
        self.momentum = model_cfg.get('MOMENTUM', 0.999)
        self.queue_size = model_cfg.get('QUEUE_SIZE', 8192)
        
        # Multi-scale feature dimensions
        self.feature_dims = {
            'conv1': 16, 'conv2': 32, 'conv3': 64, 'conv4': 128
        }
        
        print(f"🔧 Complete CMAE-3D VoxelNeXt initialized with {input_channels} input channels")
        
        # Build complete CMAE components
        if hasattr(model_cfg, 'PRETRAINING') and model_cfg.PRETRAINING:
            self._build_complete_cmae_components()
    
    def _build_complete_cmae_components(self):
        """Build complete CMAE-3D components"""
        print("🏗️ Building Complete CMAE-3D components...")
        
        # 1. Teacher Network (완전한 구조)
        self._build_complete_teacher_network()
        
        # 2. Multi-scale Latent Feature Reconstruction decoder
        self._build_complete_mlfr_decoder()
        
        # 3. Hierarchical Relational Contrastive Learning heads
        self._build_complete_hrcl_heads()
        
        # 4. Memory queues for contrastive learning
        self._register_complete_memory_queues()
        
        print("✅ Complete CMAE-3D components built successfully!")
    
    def _build_complete_teacher_network(self):
        """Build complete teacher network"""
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        teacher_input_channels = self.input_channels
        
        print(f"🏗️ Building Complete Teacher network with {teacher_input_channels} input channels")
        
        # Teacher network with exact same structure as student
        self.teacher_conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(teacher_input_channels, 16, 3, padding=1, bias=False, indice_key='teacher_subm1'),
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
        
        # Teacher initialization flag
        self._teacher_initialized = False
        
        # Disable gradients for teacher
        for param in [self.teacher_conv_input, self.teacher_conv1, 
                     self.teacher_conv2, self.teacher_conv3, self.teacher_conv4]:
            for p in param.parameters():
                p.requires_grad = False
        
        print("✅ Complete Teacher network built successfully")
    
    def _build_complete_mlfr_decoder(self):
        """Build complete Multi-scale Latent Feature Reconstruction decoder"""
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        print("🏗️ Building Complete MLFR decoder...")
        
        # Multi-scale feature reconstruction heads with proper architecture
        self.mlfr_decoder_conv4 = spconv.SparseSequential(
            spconv.SubMConv3d(128, 128, 3, padding=1, bias=False, indice_key='mlfr_dec4_1'),
            norm_fn(128), nn.ReLU(),
            spconv.SubMConv3d(128, 64, 3, padding=1, bias=False, indice_key='mlfr_dec4_2'),
            norm_fn(64), nn.ReLU(),
            spconv.SubMConv3d(64, 128, 1, bias=True, indice_key='mlfr_out4')
        )
        
        self.mlfr_decoder_conv3 = spconv.SparseSequential(
            spconv.SubMConv3d(64, 64, 3, padding=1, bias=False, indice_key='mlfr_dec3_1'),
            norm_fn(64), nn.ReLU(),
            spconv.SubMConv3d(64, 32, 3, padding=1, bias=False, indice_key='mlfr_dec3_2'),
            norm_fn(32), nn.ReLU(),
            spconv.SubMConv3d(32, 64, 1, bias=True, indice_key='mlfr_out3')
        )
        
        self.mlfr_decoder_conv2 = spconv.SparseSequential(
            spconv.SubMConv3d(32, 32, 3, padding=1, bias=False, indice_key='mlfr_dec2_1'),
            norm_fn(32), nn.ReLU(),
            spconv.SubMConv3d(32, 16, 3, padding=1, bias=False, indice_key='mlfr_dec2_2'),
            norm_fn(16), nn.ReLU(),
            spconv.SubMConv3d(16, 32, 1, bias=True, indice_key='mlfr_out2')
        )
        
        # Occupancy prediction head with deeper architecture
        self.occupancy_decoder = spconv.SparseSequential(
            spconv.SubMConv3d(128, 64, 3, padding=1, bias=False, indice_key='occ_dec1'),
            norm_fn(64), nn.ReLU(),
            spconv.SubMConv3d(64, 32, 3, padding=1, bias=False, indice_key='occ_dec2'),
            norm_fn(32), nn.ReLU(),
            spconv.SubMConv3d(32, 16, 3, padding=1, bias=False, indice_key='occ_dec3'),
            norm_fn(16), nn.ReLU(),
            spconv.SubMConv3d(16, 1, 1, bias=True, indice_key='occ_out')
        )
        
        print("✅ Complete MLFR decoder built successfully")
    
    def _build_complete_hrcl_heads(self):
        """Build complete Hierarchical Relational Contrastive Learning heads"""
        print("🏗️ Building Complete HRCL heads...")
        
        # Voxel-level contrastive projection heads (deeper)
        self.voxel_proj_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)  # Final projection
        )
        
        # Frame-level contrastive projection heads (deeper)
        self.frame_proj_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)  # Final projection
        )
        
        # Multi-scale contrastive heads for different levels
        self.conv2_proj_head = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64)
        )
        self.conv3_proj_head = nn.Sequential(
            nn.Linear(64, 96), nn.ReLU(), nn.Linear(96, 96)
        )
        
        print("✅ Complete HRCL heads built successfully")
    
    def _register_complete_memory_queues(self):
        """Register complete memory queues for contrastive learning"""
        print("🏗️ Building Complete memory queues...")
        
        # Main queues for conv4 features
        self.register_buffer("voxel_queue", torch.randn(128, self.queue_size))
        self.register_buffer("voxel_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("frame_queue", torch.randn(128, self.queue_size))
        self.register_buffer("frame_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Multi-scale queues
        self.register_buffer("conv2_queue", torch.randn(64, self.queue_size // 2))
        self.register_buffer("conv2_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("conv3_queue", torch.randn(96, self.queue_size // 2))
        self.register_buffer("conv3_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Normalize all queues
        self.voxel_queue = F.normalize(self.voxel_queue, dim=0)
        self.frame_queue = F.normalize(self.frame_queue, dim=0)
        self.conv2_queue = F.normalize(self.conv2_queue, dim=0)
        self.conv3_queue = F.normalize(self.conv3_queue, dim=0)
        
        print("✅ Complete memory queues initialized")
    
    def complete_geometric_semantic_hybrid_masking(self, voxel_coords, voxel_features):
        """
        Complete Geometric-Semantic Hybrid Masking (GSHM)
        
        Implements all three masking strategies:
        1. Geometric-aware masking: Distance-based adaptive ratios
        2. Semantic-aware masking: Feature magnitude-based importance
        3. Radial angular masking: Rotation robustness
        """
        if not self.training:
            return voxel_coords, voxel_features, None
        
        batch_size = int(voxel_coords[:, 0].max()) + 1
        masked_coords_list, masked_features_list = [], []
        mask_info_list = []
        
        for batch_idx in range(batch_size):
            batch_mask = voxel_coords[:, 0] == batch_idx
            coords = voxel_coords[batch_mask]
            features = voxel_features[batch_mask]
            
            if coords.size(0) == 0:
                continue
            
            # Convert voxel coordinates to real world coordinates
            real_coords = coords[:, 1:4].float() * torch.tensor(
                self.voxel_size, device=coords.device
            ) + torch.tensor(self.point_cloud_range[:3], device=coords.device)
            
            # 1. Geometric-aware masking based on distance
            distances = torch.norm(real_coords[:, :2], dim=1)  # XY distance from LiDAR
            
            # 2. Enhanced semantic-aware attention
            feature_magnitudes = torch.norm(features, dim=1)
            feature_std = torch.std(feature_magnitudes)
            feature_mean = torch.mean(feature_magnitudes)
            
            # Semantic importance with adaptive thresholding
            semantic_scores = torch.sigmoid(
                (feature_magnitudes - feature_mean) / (feature_std + 1e-8)
            )
            
            # 3. Height-based importance (objects vs ground)
            height_scores = torch.sigmoid(
                (real_coords[:, 2] - real_coords[:, 2].mean()) / 
                (real_coords[:, 2].std() + 1e-8)
            )
            
            # 4. Combined importance score
            importance_scores = (semantic_scores + height_scores) / 2.0
            
            # 5. Distance-based mask probability
            mask_probs = torch.ones_like(distances)
            for i, (d_min, d_max, ratio) in enumerate(zip(
                self.distance_ranges[:-1], self.distance_ranges[1:], self.mask_ratios
            )):
                distance_mask = (distances >= d_min) & (distances < d_max)
                mask_probs[distance_mask] = ratio
            
            # 6. Adjust mask probability based on importance
            # Keep more important voxels (lower mask probability)
            adjusted_mask_probs = mask_probs * (1 - 0.4 * importance_scores)
            
            # 7. Radial angular masking for rotation robustness
            angles = torch.atan2(real_coords[:, 1], real_coords[:, 0])  # Azimuth angle
            
            # Randomly select angular sectors to mask more heavily
            num_sectors = random.randint(2, 6)
            for _ in range(num_sectors):
                center_angle = random.uniform(-np.pi, np.pi)
                angular_width = np.radians(self.angular_range)
                
                # Angle difference considering circular nature
                angle_diff = torch.abs(angles - center_angle)
                angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff)
                
                angular_mask = angle_diff < angular_width / 2
                adjusted_mask_probs[angular_mask] *= 1.3  # Increase masking in selected sectors
            
            # 8. Apply adaptive masking based on voxel density
            voxel_density = coords.size(0) / (
                (real_coords[:, 0].max() - real_coords[:, 0].min() + 1) *
                (real_coords[:, 1].max() - real_coords[:, 1].min() + 1) *
                (real_coords[:, 2].max() - real_coords[:, 2].min() + 1)
            )
            density_factor = torch.clamp(voxel_density / 1000.0, 0.5, 1.5)
            adjusted_mask_probs *= density_factor
            
            # 9. Generate final mask
            random_vals = torch.rand_like(adjusted_mask_probs)
            keep_mask = random_vals > adjusted_mask_probs
            
            # 10. Ensure minimum and maximum number of voxels
            min_keep = max(int(coords.size(0) * 0.15), 10)  # At least 15% or 10 voxels
            max_keep = int(coords.size(0) * 0.4)  # At most 40%
            
            if keep_mask.sum() < min_keep:
                # Keep top importance voxels
                _, top_indices = torch.topk(importance_scores, min_keep)
                keep_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
                keep_mask[top_indices] = True
            elif keep_mask.sum() > max_keep:
                # Keep only most important voxels
                keep_indices = torch.where(keep_mask)[0]
                keep_importance = importance_scores[keep_indices]
                _, top_relative = torch.topk(keep_importance, max_keep)
                final_keep_indices = keep_indices[top_relative]
                keep_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
                keep_mask[final_keep_indices] = True
            
            # Apply mask
            masked_coords = coords[keep_mask]
            masked_features = features[keep_mask]
            
            masked_coords_list.append(masked_coords)
            masked_features_list.append(masked_features)
            
            # Store comprehensive mask information
            mask_info = {
                'original_coords': coords,
                'original_features': features,
                'keep_mask': keep_mask,
                'importance_scores': importance_scores,
                'semantic_scores': semantic_scores,
                'height_scores': height_scores,
                'distances': distances,
                'mask_ratio': 1.0 - (keep_mask.sum().float() / coords.size(0))
            }
            mask_info_list.append(mask_info)
        
        if masked_coords_list:
            final_coords = torch.cat(masked_coords_list, dim=0)
            final_features = torch.cat(masked_features_list, dim=0)
        else:
            final_coords = voxel_coords
            final_features = voxel_features
        
        return final_coords, final_features, mask_info_list
    
    @torch.no_grad()
    def complete_momentum_update_teacher(self):
        """Complete momentum update for teacher network"""
        # Student -> Teacher momentum update with proper parameter matching
        student_modules = [self.conv_input, self.conv1, self.conv2, self.conv3, self.conv4]
        teacher_modules = [self.teacher_conv_input, self.teacher_conv1, self.teacher_conv2, 
                          self.teacher_conv3, self.teacher_conv4]
        
        for student_module, teacher_module in zip(student_modules, teacher_modules):
            for s_param, t_param in zip(student_module.parameters(), teacher_module.parameters()):
                t_param.data = self.momentum * t_param.data + (1 - self.momentum) * s_param.data
    
    def _initialize_complete_teacher_from_student(self):
        """Initialize complete teacher parameters from student"""
        print("🔄 Initializing complete teacher from student...")
        
        # Copy parameters with proper structure matching
        student_modules = [self.conv_input, self.conv1, self.conv2, self.conv3, self.conv4]
        teacher_modules = [self.teacher_conv_input, self.teacher_conv1, self.teacher_conv2, 
                          self.teacher_conv3, self.teacher_conv4]
        
        for i, (student_module, teacher_module) in enumerate(zip(student_modules, teacher_modules)):
            for s_param, t_param in zip(student_module.parameters(), teacher_module.parameters()):
                t_param.data.copy_(s_param.data)
            print(f"✅ Teacher conv{i} initialized from student")
        
        print("✅ Complete teacher initialization finished")
    
    @torch.no_grad()
    def complete_teacher_forward(self, voxel_coords, voxel_features, batch_size):
        """Complete teacher forward pass"""
        # Create sparse tensor
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # Complete teacher forward pass
        x = self.teacher_conv_input(input_sp_tensor)
        x_conv1 = self.teacher_conv1(x)
        x_conv2 = self.teacher_conv2(x_conv1)
        x_conv3 = self.teacher_conv3(x_conv2)
        x_conv4 = self.teacher_conv4(x_conv3)
        
        return {
            'conv1': x_conv1, 'conv2': x_conv2,
            'conv3': x_conv3, 'conv4': x_conv4
        }
    
    def forward(self, batch_dict):
        """Complete forward pass with all CMAE-3D components"""
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        # Teacher initialization (first forward only)
        if (self.training and hasattr(self.model_cfg, 'PRETRAINING') and 
            self.model_cfg.PRETRAINING and not self._teacher_initialized):
            self._initialize_complete_teacher_from_student()
            self._teacher_initialized = True
        
        # Store original data
        batch_dict['original_voxel_coords'] = voxel_coords.clone()
        batch_dict['original_voxel_features'] = voxel_features.clone()
        
        # Pretraining mode: Apply complete GSHM and run teacher-student
        if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
            # 1. Apply complete Geometric-Semantic Hybrid Masking
            masked_coords, masked_features, mask_info = self.complete_geometric_semantic_hybrid_masking(
                voxel_coords, voxel_features
            )
            
            # 2. Complete teacher forward (complete point cloud)
            teacher_features = self.complete_teacher_forward(voxel_coords, voxel_features, batch_size)
            batch_dict['teacher_features'] = teacher_features
            
            # 3. Student forward (masked point cloud)
            student_features = self._complete_student_forward(masked_coords, masked_features, batch_size)
            batch_dict.update(student_features)
            
            # 4. Store comprehensive mask information
            batch_dict['mask_info'] = mask_info
            
            # 5. Update teacher with momentum
            self.complete_momentum_update_teacher()
            
        else:
            # Fine-tuning mode: Standard VoxelNeXt forward
            batch_dict.update(self._complete_student_forward(voxel_coords, voxel_features, batch_size))
        
        return batch_dict
    
    def _complete_student_forward(self, voxel_coords, voxel_features, batch_size):
        """Complete student network forward pass"""
        # Create sparse tensor
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # Student forward pass (standard VoxelNeXt)
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        student_features = {
            'conv1': x_conv1, 'conv2': x_conv2,
            'conv3': x_conv3, 'conv4': x_conv4
        }
        
        # Pretraining: Add complete reconstruction and contrastive features
        if self.training and hasattr(self.model_cfg, 'PRETRAINING') and self.model_cfg.PRETRAINING:
            # Complete MLFR reconstruction
            mlfr_conv4 = self.mlfr_decoder_conv4(x_conv4)
            mlfr_conv3 = self.mlfr_decoder_conv3(x_conv3) 
            mlfr_conv2 = self.mlfr_decoder_conv2(x_conv2)
            
            # Enhanced occupancy prediction
            occupancy_pred = self.occupancy_decoder(x_conv4)
            
            # Multi-scale contrastive features
            conv2_proj = None
            conv3_proj = None
            if x_conv2.features.size(0) > 0:
                conv2_proj = self.conv2_proj_head(x_conv2.features)
            if x_conv3.features.size(0) > 0:
                conv3_proj = self.conv3_proj_head(x_conv3.features)
            
            student_features.update({
                'mlfr_conv4': mlfr_conv4,
                'mlfr_conv3': mlfr_conv3, 
                'mlfr_conv2': mlfr_conv2,
                'occupancy_pred': occupancy_pred.features,
                'occupancy_coords': occupancy_pred.indices,
                'conv2_proj': conv2_proj,
                'conv3_proj': conv3_proj
            })
        
        # Standard VoxelNeXt output format
        return {
            'encoded_spconv_tensor': x_conv4,
            'encoded_spconv_tensor_stride': 8,
            'multi_scale_3d_features': {
                'x_conv1': x_conv1, 'x_conv2': x_conv2,
                'x_conv3': x_conv3, 'x_conv4': x_conv4,
            },
            'student_features': student_features
        }