import torch
import torch.nn as nn
import numpy as np
import random
from functools import partial

from ...utils.spconv_utils import spconv, replace_feature
from ..detectors.detector3d_template import Detector3DTemplate
from ..backbones_3d.spconv_backbone import post_act_block, SparseBasicBlock


class RMAEVoxelNeXt(Detector3DTemplate):
    """
    R-MAE (Radially Masked Autoencoding) implementation for VoxelNeXt
    Based on official GitHub implementation
    """
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        
        # R-MAE specific parameters
        self.masked_ratio = model_cfg.get('MASKED_RATIO', 0.8)
        self.angular_range = model_cfg.get('ANGULAR_RANGE', 5)  # degrees
        self.range_aware = model_cfg.get('RANGE_AWARE', True)  # Enable range-aware masking
        
        # For pre-training phase
        self.is_pretraining = model_cfg.get('IS_PRETRAINING', True)
        
        if self.is_pretraining:
            self.module_topology = ['vfe', 'rmae_backbone', 'mae_decoder']
        else:
            self.module_topology = ['vfe', 'backbone_3d', 'dense_head']
            
        self.module_list = self.build_networks()
        
    def build_rmae_backbone(self, model_info_dict):
        """Build R-MAE backbone for pre-training"""
        if self.model_cfg.get('RMAE_BACKBONE', None) is None:
            return None, model_info_dict
            
        rmae_backbone = RMAEBackbone(
            model_cfg=self.model_cfg.RMAE_BACKBONE,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            masked_ratio=self.masked_ratio,
            angular_range=self.angular_range,
            range_aware=self.range_aware
        )
        
        model_info_dict['module_list'].append(rmae_backbone)
        model_info_dict['num_point_features'] = rmae_backbone.num_point_features
        return rmae_backbone, model_info_dict
    
    def build_mae_decoder(self, model_info_dict):
        """Build MAE decoder for occupancy reconstruction"""
        if self.model_cfg.get('MAE_DECODER', None) is None:
            return None, model_info_dict
            
        mae_decoder = MAEDecoder(
            model_cfg=self.model_cfg.MAE_DECODER,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size']
        )
        
        model_info_dict['module_list'].append(mae_decoder)
        return mae_decoder, model_info_dict
    
    def forward(self, batch_dict):
        """Forward pass"""
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if self.is_pretraining:
                # For pre-training mode during eval, just return batch_dict
                return batch_dict
            else:
                # For downstream mode during eval, do post-processing
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts
    
    def get_training_loss(self):
        """Get training loss"""
        disp_dict = {}
        
        if self.is_pretraining:
            # MAE reconstruction loss
            loss, tb_dict = self.mae_decoder.get_loss()
        else:
            # Standard detection loss
            loss, tb_dict = self.dense_head.get_loss()
        
        return loss, tb_dict, disp_dict
    
    def post_processing(self, batch_dict):
        """Post-processing for detection (only used in downstream mode)"""
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']
            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict


class RMAEBackbone(nn.Module):
    """
    R-MAE Backbone - based on official GitHub implementation
    """
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, 
                 masked_ratio=0.8, angular_range=5, range_aware=True, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.masked_ratio = masked_ratio
        self.angular_range = angular_range
        self.range_aware = range_aware
        
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        # Encoder layers (VoxelNeXt 표준 구조를 정확히 따름)
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        
        block = post_act_block
        
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21] (VoxelNeXt 표준 설정)
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )
        
        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11] (VoxelNeXt 표준 설정)
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        
        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5] (VoxelNeXt 표준 설정 - Z축 padding=0이 핵심!)
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        
        # VoxelNeXt 표준 conv_out - 여기가 핵심!
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2] (Z축 stride=2로 더 압축)
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        
        self.num_point_features = 128
        self.forward_ret_dict = {}
        
    def radial_range_aware_masking(self, voxel_coords):
        """
        Radial range-aware masking - based on official GitHub implementation
        """
        select_ratio = 1 - self.masked_ratio  # Ratio for selecting voxels
        
        # Calculate angles and distances from voxel coordinates
        # Convert voxel indices to real-world coordinates
        real_x = (voxel_coords[:, 3].float() * self.voxel_size[0]) + self.point_cloud_range[0]
        real_y = (voxel_coords[:, 2].float() * self.voxel_size[1]) + self.point_cloud_range[1]
        
        angles = torch.atan2(real_y, real_x)  # atan2(y, x)
        angles_deg = torch.rad2deg(angles) % 360  # Convert to degrees
        voxel_coords_distance = torch.sqrt(real_x**2 + real_y**2)  # Euclidean distance
        
        # Group indices based on angles
        radial_groups = {}
        for angle in range(0, 360, self.angular_range):
            mask = (angles_deg >= angle) & (angles_deg < angle + self.angular_range)
            group_indices = torch.where(mask)[0]
            if len(group_indices) > 0:
                radial_groups[angle] = group_indices
        
        if len(radial_groups) == 0:
            # Fallback: return all indices if no groups found
            return torch.arange(len(voxel_coords), dtype=torch.long, device=voxel_coords.device)
        
        # Randomly select a portion of radial groups
        num_groups_to_select = max(1, int(select_ratio * len(radial_groups)))
        selected_group_angles = random.sample(list(radial_groups.keys()), num_groups_to_select)
        
        # Apply range-aware masking within selected radial groups
        selected_indices = []
        for angle in selected_group_angles:
            group_indices = radial_groups[angle]
            
            if self.range_aware:
                # Subdivide group indices based on distance ranges
                select_30 = voxel_coords_distance[group_indices] <= 30
                select_30to50 = (voxel_coords_distance[group_indices] > 30) & (voxel_coords_distance[group_indices] <= 50)
                select_50 = voxel_coords_distance[group_indices] > 50
                
                # Get indices for each distance range
                id_list_select_30 = group_indices[select_30].tolist()
                id_list_select_30to50 = group_indices[select_30to50].tolist()
                id_list_select_50 = group_indices[select_50].tolist()
                
                # Shuffle and select indices based on distance ranges
                random.shuffle(id_list_select_30)
                random.shuffle(id_list_select_30to50)
                random.shuffle(id_list_select_50)
                
                # Different selection ratios for different ranges (closer = higher probability)
                num_30 = max(1, int(select_ratio * len(id_list_select_30)))
                num_30to50 = max(1, int(min(1.0, select_ratio + 0.2) * len(id_list_select_30to50)))
                num_50 = max(1, int(min(1.0, select_ratio + 0.4) * len(id_list_select_50)))
                
                selected_indices.extend(id_list_select_30[:num_30])
                selected_indices.extend(id_list_select_30to50[:num_30to50])
                selected_indices.extend(id_list_select_50[:num_50])
            else:
                # Simple random selection within the group
                group_indices_list = group_indices.tolist()
                random.shuffle(group_indices_list)
                num_select = max(1, int(select_ratio * len(group_indices_list)))
                selected_indices.extend(group_indices_list[:num_select])
        
        # Convert list to tensor and return
        if len(selected_indices) == 0:
            # Fallback: return at least some indices
            num_fallback = max(100, len(voxel_coords) // 10)
            selected_indices = random.sample(range(len(voxel_coords)), min(num_fallback, len(voxel_coords)))
            
        return torch.tensor(selected_indices, dtype=torch.long, device=voxel_coords.device)
    
    def forward(self, batch_dict):
        """
        Forward pass with radial masking
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        # Apply radial masking during training
        if self.training:
            selected_indices = self.radial_range_aware_masking(voxel_coords)
            
            # Keep only selected (unmasked) voxels
            voxel_features = voxel_features[selected_indices]
            voxel_coords = voxel_coords[selected_indices]
            
            # Store original data for loss computation
            batch_dict['original_voxel_coords'] = batch_dict['voxel_coords']
            batch_dict['original_voxel_features'] = batch_dict['voxel_features']
            batch_dict['selected_indices'] = selected_indices
            batch_dict['mask_ratio'] = 1.0 - (len(selected_indices) / len(batch_dict['voxel_coords']))
        
        # Create sparse tensor with (possibly masked) voxels
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # Forward through encoder
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        
        batch_dict['encoded_spconv_tensor'] = out
        batch_dict['encoded_spconv_tensor_stride'] = 8
        
        # Store multi-scale features
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        
        return batch_dict


class MAEDecoder(nn.Module):
    """
    MAE Decoder for occupancy reconstruction - based on official implementation
    """
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        
        # Decoder architecture - simplified version using dense convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(input_channels, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, 3, stride=1, padding=1, bias=False),
            nn.Sigmoid()  # Output occupancy probabilities
        )
        
        self.criterion = nn.BCELoss()
        self.forward_ret_dict = {}
    
    def forward(self, batch_dict):
        """
        Forward pass for occupancy reconstruction
        """
        if 'encoded_spconv_tensor' not in batch_dict:
            return batch_dict
            
        # Get encoded features
        encoded_tensor = batch_dict['encoded_spconv_tensor']
        
        # Convert sparse tensor to dense for decoder
        batch_size = encoded_tensor.batch_size
        spatial_shape = encoded_tensor.spatial_shape
        
        # Create dense tensor
        dense_features = torch.zeros(
            batch_size, encoded_tensor.features.shape[1], 
            spatial_shape[0], spatial_shape[1], spatial_shape[2],
            device=encoded_tensor.features.device,
            dtype=encoded_tensor.features.dtype
        )
        
        # Fill dense tensor with sparse features
        for i in range(batch_size):
            batch_mask = encoded_tensor.indices[:, 0] == i
            if batch_mask.any():
                coords = encoded_tensor.indices[batch_mask, 1:4]
                features = encoded_tensor.features[batch_mask]
                dense_features[i, :, coords[:, 0], coords[:, 1], coords[:, 2]] = features.T
        
        # Decode to occupancy
        occupancy_pred = self.decoder(dense_features)
        
        # Create target occupancy (all original voxels are occupied)
        if self.training and 'original_voxel_coords' in batch_dict:
            original_coords = batch_dict['original_voxel_coords']
            target_occupancy = torch.zeros_like(occupancy_pred)
            
            for i in range(batch_size):
                batch_mask = original_coords[:, 0] == i
                if batch_mask.any():
                    coords = original_coords[batch_mask, 1:4]
                    # Ensure coordinates are within bounds
                    valid_mask = (
                        (coords[:, 0] >= 0) & (coords[:, 0] < target_occupancy.shape[2]) &
                        (coords[:, 1] >= 0) & (coords[:, 1] < target_occupancy.shape[3]) &
                        (coords[:, 2] >= 0) & (coords[:, 2] < target_occupancy.shape[4])
                    )
                    valid_coords = coords[valid_mask]
                    if len(valid_coords) > 0:
                        # Mark original voxel positions as occupied
                        target_occupancy[i, 0, valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = 1.0
            
            # Compute reconstruction loss
            loss = self.criterion(occupancy_pred, target_occupancy)
            batch_dict['mae_loss'] = loss
            
            self.forward_ret_dict = {
                'pred': occupancy_pred,
                'target': target_occupancy,
                'loss': loss
            }
        
        return batch_dict
    
    def get_loss(self, tb_dict=None):
        """Get MAE reconstruction loss"""
        tb_dict = {} if tb_dict is None else tb_dict
        
        if 'loss' in self.forward_ret_dict:
            loss = self.forward_ret_dict['loss']
            tb_dict['loss_mae'] = loss.item()
            return loss, tb_dict
        else:
            # Fallback loss if forward_ret_dict is empty
            dummy_loss = torch.tensor(0.0, requires_grad=True)
            tb_dict['loss_mae'] = 0.0
            return dummy_loss, tb_dict


def build_rmae_voxelnext(model_cfg, num_class, dataset):
    """Builder function for R-MAE VoxelNeXt"""
    model = RMAEVoxelNeXt(
        model_cfg=model_cfg,
        num_class=num_class, 
        dataset=dataset
    )
    return model