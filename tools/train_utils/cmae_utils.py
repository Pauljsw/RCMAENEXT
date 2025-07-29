"""
CMAE-3D Training Utilities
Additional utility functions for CMAE-3D training and evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class CMAELossMonitor:
    """Monitor and log CMAE training losses"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset all loss tracking"""
        self.losses = {
            'total_loss': [],
            'mlfr_loss': [],
            'occupancy_loss': [],
            'contrastive_loss': [],
            'voxel_contrastive': [],
            'frame_contrastive': []
        }
        self.step = 0
    
    def update(self, loss_dict: Dict[str, float]):
        """Update loss tracking with new values"""
        for key, value in loss_dict.items():
            if key in self.losses:
                self.losses[key].append(value)
                # Keep only recent values
                if len(self.losses[key]) > self.window_size:
                    self.losses[key] = self.losses[key][-self.window_size:]
        
        self.step += 1
    
    def get_averages(self) -> Dict[str, float]:
        """Get average losses over the window"""
        averages = {}
        for key, values in self.losses.items():
            if values:
                averages[f'avg_{key}'] = np.mean(values[-self.window_size:])
            else:
                averages[f'avg_{key}'] = 0.0
        return averages
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest loss values"""
        latest = {}
        for key, values in self.losses.items():
            if values:
                latest[f'latest_{key}'] = values[-1]
            else:
                latest[f'latest_{key}'] = 0.0
        return latest

class CMAECheckpointManager:
    """Manage CMAE model checkpoints and state"""
    
    def __init__(self, save_dir: str, max_checkpoints: int = 10):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoint_list = []
    
    def save_checkpoint(self, model, optimizer, epoch: int, global_step: int, 
                       loss_dict: Dict[str, float], is_best: bool = False):
        """Save model checkpoint with CMAE-specific information"""
        import os
        from pathlib import Path
        
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint state
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        checkpoint_state = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state': model_state,
            'optimizer_state': optimizer.state_dict(),
            'loss_dict': loss_dict,
            'is_pretraining': True,  # Mark as pretraining checkpoint
            'model_type': 'CMAE-3D'
        }
        
        # Regular checkpoint
        ckpt_name = save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint_state, ckpt_name)
        self.checkpoint_list.append(str(ckpt_name))
        
        # Best checkpoint
        if is_best:
            best_ckpt_name = save_dir / 'checkpoint_best.pth'
            torch.save(checkpoint_state, best_ckpt_name)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints"""
        if len(self.checkpoint_list) > self.max_checkpoints:
            import os
            for old_ckpt in self.checkpoint_list[:-self.max_checkpoints]:
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
            self.checkpoint_list = self.checkpoint_list[-self.max_checkpoints:]

class CMAEVisualization:
    """Visualization utilities for CMAE training"""
    
    @staticmethod
    def visualize_masking(original_coords: torch.Tensor, 
                         masked_coords: torch.Tensor,
                         save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Visualize masking pattern"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(15, 5))
            
            # Original point cloud
            ax1 = fig.add_subplot(131, projection='3d')
            orig_np = original_coords.cpu().numpy()
            ax1.scatter(orig_np[:, 1], orig_np[:, 2], orig_np[:, 3], 
                       c='blue', s=1, alpha=0.6)
            ax1.set_title('Original Point Cloud')
            ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
            
            # Masked point cloud
            ax2 = fig.add_subplot(132, projection='3d')
            masked_np = masked_coords.cpu().numpy()
            ax2.scatter(masked_np[:, 1], masked_np[:, 2], masked_np[:, 3], 
                       c='red', s=1, alpha=0.6)
            ax2.set_title('Masked Point Cloud')
            ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
            
            # Bird's eye view
            ax3 = fig.add_subplot(133)
            ax3.scatter(orig_np[:, 1], orig_np[:, 2], c='blue', s=1, alpha=0.3, label='Original')
            ax3.scatter(masked_np[:, 1], masked_np[:, 2], c='red', s=1, alpha=0.7, label='Kept')
            ax3.set_title('Bird\'s Eye View')
            ax3.set_xlabel('X'); ax3.set_ylabel('Y')
            ax3.legend()
            ax3.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                return None
            else:
                # Return as numpy array
                fig.canvas.draw()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close()
                return buf
        except ImportError:
            print("Matplotlib not available for visualization")
            return None
    
    @staticmethod
    def plot_loss_curves(loss_monitor: CMAELossMonitor, 
                        save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Plot training loss curves"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Total loss
            if loss_monitor.losses['total_loss']:
                axes[0, 0].plot(loss_monitor.losses['total_loss'])
                axes[0, 0].set_title('Total Loss')
                axes[0, 0].grid(True)
            
            # MLFR loss
            if loss_monitor.losses['mlfr_loss']:
                axes[0, 1].plot(loss_monitor.losses['mlfr_loss'])
                axes[0, 1].set_title('MLFR Loss')
                axes[0, 1].grid(True)
            
            # Occupancy loss
            if loss_monitor.losses['occupancy_loss']:
                axes[1, 0].plot(loss_monitor.losses['occupancy_loss'])
                axes[1, 0].set_title('Occupancy Loss')
                axes[1, 0].grid(True)
            
            # Contrastive loss
            if loss_monitor.losses['contrastive_loss']:
                axes[1, 1].plot(loss_monitor.losses['contrastive_loss'])
                axes[1, 1].set_title('Contrastive Loss')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                return None
            else:
                fig.canvas.draw()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close()
                return buf
        except ImportError:
            print("Matplotlib not available for plotting")
            return None

class CMAEEvaluator:
    """Evaluation utilities for CMAE pretraining"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset evaluation metrics"""
        self.metrics = {
            'reconstruction_accuracy': [],
            'contrastive_alignment': [],
            'feature_diversity': []
        }
    
    def evaluate_reconstruction(self, pred_occupancy: torch.Tensor, 
                              gt_occupancy: torch.Tensor) -> float:
        """Evaluate occupancy reconstruction accuracy"""
        with torch.no_grad():
            pred_binary = torch.sigmoid(pred_occupancy) > 0.5
            accuracy = (pred_binary == gt_occupancy).float().mean().item()
            self.metrics['reconstruction_accuracy'].append(accuracy)
            return accuracy
    
    def evaluate_contrastive_alignment(self, student_features: torch.Tensor, 
                                     teacher_features: torch.Tensor) -> float:
        """Evaluate student-teacher feature alignment"""
        with torch.no_grad():
            # Normalize features
            student_norm = F.normalize(student_features, dim=-1)
            teacher_norm = F.normalize(teacher_features, dim=-1)
            
            # Compute cosine similarity
            alignment = torch.mean(torch.sum(student_norm * teacher_norm, dim=-1)).item()
            self.metrics['contrastive_alignment'].append(alignment)
            return alignment
    
    def evaluate_feature_diversity(self, features: torch.Tensor) -> float:
        """Evaluate feature diversity (avoid feature collapse)"""
        with torch.no_grad():
            # Compute feature correlation matrix
            features_norm = F.normalize(features, dim=-1)
            corr_matrix = torch.mm(features_norm.T, features_norm) / features.size(0)
            
            # Diversity = 1 - mean absolute off-diagonal correlation
            mask = ~torch.eye(corr_matrix.size(0), dtype=torch.bool, device=corr_matrix.device)
            diversity = 1.0 - torch.mean(torch.abs(corr_matrix[mask])).item()
            self.metrics['feature_diversity'].append(diversity)
            return diversity
    
    def get_summary(self) -> Dict[str, float]:
        """Get evaluation summary"""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f'avg_{key}'] = np.mean(values)
                summary[f'std_{key}'] = np.std(values)
            else:
                summary[f'avg_{key}'] = 0.0
                summary[f'std_{key}'] = 0.0
        return summary

def create_cmae_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Create optimizer for CMAE training with special handling"""
    
    # Separate parameters for different components
    backbone_params = []
    teacher_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'teacher' in name:
            teacher_params.append(param)
        elif 'decoder' in name or 'proj_head' in name:
            decoder_params.append(param)
        else:
            backbone_params.append(param)
    
    # Teacher parameters don't need gradients
    for param in teacher_params:
        param.requires_grad = False
    
    # Different learning rates for different components
    param_groups = [
        {'params': backbone_params, 'lr': config.LR, 'weight_decay': config.WEIGHT_DECAY},
        {'params': decoder_params, 'lr': config.LR * 2, 'weight_decay': config.WEIGHT_DECAY * 0.5}
    ]
    
    if config.OPTIMIZER == 'adam_onecycle':
        optimizer = torch.optim.AdamW(param_groups, lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(param_groups, lr=config.LR, weight_decay=config.WEIGHT_DECAY, 
                                   momentum=config.MOMENTUM)
    else:
        optimizer = torch.optim.Adam(param_groups, lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    
    return optimizer

def load_cmae_checkpoint_for_finetune(model: nn.Module, checkpoint_path: str, 
                                     logger=None) -> Tuple[int, int]:
    """Load CMAE pretraining checkpoint for fine-tuning"""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if logger:
        logger.info(f"Loading CMAE checkpoint from {checkpoint_path}")
        logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Checkpoint type: {checkpoint.get('model_type', 'unknown')}")
    
    # Load only backbone weights (exclude decoder and teacher)
    model_state_dict = model.state_dict()
    pretrained_dict = checkpoint['model_state']
    
    # Filter out decoder and teacher weights
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if ('teacher' not in k and 'decoder' not in k and 
            'proj_head' not in k and 'queue' not in k):
            if k in model_state_dict:
                filtered_dict[k] = v
            else:
                if logger:
                    logger.warning(f"Key {k} not found in model")
    
    # Update model
    model_state_dict.update(filtered_dict)
    model.load_state_dict(model_state_dict)
    
    if logger:
        logger.info(f"Loaded {len(filtered_dict)} parameters from pretrained model")
    
    return checkpoint.get('epoch', 0), checkpoint.get('global_step', 0)
