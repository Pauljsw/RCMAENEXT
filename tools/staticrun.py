import os
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as patches
from pathlib import Path
import logging

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
config_path = "/home/duho/VoxelNeXt/models/0225_Syntehtic_Mocktest/voxelnext_synthetic3chan.yaml"
model_path = "/home/duho/VoxelNeXt/models/0225_Syntehtic_Mocktest/mocktest_3chan.pth"
input_folder = "/home/duho/VoxelNeXt/data/custom/points"
output_folder = "/home/duho/VoxelNeXt/data/custom/points_pcd"

# Load configuration
cfg_from_yaml_file(config_path, cfg)

# Build dataloader and dataset
data_set, data_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,
    dist=False,
    workers=1,
    logger=logger,
    training=False
)

# Build model
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=data_set)
model.load_params_from_file(filename=model_path, logger=logger, to_cpu=False, pre_trained_path=None)
model.cuda()
model.eval()

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def point_cloud_preprocessing(point_cloud, voxel_size, cord_range, num_pc_features, max_num_points_per_voxel, max_num_voxels):
    voxel_generator = VoxelGeneratorWrapper(
        vsize_xyz=voxel_size,
        coors_range_xyz=cord_range,
        num_point_features=num_pc_features,
        max_num_points_per_voxel=max_num_points_per_voxel,
        max_num_voxels=max_num_voxels
    )
    voxels, coordinates, num_points = voxel_generator.generate(point_cloud)
    
    def totensor(npinput):
        ten = torch.from_numpy(npinput)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ten.to(device)
    
    point_cloud_tensor = totensor(point_cloud)
    voxels_tensor = totensor(voxels)
    coordinates_tensor = totensor(coordinates)
    batch_indices = torch.zeros(coordinates_tensor.shape[0], 1, dtype=coordinates_tensor.dtype, device=coordinates_tensor.device)
    coordinates_with_batch = torch.cat((batch_indices, coordinates_tensor), dim=1)
    num_points_tensor = totensor(num_points)
    
    return {
        'points': point_cloud_tensor,
        'voxels': voxels_tensor,
        'voxel_coords': coordinates_with_batch,
        'voxel_num_points': num_points_tensor,
        'batch_size': 1
    }

def process_and_save_pcd(npy_file):
    pointcloud = np.load(npy_file)
    voxel_size = [0.1, 0.1, 0.2]
    cord_range = [-70.4, -70.4, -6, 70.4, 70.4, 10]
    num_pc_features = 3
    max_num_points_per_voxel = 5
    max_num_voxels = 150000
    
    with torch.no_grad():
        point_cloud_input = point_cloud_preprocessing(pointcloud, voxel_size, cord_range, num_pc_features, max_num_points_per_voxel, max_num_voxels)
        pred_dicts, _ = model(point_cloud_input)
        print(pred_dicts)

    # Convert pointcloud to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])  # XYZ points

    # Save to PCD
    output_filename = os.path.join(output_folder, Path(npy_file).stem + ".pcd")
    o3d.io.write_point_cloud(output_filename, pcd)
    print(f"Saved: {output_filename}")

# Process all .npy files
for npy_file in sorted(Path(input_folder).glob("*.npy")):
    process_and_save_pcd(str(npy_file))
