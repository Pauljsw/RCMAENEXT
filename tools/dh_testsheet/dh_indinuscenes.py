import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
import numpy as np
import math
import torch
from tensorboardX import SummaryWriter
from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from geometry_msgs.msg import Point
import logging
import yaml
from pcdet.datasets.processor.data_processor import DataProcessor, VoxelGeneratorWrapper
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import matplotlib.transforms as transforms


torch.set_printoptions(precision=3, sci_mode=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config_path = "/home/duho/VoxelNeXt/models/cbgs_voxel0075_voxelnext.yaml"
cfg_from_yaml_file(config_path, cfg)
#print(cfg)
#model_path = "/home/duho/VoxelNeXt/models/voxelnext_nuscenes_double.pth"
model_path = "/home/duho/VoxelNeXt/models/voxelnext_nuscenes_kernel1.pth"

data_set, data_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,  # Batch size can be set to 1 for inference
    dist=False,  # Assuming no distributed inference for simplicity
    workers=1,  # Worker threads can be reduced if not loading batches of data
    logger=logger,
    training=False  # Indicate evaluation/inference mode
)
#print(data_set)

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=data_set)
print("model build")
model.load_params_from_file(filename=model_path, logger=logger, to_cpu=False, pre_trained_path=None)
print("model load")
model.cuda()
model.eval()


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
        tensorout = ten.to(device)
        return tensorout 
        
    point_cloud_tensor = totensor(point_cloud)

    voxels_tensor = totensor(voxels)
    coordinates_tensor = totensor(coordinates)
    batch_indices = torch.zeros(coordinates_tensor.shape[0], 1, dtype=coordinates_tensor.dtype, device=coordinates_tensor.device)
    coordinates_with_batch = torch.cat((batch_indices, coordinates_tensor), dim=1)
    num_points_tensor = totensor(num_points)
    point_cloud_batch = {
        'points': point_cloud_tensor,
        'voxels': voxels_tensor,
        'voxel_coords': coordinates_with_batch,
        'voxel_num_points': num_points_tensor,
        'batch_size': 1
    }
    return point_cloud_batch


def binary_file_to_numpy_array(file_path):
    # Define the data type for each point in the point cloud, including the 'channel'
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('channel', np.float32)]
    
    # Read the entire file into a bytes object
    with open(file_path, "rb") as f:
        file_content = f.read()

    # Convert the bytes data to a structured numpy array with the specified dtype
    points_array = np.frombuffer(file_content, dtype=dtype)

    # If points_array is structured, convert it to a 2D array
    if points_array.ndim == 1 and points_array.dtype.names is not None:  # Structured array
        # Convert to 2D array and make a copy to ensure it's writable
        points_array = points_array.view(np.float32).reshape(points_array.shape[0], -1).copy()
    points_array[:, -1] = 0
    return points_array

def rotate_point(point, angle, origin=(0, 0)):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


file_path = "/home/duho/VoxelNeXt/data/custom/points/nus2.bin"
#file_path = "/home/duho/Downloads/v1.0-01/samples/LIDAR_TOP/n008-2018-08-01-15-52-19-0400__LIDAR_TOP__1533153637197907.pcd.bin"


#file_path = "/home/duho/VoxelNeXt/data/nuscenes/v1.0-mini/samples/LIDAR_TOP/n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984253447765.pcd.bin"

point_cloud_array = binary_file_to_numpy_array(file_path)
#print(point_cloud_array)

voxel_size = [0.075, 0.075, 0.2]
cord_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0] 
num_pc_features = 5
max_num_points_per_voxel = 10
max_num_voxels = 160000
point_cloud_input = point_cloud_preprocessing(point_cloud_array, voxel_size, cord_range, num_pc_features, max_num_points_per_voxel, max_num_voxels)
#print(point_cloud_input)



with torch.no_grad():
    pred_dicts, _ = model(point_cloud_input)
    


fig, ax = plt.subplots()
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_xlabel('X')
ax.set_ylabel('Y')

X = point_cloud_array[:, 0]
Y = point_cloud_array[:, 1]
Z = point_cloud_array[:, 2]
z_min, z_max = -2, 1.0
Z[Z < z_min] = z_min  # Set any Z value lower than z_min to z_min
Z[Z > z_max] = z_max  # Set any Z value higher than z_max to z_max   
norm = Normalize(vmin=z_min, vmax=z_max)
cmap = cm.cividis_r  # '_r' suffix to reverse the colormap, light to dark for increasing Z  
sc = ax.scatter(X, Y, c=Z, cmap=cmap, norm=norm, s=1)  # 's' sets the marker size  

for pred in pred_dicts:
    pred_boxes = pred['pred_boxes']  # Assuming these are [center_x, center_y, center_z, width, length, height, rotation, velocity_x, velocity_y]
    pred_scores = pred['pred_scores']
    pred_labels = pred['pred_labels']
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score > 0.5:  
            print("box", box, "Score", score, "label", label)
            center = box[:2].cpu().numpy()
            width, length = box[3].cpu().numpy(), box[4].cpu().numpy()
            rotation = box[6].cpu().numpy()-1.6
            #rotation_deg = np.degrees(rotation)-90
            velocity = box[7:].cpu().numpy()
            corners = np.array([[-length / 2, -width / 2], [-length / 2, width / 2], [length / 2, width / 2], [length / 2, -width / 2] ])   
            rotated_corners = np.array([rotate_point(corner, rotation, origin=(0, 0)) for corner in corners])
            rotated_corners += center
            polygon = plt.Polygon(rotated_corners, edgecolor='r', facecolor='none', linewidth=1)
            ax.add_patch(polygon)
            ax.set_aspect('equal', adjustable='box')

            
            #ax.scatter(center[0], center[1], s=100, c='red', marker='o', label=f'Label {label}')  # 's' is the size of the marker
            #rect = patches.Rectangle(center - np.array([length/2, width/2]), length, width, angle=rotation_deg, linewidth=1, edgecolor='r', facecolor='none')
            #ax.add_patch(rect)
            #ax.quiver(*center, *velocity, color='r', scale=5)
            #ax.annotate(f'{label}: {score:.2f}', (center[0], center[1]), color='white', weight='bold', fontsize=8, ha='center', va='center')
            
    plt.show()








