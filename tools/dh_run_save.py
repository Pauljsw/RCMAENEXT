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

import rospy
import sensor_msgs.point_cloud2 as pc2
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import PointCloud2
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
import open3d as o3d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config_path = "/home/duho/VoxelNeXt/models/0225_Syntehtic_Mocktest/voxelnext_0206s.yaml"
cfg_from_yaml_file(config_path, cfg)
#print(cfg)
model_path = "/home/duho/VoxelNeXt/models/0225_Syntehtic_Mocktest/mocktest1.pth"

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

#Global variables
pointcloud = None
predictions = None

label_colors = ['r', 'g', 'b', 'c', 'm', 'y']  # You can choose other colors if you prefer


def pointcloud2_to_array(msg):
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
    generator = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    points_array = np.array(list(generator), dtype=dtype)
    # If points_array is structured, convert it to a 2D array
    if points_array.ndim == 1 and points_array.dtype.names is not None:  # Structured array
        points_array = points_array.view(np.float32).reshape(points_array.shape[0], -1)
    #print("Shape: ", points_array.shape)
    return points_array

def pc_cb(msg):
    global pointcloud, predictions
    array = pointcloud2_to_array(msg)
    voxel_size = [0.1, 0.1, 0.2]
    #cord_range = [-49.6, -49.6, -1.0, 49.6, 49.6, 7.0] 
    cord_range = [-70.4, -70.4, -6, 70.4, 70.4, 10]
    num_pc_features = 4
    max_num_points_per_voxel = 5
    max_num_voxels = 150000
    point_cloud_input = point_cloud_preprocessing(array, voxel_size, cord_range, num_pc_features, max_num_points_per_voxel, max_num_voxels)
    #print("Point cloud Input", point_cloud_input)
    with torch.no_grad():
        pred_dicts, _ = model(point_cloud_input)
        #print("Pred_dicts: ", pred_dicts)
        pointcloud = array
        predictions = pred_dicts
        for pred in pred_dicts:
            pred_boxes = pred['pred_boxes']
            pred_scores = pred['pred_scores']
            pred_labels = pred['pred_labels']

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
     

def save_pcd(points, filename):
    """ 
    Save points to a .pcd file
    """
    # Convert to open3d format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Using only x, y, z columns
    
    # Save to pcd
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved {filename}")


# Initialize a global counter for detections
detection_counter = 0

def update_and_save_pcd(pointcloud, predictions, save_path="/home/duho/VoxelNeXt/tools/detectionsave"):
    global detection_counter  # Use the global counter
    
    for pred in predictions:
        pred_boxes = pred['pred_boxes']  # [center_x, center_y, center_z, width, length, height, rotation]
        pred_scores = pred['pred_scores']
        pred_labels = pred['pred_labels']
        
        for idx, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            if score > 0.25 and label == 4:  # Filter based on score and label
                detection_counter += 1  # Increment the detection counter for each saved file
                
                center = box[:3].cpu().numpy()  # x, y, z of the center
                width, length, height = box[3].cpu().numpy(), box[4].cpu().numpy(), box[5].cpu().numpy()
                rotation = box[6].cpu().numpy() - 1.6
                
                # Convert rotation and box dimensions to find corner points
                corners = np.array([[-length / 2, -width / 2, -height / 2],
                                    [-length / 2, width / 2, -height / 2],
                                    [length / 2, width / 2, -height / 2],
                                    [length / 2, -width / 2, -height / 2],
                                    [-length / 2, -width / 2, height / 2],
                                    [-length / 2, width / 2, height / 2],
                                    [length / 2, width / 2, height / 2],
                                    [length / 2, -width / 2, height / 2]])
                
                # Rotate the corners by the box rotation
                rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                                            [np.sin(rotation), np.cos(rotation), 0],
                                            [0, 0, 1]])
                rotated_corners = np.dot(corners, rotation_matrix.T)
                corners_3d = rotated_corners + center
                
                # Check which points fall inside the bounding box
                in_box_indices = np.all((pointcloud[:, :3] >= corners_3d.min(axis=0)) & 
                                        (pointcloud[:, :3] <= corners_3d.max(axis=0)), axis=1)
                
                points_in_box = pointcloud[in_box_indices]
                
                # Save the points within the bounding box to a .pcd file with detection order in the filename
                filename = os.path.join(save_path, f"detection_{detection_counter:04d}_box_{idx}_label_{label}_score_{score:.2f}.pcd")
                save_pcd(points_in_box, filename)
                print(f"Saved {filename}")

rospy.init_node("voxelnext")
sub = rospy.Subscriber("/flipped_velodyne_points", PointCloud2, pc_cb)

r = rospy.Rate(10)

while not rospy.is_shutdown():
    if pointcloud is not None:  # Ensure pointcloud has data
        update_and_save_pcd(pointcloud, predictions)  # Update the plot with the new data
    r.sleep()

