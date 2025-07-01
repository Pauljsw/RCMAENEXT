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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config_path = "/home/duho/VoxelNeXt/models/cbgs_voxel0075_voxelnext.yaml"
cfg_from_yaml_file(config_path, cfg)
#print(cfg)
model_path = "/home/duho/VoxelNeXt/models/voxelnext_nuscenes_double.pth"

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



def pointcloud2_to_array(msg):
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
    generator = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    points_array = np.array(list(generator), dtype=dtype)
    # If points_array is structured, convert it to a 2D array
    if points_array.ndim == 1 and points_array.dtype.names is not None:  # Structured array
        points_array = points_array.view(np.float32).reshape(points_array.shape[0], -1)
    # Create a new column of zeros with the same number of rows as points_array
    zeros_column = np.zeros((points_array.shape[0], 1), dtype=np.float32)
    # Append the zeros column to the points_array
    final_array = np.hstack((points_array, zeros_column))
    return final_array


def pc_cb(msg):
    array = pointcloud2_to_array(msg)
    voxel_size = [0.075, 0.075, 0.2]
    cord_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0] 
    num_pc_features = 5
    max_num_points_per_voxel = 10
    max_num_voxels = 160000
    point_cloud_input = point_cloud_preprocessing(array, voxel_size, cord_range, num_pc_features, max_num_points_per_voxel, max_num_voxels)
    with torch.no_grad():
        pred_dicts, _ = model(point_cloud_input)
        for pred in pred_dicts:
            pred_boxes = pred['pred_boxes']
            pred_scores = pred['pred_scores']
            pred_labels = pred['pred_labels']
            # Filter boxes with scores greater than 0.25
            mask = pred_scores > 0.25
            filtered_boxes = pred_boxes[mask]
            filtered_scores = pred_scores[mask]
            filtered_labels = pred_labels[mask]
            
            
            # Compute and print the midpoints of the filtered boxes
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                # The midpoint is the first three values (x, y, z) of the box
                midpoint = box[:3].cpu().numpy()  # Ensure tensor is moved to CPU and converted to numpy array
                print("Midpoint:", midpoint, "Score:", score.item(), "Label:", label.item())
     


rospy.init_node("voxnext")
sub = rospy.Subscriber("/velodyne_points", PointCloud2, pc_cb)

r = rospy.Rate(4)

while not rospy.is_shutdown():
    r.sleep()


