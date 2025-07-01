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

#Global variables
pointcloud = None
predictions = None


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
    global pointcloud, predictions
    array = pointcloud2_to_array(msg)
    voxel_size = [0.075, 0.075, 0.2]
    cord_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0] 
    num_pc_features = 5
    max_num_points_per_voxel = 10
    max_num_voxels = 160000
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
     
plt.ion() 
fig, ax = plt.subplots()

def update_plot(pointcloud, predictions):
    global ax, fig
    ax.clear()  # Clear the axes for the new plot
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Point Cloud Visualization in XY Plane')

    X = pointcloud[:, 0]
    Y = pointcloud[:, 1]
    Z = pointcloud[:, 2]
    # Set the fixed range for Z values
    z_min, z_max = -0.5, 1.0
    Z[Z < z_min] = z_min  # Set any Z value lower than z_min to z_min
    Z[Z > z_max] = z_max  # Set any Z value higher than z_max to z_max
    
    norm = Normalize(vmin=z_min, vmax=z_max)
    
    # Set the colormap (You might choose a different colormap that suits your preference for light-to-dark mapping)
    cmap = cm.cividis_r  # '_r' suffix to reverse the colormap, light to dark for increasing Z
    
    # Create scatter plot
    sc = ax.scatter(X, Y, c=Z, cmap=cmap, norm=norm, s=1)  # 's' sets the marker size
    
    # Check if the colorbar already exists
    if not hasattr(update_plot, "colorbar"):
        update_plot.colorbar = plt.colorbar(sc, ax=ax, label='Z value', extend='both')
    else:
        # Update the colorbar with the new data
        update_plot.colorbar.update_normal(sc)
    
    for pred in predictions:
        pred_boxes = pred['pred_boxes']  # Assuming these are [center_x, center_y, center_z, width, length, height, rotation, velocity_x, velocity_y]
        pred_scores = pred['pred_scores']
        pred_labels = pred['pred_labels']
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score > 0.5:  
                print("Box", box, "Score: ", score, "Label: ", label)
                center = box[:2].cpu().numpy()
                width, length = box[3].cpu().numpy(), box[4].cpu().numpy()
                rotation = box[6].cpu().numpy() - 1.6
                #rotation_deg = np.degrees(rotation)
                velocity = box[7:].cpu().numpy()

                # Plot the box into the xy plane
                corners = np.array([[-length / 2, -width / 2], [-length / 2, width / 2], [length / 2, width / 2], [length / 2, -width / 2] ])  
                rotated_corners = np.array([rotate_point(corner, rotation, origin=(0, 0)) for corner in corners])
                rotated_corners += center
                polygon = plt.Polygon(rotated_corners, edgecolor='r', facecolor='none', linewidth=1)
                ax.add_patch(polygon)
                
                #ax.scatter(center[0], center[1], s=100, c='red', marker='o', label=f'Label {label}') 
                # Show velocity as an arrow
                #ax.quiver(*center, *velocity, color='r', scale=5)

                # Annotate the box with scores and labels
                #ax.annotate(f'{label}: {score:.2f}', (center[0], center[1]), color='white', weight='bold', fontsize=8, ha='center', va='center')
    

    
    plt.draw()  # Redraw the plot with the new data
    plt.pause(0.001)  # Short pause to allow the GUI to update




rospy.init_node("voxnext")
sub = rospy.Subscriber("/flipped_velodyne_points", PointCloud2, pc_cb)

r = rospy.Rate(10)

while not rospy.is_shutdown():
    if pointcloud is not None:  # Ensure pointcloud has data
        update_plot(pointcloud, predictions)  # Update the plot with the new data
    r.sleep()

