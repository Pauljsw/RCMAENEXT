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
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from message_filters import Subscriber, TimeSynchronizer

from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import Point

import logging
import yaml
from pcdet.datasets.processor.data_processor import DataProcessor, VoxelGeneratorWrapper
from visualization_msgs.msg import Marker, MarkerArray


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#config_path = "/home/duho/VoxelNeXt/models/isarc_3channel/voxelnext_isarc_3chan.yaml"
config_path = "/home/duho/VoxelNeXt/models/onlysynthetic_0522/voxelnext_isarc_3chan.yaml"

cfg_from_yaml_file(config_path, cfg)

#model_path = "/home/duho/VoxelNeXt/models/isarc_3channel/checkpoint_epoch_50.pth"
model_path = "/home/duho/VoxelNeXt/models/onlysynthetic_0522/checkpoint_epoch_21.pth"

data_set, data_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,  # Batch size can be set to 1 for inference
    dist=False,  # Assuming no distributed inference for simplicity
    workers=1,  # Worker threads can be reduced if not loading batches of data
    logger=logger,
    training=False  # Indicate evaluation/inference mode
)

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



label_colors = [
    [0.0, 0.0, 0.0],  # Black #0
    [1.0, 0.0, 0.0],  # Red #1 dumptruck
    [0.0, 1.0, 0.0],  # Green  #2 dozer
    [0.0, 0.0, 1.0],  # Blue   #3 excavator
    [0.0, 1.0, 1.0],  # Cyan   #4 grader
    [1.0, 0.0, 1.0],  # Magenta #5 roller
    [1.0, 1.0, 0.0],   # Yellow #6
]

def pointcloud2_to_array(msg):
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
    generator = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    points_array = np.array(list(generator), dtype=dtype)
    if points_array.ndim == 1 and points_array.dtype.names is not None:  # Structured array
        points_array = points_array.view(np.float32).reshape(points_array.shape[0], -1)
    return points_array



def pc_cb(msg):
    global pointcloud, predictions
    raw_array = pointcloud2_to_array(msg)
    
    voxel_size = [0.1, 0.1, 0.2]
    #cord_range = [-49.6, -49.6, -1.0, 49.6, 49.6, 7.0] 
    cord_range = [-70.4, -70.4, -6, 70.4, 70.4, 10]
    num_pc_features = 3
    max_num_points_per_voxel = 5
    max_num_voxels = 150000
    point_cloud_input = point_cloud_preprocessing(raw_array, voxel_size, cord_range, num_pc_features, max_num_points_per_voxel, max_num_voxels)
    with torch.no_grad():
        pred_dicts, _ = model(point_cloud_input)
        pointcloud = raw_array
        if pred_dicts is not None:
            print("Pred_dicts is not None...")
        predictions = pred_dicts
        if predictions is not None:
            print("predictions is not None...")

        #print("Predictions ", predictions)
        
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
     


def create_marker(id, position, dimensions, orientation, color):
    marker = Marker()
    marker.header.frame_id = "velodyne"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "bounding_boxes"
    marker.id = id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    
    # Set position
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    
    # Set orientation
    marker.pose.orientation.x = orientation[0]
    marker.pose.orientation.y = orientation[1]
    marker.pose.orientation.z = orientation[2]
    marker.pose.orientation.w = orientation[3]
    
    # Set dimensions
    marker.scale.x = dimensions[0]
    marker.scale.y = dimensions[1]
    marker.scale.z = dimensions[2]
    
    # Set color
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 0.6 # Adjust transparency if desired
    
    marker.lifetime = rospy.Duration(1)  # Set the lifetime of each marker
    return marker


def publish_bounding_boxes(predictions):
    marker_array = MarkerArray()
    marker_id = 0
    for pred in predictions:
        pred_boxes = pred['pred_boxes']
        pred_scores = pred['pred_scores']
        pred_labels = pred['pred_labels']
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score > 0.5 and label > 0:
                # Center position of bounding box
                center_x, center_y, center_z = box[:3].cpu().numpy()
                position = (center_x, center_y, center_z)

                
                
                # Dimensions and rotation
                width, length, height = box[3].cpu().numpy(), box[4].cpu().numpy(), box[5].cpu().numpy()
                rotation = box[6].cpu().numpy() - 1.6
                orientation = [0, 0, np.sin(rotation / 2), np.cos(rotation / 2)]  # Quaternion from rotation angle
                
                # Set color based on label
                if label < len(label_colors):
                    color = label_colors[label]
                else:
                    color = [1.0, 1.0, 1.0]  # Default color if label out of range
                    
                                
                # Create and add marker
                marker = create_marker(marker_id, position, [length, width, height], orientation, color)
                marker_array.markers.append(marker)
                marker_id += 1
    print("Publishing box")
    # Publish the MarkerArray
    marker_pub.publish(marker_array)
    

    
    
# ROS node initialization and loop
rospy.init_node("voxelnext")    
# Subscribers 
sub = rospy.Subscriber("/flipped_velodyne_points", PointCloud2, pc_cb)
#sub = rospy.Subscriber("/velodyne_points", PointCloud2, pc_cb)
#sub = rospy.Subscriber("/xyz_only_cloud", PointCloud2, pc_cb)
#publishers
marker_pub = rospy.Publisher('/bounding_boxes', MarkerArray, queue_size=10)


r = rospy.Rate(10)

while not rospy.is_shutdown():
    print("Predictions: ", predictions)
    if predictions is not None:  
        print("Predicted")
        publish_bounding_boxes(predictions)  # Publish bounding boxes
    r.sleep()
