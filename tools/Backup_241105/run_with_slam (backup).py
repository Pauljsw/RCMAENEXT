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
from message_filters import Subscriber, TimeSynchronizer

from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import Point

import logging
import yaml
from pcdet.datasets.processor.data_processor import DataProcessor, VoxelGeneratorWrapper
from visualization_msgs.msg import Marker, MarkerArray


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config_path = "/home/duho/VoxelNeXt/models/customV2.1/voxelnext.yaml"
cfg_from_yaml_file(config_path, cfg)

model_path = "/home/duho/VoxelNeXt/models/customV2.1/checkpoint_epoch_50.pth"

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
slam_pointcloud = None
pointcloud = None
predictions = None
odom = None
x, y, theta = 0, 0, 0

def euler_from_quaternion(quaternion):
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw
def odometry_cb(msg):
    global x, y, theta
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    rot_q = msg.pose.pose.orientation
    _, _, theta = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])




label_colors = [
    [1.0, 0.0, 0.0],  # Red
    [0.0, 1.0, 0.0],  # Green
    [0.0, 0.0, 1.0],  # Blue
    [0.0, 1.0, 1.0],  # Cyan
    [1.0, 0.0, 1.0],  # Magenta
    [1.0, 1.0, 0.0]   # Yellow
]

def pointcloud2_to_array(msg):
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
    generator = point_cloud2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    points_array = np.array(list(generator), dtype=dtype)
    if points_array.ndim == 1 and points_array.dtype.names is not None:  # Structured array
        points_array = points_array.view(np.float32).reshape(points_array.shape[0], -1)
    return points_array



def pc_cb(pcslam_msg, odom_msg, raw_pc_msg):
    global slam_pointcloud, pointcloud, predictions, x, y, theta
    odometry_cb(odom_msg)
    raw_array = pointcloud2_to_array(raw_pc_msg)
    slam_array = pointcloud2_to_array(pcslam_msg)
    
    voxel_size = [0.1, 0.1, 0.2]
    #cord_range = [-49.6, -49.6, -1.0, 49.6, 49.6, 7.0] 
    cord_range = [-70.4, -70.4, -6, 70.4, 70.4, 10]
    num_pc_features = 4
    max_num_points_per_voxel = 5
    max_num_voxels = 150000
    point_cloud_input = point_cloud_preprocessing(raw_array, voxel_size, cord_range, num_pc_features, max_num_points_per_voxel, max_num_voxels)
    #print("Point cloud Input", point_cloud_input)
    with torch.no_grad():
        pred_dicts, _ = model(point_cloud_input)
        #print("Pred_dicts: ", pred_dicts)
        pointcloud = raw_array
        predictions = pred_dicts
        #print("Predictions: ", pred_dicts)
        for pred in pred_dicts:
            pred_boxes = pred['pred_boxes']
            pred_scores = pred['pred_scores']
            pred_labels = pred['pred_labels']
        
        pointcloud = raw_array
        slam_pointcloud = slam_array

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
     

marker_pub = rospy.Publisher('/bounding_boxes', MarkerArray, queue_size=10)

def create_marker(id, position, dimensions, orientation, color):
    marker = Marker()
    marker.header.frame_id = "map"  # Change this to your coordinate frame
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
    marker.color.a = 0.5  # Adjust transparency if desired
    
    marker.lifetime = rospy.Duration(1)  # Set the lifetime of each marker
    return marker


def publish_bounding_boxes(predictions, x, y, theta):
    marker_array = MarkerArray()
    marker_id = 0
    
    for pred in predictions:
        pred_boxes = pred['pred_boxes']
        pred_scores = pred['pred_scores']
        pred_labels = pred['pred_labels']
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score > 0.5:
                # Center position of bounding box
                center_x, center_y, center_z = box[:3].cpu().numpy()
                x_slam = np.cos(theta) * center_x - np.sin(theta) * center_y + x
                y_slam = np.sin(theta) * center_x + np.cos(theta) * center_y + y
                position = [x_slam, y_slam, center_z]
                
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

    # Publish the MarkerArray
    marker_pub.publish(marker_array)



rospy.init_node("voxelnext")
# Subscribers for the three topics
pcslam_sub = Subscriber('/lio_sam/mapping/cloud_output', PointCloud2)
odo_sub = Subscriber('/lio_sam/mapping/odometry', Odometry)
#pcraw_sub = Subscriber("/flipped_velodyne_points", PointCloud2)
pcraw_sub = Subscriber("/lio_sam/deskew/cloud_deskewed", PointCloud2)

sync = TimeSynchronizer([pcslam_sub, odo_sub, pcraw_sub], 30)
sync.registerCallback(pc_cb)

r = rospy.Rate(10)

while not rospy.is_shutdown():
    if slam_pointcloud is not None:  # Ensure slam_pointcloud has data
        publish_bounding_boxes(predictions, x, y, theta)  # Publish bounding boxes
    r.sleep()
