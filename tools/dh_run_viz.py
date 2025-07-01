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
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

import logging
from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model configuration and weights
config_path = "/home/duho/VoxelNeXt/models/isarc_2/voxelnext_isarc.yaml"
cfg_from_yaml_file(config_path, cfg)
model_path = "/home/duho/VoxelNeXt/models/isarc_2/checkpoint_epoch_50.pth"

pointcloud = None
predictions = None

data_set, data_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,
    dist=False,
    workers=1,
    logger=logger,
    training=False
)

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=data_set)
print("model build")
model.load_params_from_file(filename=model_path, logger=logger, to_cpu=False, pre_trained_path=None)
print("model load")
model.cuda()
model.eval()

# Color map for labels
label_colors = [
    [0.0, 0.0, 0.0],  # 0
    [1.0, 0.0, 0.0],  # 1 dumptruck
    [0.0, 1.0, 0.0],  # 2 dozer
    [0.0, 0.0, 1.0],  # 3 excavator
    [0.0, 1.0, 1.0],  # 4 grader
    [1.0, 0.0, 1.0],  # 5 roller
    [1.0, 1.0, 0.0],  # 6
]


# Publisher for RViz bounding box markers
marker_pub = rospy.Publisher("/bounding_boxes", MarkerArray, queue_size=10)

#For 3channel
'''
def pointcloud2_to_array(msg):
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
    generator = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    points_array = np.array(list(generator), dtype=dtype)

    if points_array.ndim == 1 and points_array.dtype.names is not None:  
        points_array = points_array.view(np.float32).reshape(points_array.shape[0], -1)

    return points_array
'''
#For 4channel 
def pointcloud2_to_array(msg):
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
    generator = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    points_array = np.array(list(generator), dtype=dtype)
    if points_array.ndim == 1 and points_array.dtype.names is not None:  # Structured array
        points_array = points_array.view(np.float32).reshape(points_array.shape[0], -1)
    return points_array

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

def create_marker(id, position, dimensions, orientation, color):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "bounding_boxes"
    marker.id = id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = position
    marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = orientation

    marker.scale.x, marker.scale.y, marker.scale.z = dimensions
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color[0], color[1], color[2], 0.6
    marker.lifetime = rospy.Duration(1.0)
    return marker

def publish_bounding_boxes(predictions):
    marker_array = MarkerArray()
    marker_id = 0
    for pred in predictions:
        for box, score, label in zip(pred['pred_boxes'], pred['pred_scores'], pred['pred_labels']):
            if score > 0.7 and label > 0:
                cx, cy, cz = box[:3].cpu().numpy()
                l, w, h = box[3:6].cpu().numpy()
                yaw = box[6].cpu().numpy() - 1.6
                q = [0, 0, np.sin(yaw / 2), np.cos(yaw / 2)]
                color = label_colors[label] if label < len(label_colors) else [1.0, 1.0, 1.0]
                marker = create_marker(marker_id, (cx, cy, cz), (l, w, h), q, color)
                marker_array.markers.append(marker)
                marker_id += 1
    marker_pub.publish(marker_array)

def pc_cb(msg):
    global pointcloud, predictions
    array = pointcloud2_to_array(msg)
    voxel_size = [0.1, 0.1, 0.2]
    #cord_range = [-49.6, -49.6, -1.0, 49.6, 49.6, 7.0] 
    cord_range = [-70.4, -70.4, -6, 70.4, 70.4, 10]
    num_pc_features = 3
    max_num_points_per_voxel = 5
    max_num_voxels = 150000
    point_cloud_input = point_cloud_preprocessing(array, voxel_size, cord_range, num_pc_features, max_num_points_per_voxel, max_num_voxels)
    #print("Point cloud Input", point_cloud_input)
    with torch.no_grad():
        pred_dicts, _ = model(point_cloud_input)
        #print("Pred_dicts: ", pred_dicts)
        pointcloud = array
        predictions = pred_dicts
        print("Predictions: ", pred_dicts)
        for pred in pred_dicts:
            pred_boxes = pred['pred_boxes']
            pred_scores = pred['pred_scores']
            pred_labels = pred['pred_labels']


     


rospy.init_node("voxelnext")
sub = rospy.Subscriber("/flipped_velodyne_points", PointCloud2, pc_cb)

r = rospy.Rate(10)

while not rospy.is_shutdown():
    if pointcloud is not None:  # Ensure pointcloud has data
        publish_bounding_boxes(predictions)
    r.sleep()

