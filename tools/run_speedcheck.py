import rospy
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper
import time
import logging
from sensor_msgs.msg import PointCloud2
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config_path = "/home/duho/VoxelNeXt/models/customV2.1/voxelnext.yaml"
cfg_from_yaml_file(config_path, cfg)
model_path = "/home/duho/VoxelNeXt/models/customV2.1/checkpoint_epoch_50.pth"

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
model.load_params_from_file(filename=model_path, logger=logger, to_cpu=False, pre_trained_path=None)
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
    if points_array.ndim == 1 and points_array.dtype.names is not None:
        points_array = points_array.view(np.float32).reshape(points_array.shape[0], -1)
    return points_array

def pc_cb(msg):
    global model
    start_time = time.time()

    array = pointcloud2_to_array(msg)
    voxel_size = [0.1, 0.1, 0.2]
    cord_range = [-70.4, -70.4, -6, 70.4, 70.4, 10]
    num_pc_features = 4
    max_num_points_per_voxel = 5
    max_num_voxels = 150000
    point_cloud_input = point_cloud_preprocessing(array, voxel_size, cord_range, num_pc_features, max_num_points_per_voxel, max_num_voxels)
    
    with torch.no_grad():
        pred_dicts, _ = model(point_cloud_input)
        end_time = time.time()
        pointcloud = array
        predictions = pred_dicts
        for pred in pred_dicts:
            pred_boxes = pred['pred_boxes']
            pred_scores = pred['pred_scores']
            pred_labels = pred['pred_labels']
        detection_time = end_time - start_time
        detection_hz = 1.0 / detection_time if detection_time > 0 else 0
        print("Detection speed :", detection_hz)

rospy.init_node("voxelnext")
sub = rospy.Subscriber("/flipped_velodyne_points", PointCloud2, pc_cb)

rospy.spin()


