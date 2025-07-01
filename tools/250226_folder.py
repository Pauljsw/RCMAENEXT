#!/usr/bin/env python
import glob
import os
import numpy as np
import torch
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper


# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration and model
config_path = "/home/duho/VoxelNeXt/models/customV2.1/voxelnext.yaml"
model_path = "/home/duho/VoxelNeXt/models/customV2.1/checkpoint_epoch_50.pth"

cfg_from_yaml_file(config_path, cfg)

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

model.load_params_from_file(filename=model_path, logger=logger, to_cpu=False, pre_trained_path=None)
model.cuda()
model.eval()

# Define visualization parameters
label_colors = ['r', 'g', 'b', 'c', 'm', 'y']

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

# Function to load PCD file
def load_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points, dtype=np.float32)

# Function to preprocess point cloud
def point_cloud_preprocessing(point_cloud):
    voxel_size = [0.1, 0.1, 0.2]
    cord_range = [-70.4, -70.4, -6, 70.4, 70.4, 10]
    num_pc_features = 3  # x, y, z
    max_num_points_per_voxel = 5
    max_num_voxels = 150000

    voxel_generator = VoxelGeneratorWrapper(
        vsize_xyz=voxel_size,
        coors_range_xyz=cord_range,
        num_point_features=num_pc_features,
        max_num_points_per_voxel=max_num_points_per_voxel,
        max_num_voxels=max_num_voxels
    )

    voxels, coordinates, num_points = voxel_generator.generate(point_cloud)

    # Convert to torch tensors and send to CUDA
    def totensor(npinput):
        ten = torch.from_numpy(npinput)
        return ten.cuda() if torch.cuda.is_available() else ten

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

# Function to rotate points for bounding boxes
def rotate_point(point, angle, origin=(0, 0)):
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

# Function to update the plot
def update_plot(pointcloud, predictions):
    global ax, fig
    ax.clear()  # Clear the previous plot
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Point Cloud Visualization in XY Plane')

    X, Y, Z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]

    # Normalize Z values for color mapping
    z_min, z_max = -0.5, 1.0
    Z[Z < z_min] = z_min
    Z[Z > z_max] = z_max
    norm = Normalize(vmin=z_min, vmax=z_max)
    cmap = cm.cividis_r  # Reversed colormap for light-to-dark Z values

    # Scatter plot for point cloud
    sc = ax.scatter(X, Y, c=Z, cmap=cmap, norm=norm, s=1)

    # Colorbar setup
    if not hasattr(update_plot, "colorbar"):
        update_plot.colorbar = plt.colorbar(sc, ax=ax, label='Z value', extend='both')
    else:
        update_plot.colorbar.update_normal(sc)

    # Draw predictions
    for pred in predictions:
        pred_boxes = pred['pred_boxes']  # center_x, center_y, center_z, width, length, height, rotation
        pred_scores = pred['pred_scores']
        pred_labels = pred['pred_labels']

        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score > 0.25:  # Confidence threshold
                center = box[:2].cpu().numpy()
                width, length = box[3].cpu().numpy(), box[4].cpu().numpy()
                rotation = box[6].cpu().numpy() - 1.6  # Adjust rotation

                # Get rotated bounding box corners
                corners = np.array([
                    [-length / 2, -width / 2], [-length / 2, width / 2],
                    [length / 2, width / 2], [length / 2, -width / 2]
                ])
                rotated_corners = np.array([rotate_point(corner, rotation, origin=(0, 0)) for corner in corners])
                rotated_corners += center

                # Assign colors based on label index
                color = label_colors[label % len(label_colors)]

                # Draw bounding box
                polygon = plt.Polygon(rotated_corners, edgecolor=color, facecolor='none', linewidth=1)
                ax.add_patch(polygon)

    plt.draw()  # Update the plot
    plt.pause(0.001)  # Short pause for GUI update

# Path to PCD files folder
pcd_folder = "/home/duho/Duho_Dataset/pointcloud/Dataset_V2_0520/0522_Test_Dataset/pcd/"

# Get all PCD files
pcd_files = glob.glob(os.path.join(pcd_folder, '*.pcd'))
if not pcd_files:
    print("No PCD files found. Check your folder path.")

# Process and visualize each PCD file
for pcd_file in pcd_files:
    print(f"Processing {pcd_file}...")
    pointcloud = load_pcd(pcd_file)  # Load PCD
    point_cloud_input = point_cloud_preprocessing(pointcloud)  # Preprocess for model
    print("Voxel features shape:", point_cloud_input['voxels'].shape)  # (num_voxels, num_points_per_voxel, num_features)
    print("Point cloud tensor shape:", point_cloud_input['points'].shape)  # (num_points, num_features)


    with torch.no_grad():
        pred_dicts, _ = model(point_cloud_input)  # Run model inference
        print(f"Predictions for {os.path.basename(pcd_file)}: {pred_dicts}")

    # Update visualization
    update_plot(pointcloud, pred_dicts)

print("Processing completed.")
plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the final visualization open

