#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:20:23 2024

@author: duho
"""
import numpy as np


def binary_file_to_numpy_array(file_path):
    # Define the data type for each point in the point cloud
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('c', np.float32)]
    # Read the entire file into a bytes object
    with open(file_path, "rb") as f:
        file_content = f.read()
    # Convert the bytes data to a structured numpy array with the specified dtype
    points_array = np.frombuffer(file_content, dtype=dtype)
    # If points_array is structured, convert it to a 2D array
    if points_array.ndim == 1 and points_array.dtype.names is not None:  # Structured array
        points_array = points_array.view(np.float32).reshape(points_array.shape[0], -1)
    return points_array
        
def pcd_to_numpy_array(pcd_path):
    with open(pcd_path, 'r') as f:
        # Skip the header lines
        while True:
            line = f.readline().strip()
            if line.startswith('DATA'):
                break
        # Read the point data
        data = []
        for line in f:
            parts = line.strip().split()
            data.append([float(part) for part in parts])
    # Convert to numpy array
    points_array = np.array(data, dtype=np.float32)
    # Assuming the PCD has x, y, z, and intensity fields in this order
    # Check if points_array has only x, y, z and append a zeros column if intensity is missing
    if points_array.shape[1] == 3:
        zeros_column = np.zeros((points_array.shape[0], 1), dtype=np.float32)
        points_array = np.hstack((points_array, zeros_column))
    # Append an additional zeros column to the end, as per your original function
    additional_zeros_column = np.zeros((points_array.shape[0], 1), dtype=np.float32)
    final_array = np.hstack((points_array, additional_zeros_column))
    return final_array



file_path_pcd = "/home/duho/0307test/velodyne_frame_3.pcd"
point_cloud_array_rd=pcd_to_numpy_array(file_path_pcd)

file_path_bin = "/home/duho/VoxelNeXt/random3.pcd.bin"
point_cloud_array_nuc = binary_file_to_numpy_array(file_path_bin)
