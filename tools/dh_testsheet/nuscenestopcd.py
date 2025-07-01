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
        
def save_array_as_pcd(file_path, array):
    with open(file_path, 'w') as f:
        # Write the PCD header
        f.write("VERSION .7\n")
        f.write("FIELDS x y z intensity channel\n")
        f.write("SIZE 4 4 4 4 4\n")  # Assuming all fields are float32
        f.write("TYPE F F F F F\n")  # F indicates float32
        f.write("COUNT 1 1 1 1 1\n")  # Indicates each field is 1 element
        f.write("WIDTH {}\n".format(array.shape[0]))
        f.write("HEIGHT 1\n")  # Unorganized point cloud has HEIGHT 1
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")  # Default viewpoint
        f.write("POINTS {}\n".format(array.shape[0]))
        f.write("DATA ascii\n")  # Use 'ascii' for human-readable format; 'binary' is also an option

        # Write the point data
        for point in array:
            f.write(' '.join(map(str, point)) + '\n')
            

file_path = "/home/duho/VoxelNeXt/data/custom/points/nus10.bin"
output_pcd_path = "/home/duho/VoxelNeXt/data/custom/points/nus10.pcd"

points_array = binary_file_to_numpy_array(file_path)
save_array_as_pcd(output_pcd_path, points_array)


