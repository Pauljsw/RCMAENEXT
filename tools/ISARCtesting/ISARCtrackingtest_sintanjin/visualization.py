import numpy as np
import open3d as o3d

def read_coordinates(file_path):
    """Reads a file and extracts the first two columns as x, y coordinates."""
    x_coords = []
    y_coords = []
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.strip().split(',')
            x_coords.append(float(columns[0].strip()))
            y_coords.append(float(columns[1].strip()))
    return np.array(x_coords), np.array(y_coords)

def calculate_arc_length_cumulative(x, y):
    """Calculates cumulative arc length along a trajectory."""
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    arc_length = np.insert(np.cumsum(distances), 0, 0)  # Start at 0
    return arc_length

def interpolate_trajectory(x, y, num_points):
    """Interpolates a trajectory to a given number of evenly spaced points using numpy."""
    arc_length = calculate_arc_length_cumulative(x, y)
    interp_length = np.linspace(0, arc_length[-1], num_points)  # Uniform spacing
    x_interp = np.interp(interp_length, arc_length, x)
    y_interp = np.interp(interp_length, arc_length, y)
    return x_interp, y_interp

def create_lineset(x, y, z_value=0, color=[1, 0, 0]):
    """Creates an Open3D LineSet for a trajectory."""
    points = np.column_stack((x, y, np.full_like(x, z_value)))  # Add z=0 for all points
    lines = [[i, i + 1] for i in range(len(points) - 1)]  # Line indices
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))  # Set line color
    return line_set

# File paths
file1 = 'tracking_trajectories.txt'
file2 = 'actual_trails.txt'
global_map_pcd_path = 'GlobalMap.pcd'  # Path to the .pcd file

# Read trajectories
x1, y1 = read_coordinates(file1)
x2, y2 = read_coordinates(file2)

# Interpolate both trajectories to the same number of points
num_interp_points = 100  # Adjust based on desired resolution
x1_interp, y1_interp = interpolate_trajectory(x1, y1, num_interp_points)
x2_interp, y2_interp = interpolate_trajectory(x2, y2, num_interp_points)

# Create line sets for both trajectories
line1 = create_lineset(x1_interp, y1_interp, z_value=0, color=[0, 0, 1])  # Trajectory 1 (blue)
line2 = create_lineset(x2_interp, y2_interp, z_value=1, color=[1, 0, 0])  # Trajectory 2 (red)

# Load the GlobalMap.pcd file
global_map_pcd = o3d.io.read_point_cloud(global_map_pcd_path)

# Visualize the global map and trajectories as lines
o3d.visualization.draw_geometries([global_map_pcd, line1, line2],
                                  window_name="Trajectory and Global Map Comparison",
                                  width=800, height=600, point_show_normal=False)



