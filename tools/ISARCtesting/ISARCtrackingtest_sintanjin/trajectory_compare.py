import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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

def calculate_arc_length(x, y):
    """Calculates total arc length along a trajectory."""
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.sum(distances)

def calculate_arc_length_cumulative(x, y):
    """Calculates cumulative arc length along a trajectory."""
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    arc_length = np.insert(np.cumsum(distances), 0, 0)  # Start at 0
    return arc_length

def interpolate_trajectory(x, y, num_points):
    """Interpolates a trajectory to a given number of evenly spaced points."""
    arc_length = calculate_arc_length_cumulative(x, y)
    interp_length = np.linspace(0, arc_length[-1], num_points)  # Uniform spacing
    x_interp = interp1d(arc_length, x, kind='linear')(interp_length)
    y_interp = interp1d(arc_length, y, kind='linear')(interp_length)
    return x_interp, y_interp

def calculate_rmse(x1, y1, x2, y2):
    """Calculates RMSE between two trajectories."""
    return np.sqrt(np.mean((x1 - x2)**2 + (y1 - y2)**2))

# File paths
file1 = 'tracking_trajectories.txt'
file2 = 'actual_trails.txt'

# Read trajectories
x1, y1 = read_coordinates(file1)
x2, y2 = read_coordinates(file2)

# Calculate lengths of the trajectories
length_file1 = calculate_arc_length(x1, y1)
length_file2 = calculate_arc_length(x2, y2)

# Interpolate both trajectories to the same number of points
num_interp_points = 100  # Adjust based on desired resolution
x1_interp, y1_interp = interpolate_trajectory(x1, y1, num_interp_points)
x2_interp, y2_interp = interpolate_trajectory(x2, y2, num_interp_points)

# Calculate RMSE
rmse = calculate_rmse(x1_interp, y1_interp, x2_interp, y2_interp)

# Normalize RMSE by the length of the trajectory from file2
normalized_rmse = (rmse / length_file2) * 100  # Express as a percentage

# Plot both trajectories (interpolated)
plt.figure(figsize=(10, 6))
plt.plot(x1_interp, y1_interp, linestyle='-', color='b', label='Autonomous tracking')
plt.plot(x2_interp, y2_interp, linestyle='--', color='r', label='Manual tracking')
plt.title(f'Trajectory Comparison (RMSE = {rmse:.3f}, Normalized RMSE = {normalized_rmse:.2f}%)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.show()

# Output results
print(f"Length of Trajectory 1: {length_file1:.3f}")
print(f"Length of Trajectory 2: {length_file2:.3f}")
print(f"RMSE between the two trajectories: {rmse:.3f}")
print(f"Normalized RMSE (as percentage of Trajectory 2 length): {normalized_rmse:.2f}%")

