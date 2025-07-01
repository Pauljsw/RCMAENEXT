import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from matplotlib.colors import Normalize

#Global variables
pointcloud = None

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
    pointcloud = array 
     
plt.ion() 
fig, ax = plt.subplots()

def update_plot(array):
    global ax, fig
    ax.clear()  # Clear the axes for the new plot
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Point Cloud Visualization in XY Plane')
    
    X = array[:, 0]
    Y = array[:, 1]
    Z = array[:, 2]
    
    # Set the fixed range for Z values
    z_min, z_max = -0.5, 2.0
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
    
    plt.draw()  # Redraw the plot with the new data
    plt.pause(0.001)  # Short pause to allow the GUI to update





rospy.init_node("voxnext")
sub = rospy.Subscriber("/velodyne_points", PointCloud2, pc_cb)

r = rospy.Rate(4)


while not rospy.is_shutdown():
    if pointcloud is not None:  # Ensure pointcloud has data
        update_plot(pointcloud)  # Update the plot with the new data
    r.sleep()

