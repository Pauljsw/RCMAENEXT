#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point
import numpy as np
import math
import time
from prettytable import PrettyTable

from sklearn.linear_model import LinearRegression  # Import Linear Regression for speed calculation
import tf.transformations as tf_trans


# Define the color-to-label mapping
label_colors = [
    [0.0, 0.0, 0.0],  # Black #0
    [1.0, 0.0, 0.0],  # Red #1 Worker
    [0.0, 1.0, 0.0],  # Green  #2 Car
    [0.0, 0.0, 1.0],  # Blue   #3 Truck
    [0.0, 1.0, 1.0],  # Cyan   #4 Excavator
    [1.0, 0.0, 1.0],  # Magenta #5 CrawlingDrill
    [1.0, 1.0, 0.0],  # Yellow #6 Grader
]


last_processed_time = None

def get_label_from_color(r, g, b):
    color = [r, g, b]
    for i, label_color in enumerate(label_colors):
        if np.allclose(color, label_color, atol=0.01):  # Allowing slight tolerance
            return i
    return -1  # Undefined color
    


# Tracking variables
excavator_index = []  # List of tracked excavator positions by index
inactive_counter = []  # List to track inactive frames for each index
max_inactive_frames = 10  # Max frames before removing stale indices
distance_threshold_base = 5  # Threshold for matching

# Initialize ROS node
rospy.init_node('excavator_tracker', anonymous=True)

# Visualization publisher
trail_pub = rospy.Publisher('/excavator_trails', MarkerArray, queue_size=10)

def euclidean_distance(pos1, pos2):
    """ Calculate the Euclidean distance between two 3D points. """
    return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2)



def calculate_average_speed(positions):
    """ Calculate the average speed and direction using the last 10 frames (or fewer if not available, down to 5). """
    num_frames = len(positions)
    # Use up to the last 10 frames, but require at least 5 frames
    if num_frames < 5:
        return 0, (0, 0, 0)  # Not enough data for reliable speed calculation
    frames_to_use = min(num_frames, 10)  # Use the last 10 frames if possible, else use available frames
    recent_positions = positions[-frames_to_use:]
    # Time vector for linear regression, e.g., (1, 2, ..., frames_to_use), assuming 1-second intervals
    time_vector = np.arange(1, frames_to_use + 1).reshape(-1, 1)
    # Linear regression for each axis
    lr_x = LinearRegression().fit(time_vector, [pos[0] for pos in recent_positions])
    lr_y = LinearRegression().fit(time_vector, [pos[1] for pos in recent_positions])
    lr_z = LinearRegression().fit(time_vector, [pos[2] for pos in recent_positions])
    # Slopes (velocities) in each direction
    velocity_x = lr_x.coef_[0]
    velocity_y = lr_y.coef_[0]
    velocity_z = lr_z.coef_[0]
    # Calculate speed and direction
    speed = math.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)
    direction = (velocity_x, velocity_y, velocity_z)
    return speed, direction
    

def display_tracking_status():
    """ Display the current status of tracked excavators in a table. """
    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = ["Index", "Trail Length", "Latest X", "Latest Y", "Latest Z", "Inactive Frames", "Speed", "Direction"]

    for idx, positions in enumerate(excavator_index):
        trail_length = len(positions)  # Number of connected locations
        if trail_length < 5:
            continue  # Skip if trail length is less than 5

        latest_position = positions[-1]
        inactive_frames = inactive_counter[idx]
        
        # Calculate average speed and direction
        speed, direction = calculate_average_speed(positions)
        
        # Format direction to two decimal places for display
        direction_str = f"({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})"
        table.add_row([
            idx, 
            trail_length,
            round(latest_position[0], 2), 
            round(latest_position[1], 2), 
            round(latest_position[2], 2), 
            inactive_frames,
            round(speed, 2),
            direction_str
        ])

    print(table)

def closest_matching(position_list):
    global excavator_index, inactive_counter
    updated_indices = set() # Set to track which indices have been updated
    new_positions = []  # List to store positions that need a new index
    # Step 1: Update existing indices with closest matching positions
    for position in position_list:
        min_distance = min_distance = float('inf')
        closest_index = None    

        for idx, tracked_positions in enumerate(excavator_index):
            last_position = tracked_positions[-1]
            distance_threshold = distance_threshold_base * (1 + inactive_counter[idx])  # Adjust threshold
            distance = euclidean_distance(position, last_position)
            if distance < min_distance and distance < distance_threshold:
                min_distance = distance
                closest_index = idx
        # Update the closest index if within the threshold
        if closest_index is not None:
            excavator_index[closest_index].append(position)
            inactive_counter[closest_index] = 0  # Reset inactive counter
            updated_indices.add(closest_index)
        else:
            new_positions.append(position)  # Position needs a new index


    # Step 2: Create new indices for unmatched positions
    for position in new_positions:
        excavator_index.append([position])
        inactive_counter.append(0)

    # Step 3: Update inactive counters and remove stale indices
    for idx in range(len(excavator_index) - 1, -1, -1):
        if idx not in updated_indices:
            inactive_counter[idx] += 1
            if inactive_counter[idx] > max_inactive_frames:
                # Remove stale index
                excavator_index.pop(idx)
                inactive_counter.pop(idx)
    display_tracking_status()

def visualize_trails():
    """ Publishes trails of tracked excavators in RViz with arrows floating above the last position. """
    marker_array = MarkerArray()

    for idx, positions in enumerate(excavator_index):
        if len(positions) < 2:
            continue  # Skip if fewer than two points in the trail

        # Create a LINE_STRIP marker for the trail
        trail_marker = Marker()
        trail_marker.header = Header(stamp=rospy.Time.now(), frame_id="map")
        trail_marker.ns = f"excavator_trail_{idx}"
        trail_marker.id = idx
        trail_marker.type = Marker.LINE_STRIP
        trail_marker.action = Marker.ADD
        trail_marker.scale.x = 0.1  # Line width

        # Color: Cyan for trail
        trail_marker.color.r = 0.0
        trail_marker.color.g = 1.0
        trail_marker.color.b = 1.0
        trail_marker.color.a = 1.0
        trail_marker.points = [Point(x=pos[0], y=pos[1], z=pos[2]) for pos in positions]
        trail_marker.pose.orientation.w = 1.0  # Identity quaternion
        marker_array.markers.append(trail_marker)

        # Calculate speed and direction for the arrow marker
        speed, direction = calculate_average_speed(positions)
        latest_position = positions[-1]

        # Create an ARROW marker positioned at the latest position, oriented in the direction
        arrow_marker = Marker()
        arrow_marker.header = Header(stamp=rospy.Time.now(), frame_id="map")
        arrow_marker.ns = f"excavator_arrow_{idx}"
        arrow_marker.id = idx + 1000  # Unique ID offset
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD

        # Set scale to make the arrow appear fixed in size
        arrow_marker.scale.x = 0.5  # Arrow shaft length
        arrow_marker.scale.y = 0.1  # Arrow head width
        arrow_marker.scale.z = 0.1  # Arrow head length

        # Color: Green for arrow to indicate direction
        arrow_marker.color.r = 0.0
        arrow_marker.color.g = 1.0
        arrow_marker.color.b = 0.0
        arrow_marker.color.a = 1.0

        # Position the arrow at the latest position
        arrow_marker.pose.position.x = latest_position[0]
        arrow_marker.pose.position.y = latest_position[1]
        arrow_marker.pose.position.z = latest_position[2] + 0.5  # Floating slightly above the position

        # Calculate the 2D angle in the xy-plane and set quaternion
        angle = math.atan2(direction[1], direction[0])
        quaternion = tf_trans.quaternion_about_axis(angle, (0, 0, 1))  # Rotation around z-axis only
        arrow_marker.pose.orientation.x = quaternion[0]
        arrow_marker.pose.orientation.y = quaternion[1]
        arrow_marker.pose.orientation.z = quaternion[2]
        arrow_marker.pose.orientation.w = quaternion[3]
        marker_array.markers.append(arrow_marker)

        # Create TEXT_VIEW_FACING marker to display speed
        speed_marker = Marker()
        speed_marker.header = Header(stamp=rospy.Time.now(), frame_id="map")
        speed_marker.ns = f"excavator_speed_{idx}"
        speed_marker.id = idx + 2000
        speed_marker.type = Marker.TEXT_VIEW_FACING
        speed_marker.action = Marker.ADD
        speed_marker.scale.z = 0.5  # Text height

        # Color: White for speed text
        speed_marker.color.r = 1.0
        speed_marker.color.g = 1.0
        speed_marker.color.b = 1.0
        speed_marker.color.a = 1.0

        # Position speed text above the latest position
        speed_marker.pose.position.x = latest_position[0]
        speed_marker.pose.position.y = latest_position[1]
        speed_marker.pose.position.z = latest_position[2] + 1.0  # Offset above the position
        speed_marker.text = f"Speed: {speed:.2f} m/s"
        speed_marker.pose.orientation.w = 1.0  # Identity quaternion
        marker_array.markers.append(speed_marker)

    # Publish all markers to RViz
    trail_pub.publish(marker_array)




def marker_callback(marker_array):
    global last_processed_time

    # Get the current timestamp
    current_time = marker_array.markers[0].header.stamp.secs

    # Check if we have processed a message in this second
    if last_processed_time == current_time:
        return  # Skip processing if this second has already been processed

    # Update the last processed time
    last_processed_time = current_time

    # Extract bounding boxes with label 4 only
    position_list = []
    for marker in marker_array.markers:
        position = [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]
        orientation_z = marker.pose.orientation.z
        scale = [marker.scale.x, marker.scale.y, marker.scale.z]

        # Get label based on color
        label = get_label_from_color(marker.color.r, marker.color.g, marker.color.b)

        # Process only if label is 4 (cyan)
        if label == 4:
            bounding_box = [label] + position + [orientation_z] + scale
            position_list.append(position)  # Add only position for tracking

    # Apply tracking algorithm
    closest_matching(position_list)

    # Visualize the tracked trails
    visualize_trails()


if __name__ == '__main__':
    rospy.Subscriber('/bounding_boxes', MarkerArray, marker_callback)
    rospy.spin()
