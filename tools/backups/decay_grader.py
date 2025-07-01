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
max_inactive_frames = 100  # Max frames before removing stale indices

min_dist_threshold = 1 # Threshold for matching

# Initialize ROS node
rospy.init_node('excavator_tracker', anonymous=True)

# Visualization publisher
trail_pub = rospy.Publisher('/grader_trails', MarkerArray, queue_size=10)

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
    """ Display the current status of tracked excavators in a table and save trajectories to a text file. """
    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = ["Index", "Trail Length", "Latest X", "Latest Y", "Latest Z", "Inactive Frames", "Speed", "Direction"]

    # Open a text file to save trajectories
    with open("tracking_trajectories.txt", "w") as file:
        file.write("Tracking Trajectories\n")
        file.write("====================\n\n")

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

            # Save trajectories to the file if trail length > 10
            if trail_length > 10:
                for position in positions:
                    file.write(f"{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f} {idx}\n")

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
            speed, direction = calculate_average_speed(tracked_positions)
            distance_threshold = max(speed*2, min_dist_threshold) * (1 + inactive_counter[idx])  # Adjust threshold
            # Calculate predicted position using speed and direction
            predicted_position = (
                last_position[0] + direction[0] * inactive_counter[idx],
                last_position[1] + direction[1] * inactive_counter[idx],
                last_position[2] + direction[2] * inactive_counter[idx]
            )            
            # Calculate distances to the current and predicted positions
            distance_to_last = euclidean_distance(position, last_position)
            distance_to_predicted = euclidean_distance(position, predicted_position)
            # Use the smaller of the two distances for matching
            distance = min(distance_to_last, distance_to_predicted)

            # Update closest index if within the threshold and is the smallest distance
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
        valid_new_index = True
        for idx, tracked_positions in enumerate(excavator_index):
            last_position = tracked_positions[-1]
            speed, direction = calculate_average_speed(tracked_positions)

            # Calculate predicted position using speed, direction, and inactive counter
            predicted_position = (
                last_position[0] + direction[0] * inactive_counter[idx],
                last_position[1] + direction[1] * inactive_counter[idx],
                last_position[2] + direction[2] * inactive_counter[idx]
            )

            # Ensure the new position is at least 5m away from existing indices and predictions
            if euclidean_distance(position, last_position) < 5.0 or euclidean_distance(position, predicted_position) < 5.0:
                valid_new_index = False
                break

        if valid_new_index:
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
    """ Publishes all bounding boxes contributing to trajectories with trail_length >= 5 in RViz. """
    marker_array = MarkerArray()

    for idx, positions in enumerate(excavator_index):
        if len(positions) < 3:
            continue  # Skip if trail length is less than 5

        # Iterate over all positions in the trajectory
        for pos_idx, position in enumerate(positions):
            # Create a bounding box marker for each position
            bbox_marker = Marker()
            bbox_marker.header = Header(stamp=rospy.Time.now(), frame_id="map")
            bbox_marker.ns = f"excavator_bbox_{idx}"
            bbox_marker.id = idx * 100 + pos_idx  # Unique ID per marker
            bbox_marker.type = Marker.CUBE  # Represents a bounding box
            bbox_marker.action = Marker.ADD

            # Set bounding box properties (size and orientation)
            bbox_marker.scale.x = position[3]  # Size X (scale.x from subscribed topic)
            bbox_marker.scale.y = position[4]  # Size Y (scale.y from subscribed topic)
            bbox_marker.scale.z = position[5]  # Size Z (scale.z from subscribed topic)
            bbox_marker.pose.position.x = position[0]  # X position
            bbox_marker.pose.position.y = position[1]  # Y position
            bbox_marker.pose.position.z = position[2] + bbox_marker.scale.z / 2  # Z position (centered)

            bbox_marker.pose.orientation.z = position[6]  # Orientation Z
            bbox_marker.pose.orientation.w = 1.0  # Identity quaternion for simplicity

            # Set color for bounding boxes (cyan for excavators)
            bbox_marker.color.r = 0.0
            bbox_marker.color.g = 1.0
            bbox_marker.color.b = 1.0
            bbox_marker.color.a = 0.8  # Transparency for better visualization

            # Add the marker to the array
            marker_array.markers.append(bbox_marker)

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
        position = [
            marker.pose.position.x,  # x
            marker.pose.position.y,  # y
            marker.pose.position.z,  # z
            marker.scale.x,          # scale.x
            marker.scale.y,          # scale.y
            marker.scale.z,          # scale.z
            marker.pose.orientation.z  # orientation.z
        ]
        
        
        #position_list.append(position)

        # Get label based on color
        label = get_label_from_color(marker.color.r, marker.color.g, marker.color.b)

        # Process only if label is 4 (cyan)
        if label == 4:
            position_list.append(position)

    # Apply tracking algorithm
    closest_matching(position_list)

    # Visualize the tracked bounding boxes
    visualize_trails()


if __name__ == '__main__':
    rospy.Subscriber('/bounding_boxes', MarkerArray, marker_callback)
    rospy.spin()
