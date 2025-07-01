#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point
import numpy as np
import math
import time

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
distance_threshold = 5  # Threshold for matching

# Initialize ROS node
rospy.init_node('excavator_tracker', anonymous=True)

# Visualization publisher
trail_pub = rospy.Publisher('/excavator_trails', MarkerArray, queue_size=10)

def euclidean_distance(pos1, pos2):
    """ Calculate the Euclidean distance between two 3D points. """
    return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2)

def closest_matching(position_list):
    global excavator_index, inactive_counter
    print("Excavator index", excavator_index)
    updated_indices = set() # Set to track which indices have been updated
    new_positions = []  # List to store positions that need a new index
    # Step 1: Update existing indices with closest matching positions
    for position in position_list:
        min_distance = distance_threshold
        closest_index = None    
        for idx, tracked_positions in enumerate(excavator_index):
            last_position = tracked_positions[-1]
            distance = euclidean_distance(position, last_position)
            if distance < min_distance:
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

def visualize_trails():
    """ Publishes trails of tracked excavators in RViz. """
    marker_array = MarkerArray()

    for idx, positions in enumerate(excavator_index):
        # Ensure there are at least two points in the trail
        if len(positions) < 2:
            continue  # Skip this trail if it has fewer than two points

        marker = Marker()
        marker.header = Header(stamp=rospy.Time.now(), frame_id="map")
        marker.ns = f"excavator_trail_{idx}"
        marker.id = idx
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # Line width

        # Color: Cyan for excavator
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Set orientation to identity quaternion
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set the points for the trail
        marker.points = [Point(x=pos[0], y=pos[1], z=pos[2]) for pos in positions]
        marker_array.markers.append(marker)

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
