#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import PointCloud2  # Assuming positions are published as PointCloud2
import numpy as np
import math
import tf

# Initialize variables
tracked_excavators = []  # List to store tracked excavator trails by index
inactive_counter = []    # List to track inactive frames for each index
frame_count = 0          # Frame count for indexing purposes

# Parameters
DISTANCE_THRESHOLD = 10.0  # Matching distance threshold
MAX_INACTIVE_FRAMES = 5    # Max frames before deleting an inactive index

# Helper function to calculate Euclidean distance
def euclidean_distance(pos1, pos2):
    return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2)


# Callback to process incoming position data
def position_callback(msg):
    global frame_count, tracked_excavators, inactive_counter
    
    # Convert PointCloud2 message to position list (x_slam, y_slam, center_z)
    position_list = extract_positions_from_pointcloud(msg)  # Implement a helper to extract positions

    # Step 1: Initialize tracking on the first frame
    if frame_count == 0:
        for position in position_list:
            tracked_excavators.append([position])  # Each new position starts a new index
            inactive_counter.append(0)             # Initialize inactive counter for each index
    else:
        updated_indices = set()    # Set to keep track of updated indices
        new_positions = []         # List for positions needing a new index

        # Step 2: Update existing indices with closest matching positions
        for position in position_list:
            min_distance = DISTANCE_THRESHOLD
            closest_index = None

            # Find the closest existing position within the distance threshold
            for idx, trail in enumerate(tracked_excavators):
                last_position = trail[-1]
                distance = euclidean_distance(position, last_position)

                if distance < min_distance:
                    min_distance = distance
                    closest_index = idx

            # Update the index if a close match was found
            if closest_index is not None:
                tracked_excavators[closest_index].append(position)
                inactive_counter[closest_index] = 0       # Reset inactivity counter
                updated_indices.add(closest_index)
            else:
                new_positions.append(position)            # No match, needs a new index

        # Step 3: Create new indices for unmatched positions
        for position in new_positions:
            tracked_excavators.append([position])  # Start new trail
            inactive_counter.append(0)             # Initialize counter

        # Step 4: Update inactive counters and remove stale indices
        for idx in range(len(inactive_counter)):
            if idx not in updated_indices:
                inactive_counter[idx] += 1         # Increment inactive count
            if inactive_counter[idx] > MAX_INACTIVE_FRAMES:
                tracked_excavators[idx] = None     # Mark for deletion

        # Clean up inactive indices
        tracked_excavators = [t for t in tracked_excavators if t is not None]
        inactive_counter = [c for c in inactive_counter if c <= MAX_INACTIVE_FRAMES]

    # Step 5: Publish the updated trails to RViz
    publish_trails_to_rviz(tracked_excavators)

    frame_count += 1  # Increment the frame count for tracking


# Helper function to extract positions from PointCloud2 (assume x, y, z fields)
def extract_positions_from_pointcloud(msg):
    points = []
    for p in PointCloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append((p[0], p[1], p[2]))  # x_slam, y_slam, center_z
    return points

# Helper function to publish trails as MarkerArray for RViz visualization
def publish_trails_to_rviz(tracked_excavators):
    marker_array = MarkerArray()
    for idx, trail in enumerate(tracked_excavators):
        if trail is None:
            continue

        # Create a line strip marker for each excavator trail
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "excavator_trail"
        marker.id = idx
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.2  # Width of the line

        # Set color based on index for easy differentiation
        color = ColorRGBA(r=(idx * 37 % 255) / 255.0, g=(idx * 53 % 255) / 255.0, b=(idx * 97 % 255) / 255.0, a=1.0)
        marker.color = color

        # Populate points for the line strip from the excavator trail
        for pos in trail:
            point = Point()
            point.x, point.y, point.z = pos
            marker.points.append(point)

        marker_array.markers.append(marker)

    # Publish the marker array to the RViz topic
    trail_pub.publish(marker_array)

# Main function to initialize ROS node and subscribers
if __name__ == "__main__":
    rospy.init_node("excavator_tracker")
    trail_pub = rospy.Publisher("/excavator_trails", MarkerArray, queue_size=10)
    
    # Subscriber for the position topic
    rospy.Subscriber("/excavator_positions", PointCloud2, position_callback)

    rospy.spin()
