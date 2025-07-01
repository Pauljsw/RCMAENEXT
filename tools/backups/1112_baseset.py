#!/usr/bin/env python
import rospy
from visualization_msgs.msg import MarkerArray
import numpy as np

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
    

def marker_callback(marker_array):
    global last_processed_time

    # Get the current timestamp
    current_time = marker_array.markers[0].header.stamp.secs

    # Check if we have processed a message in this second
    if last_processed_time == current_time:
        return  # Skip processing if this second has already been processed

    # Update the last processed time
    last_processed_time = current_time

    bounding_boxes = []
    for marker in marker_array.markers:
        position = [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]
        orientation_z = marker.pose.orientation.z
        scale = [marker.scale.x, marker.scale.y, marker.scale.z]
        
        # Get label based on color
        label = get_label_from_color(marker.color.r, marker.color.g, marker.color.b)
        
        # Flatten to a single list
        bounding_box = [label] + position + [orientation_z] + scale 
        bounding_boxes.append(bounding_box)
    
    # Convert to numpy array
    bounding_boxes_np = np.round(np.array(bounding_boxes), 1)
    print(bounding_boxes_np)
    

if __name__ == '__main__':
    rospy.init_node('bounding_box_listener', anonymous=True)
    rospy.Subscriber('/bounding_boxes', MarkerArray, marker_callback)
    rospy.spin()
