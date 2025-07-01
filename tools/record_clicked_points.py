#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PointStamped

# File to save clicked points
OUTPUT_FILE = "clicked_points.txt"

# Callback function to save the clicked point
def save_clicked_point(msg):
    with open(OUTPUT_FILE, "a") as file:
        point = msg.point
        file.write(f"{point.x:.3f}, {point.y:.3f}, {point.z:.3f}\n")
    rospy.loginfo(f"Saved point: ({point.x:.3f}, {point.y:.3f}, {point.z:.3f})")

def record_clicked_points():
    rospy.init_node("record_clicked_points", anonymous=True)
    rospy.Subscriber("/clicked_point", PointStamped, save_clicked_point)
    rospy.loginfo(f"Recording points to {OUTPUT_FILE}. Click in RViz to add points.")
    rospy.spin()

if __name__ == "__main__":
    record_clicked_points()

