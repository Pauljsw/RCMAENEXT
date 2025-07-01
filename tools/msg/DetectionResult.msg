# DetectionResult.msg
int32 label            # Label ID for the detected object
float32 x_slam         # SLAM-adjusted X-coordinate of the bounding box center
float32 y_slam         # SLAM-adjusted Y-coordinate of the bounding box center
float32 center_z       # Z-coordinate of the bounding box center
float32 width          # Width of the bounding box
float32 length         # Length of the bounding box
float32 height         # Height of the bounding box
float32[4] orientation # Quaternion orientation [x, y, z, w]

