#!/bin/bash

# Ensure both scripts are executable
chmod +x run_with_slam.py
chmod +x excavator_tracking.py

# Run the first script in the background
echo "Starting run_with_slam.py..."
./run_with_slam.py &

# Get the process ID of the first script
PID1=$!

# Run the second script in the background
echo "Starting excavator_tracking.py..."
./excavator_tracking.py &

# Get the process ID of the second script
PID2=$!

# Wait for both scripts to finish
wait $PID1 $PID2

echo "Both scripts have completed."

