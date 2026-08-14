#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the current time for logging
timestamp=$(date +"%Y-%m-%d %H:%M:%S")

# Set environment variables
export DESTINATION_SERVER="http://192.168.100.1"  # Replace with your Destination Server's IP or hostname
export PORTS="8000,8001,8002,8003"  # Replace with the ports you want to forward

# Optional: Activate virtual environment
# Uncomment and set the correct path if using a virtual environment
# source /path/to/venv/bin/activate

# Start the FastAPI application using nohup and run it in the background
nohup python ./src/distributed_deployment_utils/forwarding_app/main.py &

# Capture the Process ID (PID) of the last background command
APP_PID=$!

# Output with timestamp
echo "$timestamp - Started forwarding server with PID $APP_PID!"
echo "$timestamp - The output is written to ./outputs/forwarding_app.log"
