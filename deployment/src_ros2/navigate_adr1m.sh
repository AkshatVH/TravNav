#!/bin/bash

# Create a new tmux session
session_name="vint_panther_$(date +%s)"
base_folder="/home/szilard/projects/visualnav-transformer/deployment/src_ros2"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves

tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -h -p 100 # split it

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "cd $base_folder" Enter
tmux send-keys "ros2 launch visnav_usbcam_ros2.launch.py" Enter

# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 1
tmux send-keys "cd $base_folder" Enter
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python navigate.py $1" Enter

tmux select-pane -t 2
tmux send-keys "cd $base_folder" Enter
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "ros2 bag record -o visnav_adr1m_$2_$(date +"%Y-%m-%d"-%H-%M-%S) /ad_r1m_13/cmd_vel /ad_r1m_13/imu /ad_r1m_13/odometry/filtered /usb_cam/image_raw/compressed" # change topic if necessary

# Run the teleop.py script in the second pane
tmux select-pane -t 3
tmux send-keys "ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/ad_r1m_13/cmd_vel" Enter

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 4
tmux send-keys "cd $base_folder" Enter
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python pd_controller.py" Enter

tmux select-pane -t 2

# Attach to the tmux session
tmux -2 attach-session -t $session_name


#### To kill it from another terminal run: tmux kill-server