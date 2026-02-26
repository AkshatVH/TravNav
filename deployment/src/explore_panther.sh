#!/bin/bash

# Create a new tmux session
session_name="vint_panther_$(date +%s)"
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

tmux selectp -t 1    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "useros1" Enter
tmux send-keys "roslaunch visnav_panther.launch" Enter

# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 1
tmux send-keys "useros1" Enter
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python explore.py $1" Enter

tmux select-pane -t 2
tmux send-keys "useros2" Enter
tmux send-keys "cd ../odometries" Enter
tmux send-keys "ros2 bag record -o odom_$2_$(date +"%Y-%m-%d"-%H-%M-%S) /panther/battery /panther/cmd_vel /panther/imu_broadcaster/imu /panther/odometry/filtered /panther/odometry/wheels /panther/joint_states"

# # relay rgb topic to usb_cam
# tmux select-pane -t 2
# tmux send-keys "useros1" Enter
# tmux send-keys "rosrun topic_tools relay /rgb /usb_cam/image_raw" Enter

# Run the ros1_bridge on the last panel
tmux select-pane -t 4
tmux send-keys "useros1" Enter
tmux send-keys "useros2" Enter
tmux send-keys "ros2 run ros1_bridge dynamic_bridge --bridge-all-topics" Enter

# Run the teleop.py script in the third pane
tmux select-pane -t 3
tmux send-keys "useros1" Enter
# tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "rosrun teleop_twist_keyboard teleop_twist_keyboard.py /cmd_vel:=/panther/cmd_vel" Enter

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 5
tmux send-keys "useros1" Enter
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python pd_controller.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name


#### To kill it from another terminal run: tmux kill-server