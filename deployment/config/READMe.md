# Run it with ROS2

## You can use it normally (TODO tests).

### DDS

We will use tmux to open the softwares, you can close it at once, by running this command: 

```bash
tmux kill-server
```

#### Record bag 

```bash

```

### Zenoh

The RGB camera feed will be extremely slow, however the ToF camera  will work at least.

#### Terminal 1 

```bash
ros2 daemon stop

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

export ZENOH_CONFIG_OVERRIDE="connect/endpoints=[\"tcp/ad-r1m-13.local:7447\"];mode=\"client\""

ros2 run rmw_zenoh_cpp rmw_zenohd
```

#### Terminal 2
OAK D-Lite for 1080P (4K@3840x2160 is too much):

```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

export ZENOH_CONFIG_OVERRIDE="connect/endpoints=[\"tcp/ad-r1m-13.local:7447\",\"tcp/szilard-xpsu.local:7447\"];scouting/multicast/enabled=true;mode=\"client\""

## Oak D-Lite
ros2 run depthai_ros_driver camera_node --ros-args -p camera.i_pipeline_type:=RGB -p camera.i_nn_type:=none -p pipeline_gen.i_enable_imu:=True -p rgb.r_set_man_focus:=True -p rgb.i_fps:=35.0 -p publish_tf:=True -p rgb.i_resolution:=1080P -p rgb.i_width:=1920 -p rgb.i_height:=1080

## Up FHD camera
cd /home/szilard/projects/visualnav-transformer/deployment/src_ros2

ros2 launch visnav_usbcam_ros2.launch.py
```