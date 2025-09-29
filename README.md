This is a fork of [https://github.com/robodhruv/visualnav-transformer](https://github.com/robodhruv/visualnav-transformer). This repository contains additional config files and runners so that the deployment part can be run on a Jetson Xavier AGX, Pioneer3-AT and Husarion Panther robots, and an Up camera. Modified by Szilard Molnar from [ROCON](http://rocon.utcluj.ro/) team UTCN.

# General Navigation Models: GNM, ViNT and NoMaD

**Original Contributors**: Dhruv Shah, Ajay Sridhar, Nitish Dashora, Catherine Glossop, Kyle Stachowicz, Arjun Bhorkar, Kevin Black, Noriaki Hirose, Sergey Levine

_Berkeley AI Research_

[Project Page](https://general-navigation-models.github.io) | [Citing](https://github.com/robodhruv/visualnav-transformer#citing) | [Pre-Trained Models](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)

---

For more information on training, visit the original repo.

General Navigation Models are general-purpose goal-conditioned visual navigation policies trained on diverse, cross-embodiment training data, and can control many different robots in zero-shot. They can also be efficiently fine-tuned, or adapted, to new robots and downstream tasks. Our family of models is described in the following research papers (and growing):
1. [GNM: A General Navigation Model to Drive Any Robot](https://sites.google.com/view/drive-any-robot) (_October 2022_, presented at ICRA 2023)
2. [ViNT: A Foundation Model for Visual Navigation](https://general-navigation-models.github.io/vint/index.html) (_June 2023_, presented at CoRL 2023)
3. [NoMaD: Goal Masking Diffusion Policies for Navigation and Exploration](https://general-navigation-models.github.io/nomad/index.html) (_October 2023_)


## Deployment
This subfolder contains code to load a pre-trained ViNT and deploy it on the Pioneer3-AT robot with a NVIDIA Jetson Xavier AGX.

### P3-AT Setup

This software was tested on a P3-AT running Ubuntu 20.04.


#### Software Installation (in this order)
1. ROS: [ros-noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)
2. ROS packages: 
    ```bash
    sudo apt-get install ros-noetic-usb-cam ros-noetic-joy
    ```
3. [kobuki](http://wiki.ros.org/kobuki/Tutorials/Installation) - might not be needed for the P3-AT robot (can be installed just to be sure)
    ```bash
    sudo apt-get install ros-noetic-kobuki*
    ```

    Instead install rosaria:
   ```bash
   sudo apt install aria2 libaria-dev
   ```
   Download it to a catkin_ws from [https://github.com/amor-ros-pkg/rosaria](https://github.com/amor-ros-pkg/rosaria), also check [http://wiki.ros.org/ROSARIA/Tutorials/How%20to%20use%20ROSARIA](http://wiki.ros.org/ROSARIA/Tutorials/How%20to%20use%20ROSARIA).
   The current Runner files are especially include the install location of `~/catkin_ws`, change it if you have installed it to another location.
4. Conda 
    - Install anaconda/[miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux)/etc. for managing environments
    - Make conda env with environment.yml (run this inside the `visualnav-transformer/` directory)
        ```bash
        conda env create -f deployment/deployment_environment.yaml
        ```
    - Source env 
        ```bash
        conda activate vint_deployment
        ```
    - (Optional) add to `~/.bashrc`: 
        ```bash
        echo “conda activate vint_deployment” >> ~/.bashrc 
        ```
5. Install the `vint_train` packages (run this inside the `visualnav-transformer/` directory):
    ```bash
    pip install -e train/
    ```
6. Install the `diffusion_policy` package from this [repo](https://github.com/real-stanford/diffusion_policy):
    ```bash
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```
7. Install [tmux](https://github.com/tmux/tmux/wiki/Installing) if not present.
    Many of the bash scripts rely on tmux to launch multiple screens with different commands. This will be useful for debugging because you can see the output of each screen.

   Be familiar with a few important commands: Navigate between screens: `Ctrl+b` then `ArrowKeys`, `Ctrl+d` closes the screens. If you want to list all the tmux sessions use: `tmux list-sessions`. If you want to kill all the tmux session use: `tmux kill-server`

8. >[!Note] Using the Husarion Panther:
    You have to have ROS2 installed alongside ROS1 (recommended: Ros Noetic + ROS2 Foxy)
    For ease of use I added to aliasess into the `nano ~/.bashrc` file: 
    ```
    alias useros1="source /opt/ros/noetic/setup.bash"
    alias useros2="source /opt/ros/foxy/setup.bash"
    ```
    After you have both ROS versions installed, you need to install an additional package: 
    ```
    sudo apt install ros-foxy-ros1-bridge
    ```
    This installs the [ros1_bridge](https://github.com/ros2/ros1_bridge), which makes it possible for the ROS1 code to control the ROS2 Panther.

#### Hardware Requirements
- RosARIA for P3-AT
- Up camera (original repo recommends a wide-angle RGB camera: [Example](https://www.amazon.com/ELP-170degree-Fisheye-640x480-Resolution/dp/B00VTHD17W). The `vint_locobot.launch` file uses camera parameters that work with cameras like the ELP fisheye wide angle, feel free to modify to your own. Adjust the camera parameters in `visualnav-transformer/deployment/config/camera.yaml` your camera accordingly (used for visualization)).
- Teleop-twist keyboard (original repo uses [Joystick](https://www.amazon.com/Logitech-Wireless-Nano-Receiver-Controller-Vibration/dp/B0041RR0TW)/[keyboard teleop](http://wiki.ros.org/teleop_twist_keyboard) that works with Linux. Add the index mapping for the _deadman_switch_ on the joystick to the `visualnav-transformer/deployment/config/joystick.yaml`. You can find the mapping from buttons to indices for common joysticks in the [wiki](https://wiki.ros.org/joy).)


### Loading the model weights

Save the model weights *.pth file in `visualnav-transformer/deployment/model_weights` folder. Our model's weights are in [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing).

### Collecting a Topological Map

_Make sure to run these scripts inside the `visualnav-transformer/deployment/src/` directory._


This section discusses a simple way to create a topological map of the target environment for deployment. For simplicity, we will use the robot in “path-following” mode, i.e., given a single trajectory in an environment, the task is to follow the same trajectory to the goal. The environment may have new/dynamic obstacles, lighting variations etc.

#### Record the rosbag:

>[!Warning] Do not forget to modify the `vel_teleop_topic` and `vel_navi_topic` in your `deployment/config/robot.yaml` file, both of them to: `vel_teleop_topic: /RosAria/cmd_vel` for the P3-AT robot and  `vel_teleop_topic: /panther/cmd_vel` for the Husarion Panther robot.

##### P3-AT
```bash
./record_bag_p3.sh <bag_name>
```
##### Husarion Panther
```bash
./record_bag_panther.sh <bag_name>
```

Run this command to teleoperate the robot with the joystick and camera. This command opens up three windows 
1. `roslaunch vint_p3.launch` or `roslaunch vint_panther.launch`: This launch file opens the `usb_cam` node for the camera, and nodes for the robot’s mobile base for P3-AT.
2. `rosrun teleop_twist_keyboard teleop_twist_keyboard.py /cmd_vel:=/RosAria/cmd_vel`: This node teleoperates the robot’s base using keyboard for P3_AT. OR `rosrun teleop_twist_keyboard teleop_twist_keyboard.py /cmd_vel:=/panther/cmd_vel`: This node teleoperates the robot’s base using keyboard.
3. `rosbag record /usb_cam/image_raw -o <bag_name>`: This command isn’t run immediately (you have to click Enter). It will be run in the visualnav-transformer/deployment/topomaps/bags directory, where we recommend you store your rosbags.
4. Using the `navigate_panther.sh` additionally creates: `ros2 run ros1_bridge dynamic_bridge`: This node passes the `cmd_vel` commands from ROS1 to ROS2

Once you are ready to record the bag, run the `rosbag record` script and teleoperate the robot on the map you want the robot to follow. When you are finished with recording the path, kill the `rosbag record` command, and then kill the tmux session.

#### Make the topological map: 
```bash
./create_topomap.sh <topomap_name> <bag_filename>
```

This command opens up 3 windows:
1. `roscore`
2. `python create_topomap.py —dt 1 —dir <topomap_dir>`: This command creates a directory in `/visualnav-transformer/deployment/topomaps/images` and saves an image as a node in the map every second the bag is played.
3. `rosbag play -r 1.5 <bag_filename>`: This command plays the rosbag at x5 speed, so the python script is actually recording nodes 1.5 seconds apart. The `<bag_filename>` should be the entire bag name with the .bag extension. You can change this value in the `make_topomap.sh` file. The command does not run until you hit Enter, which you should only do once the python script gives its waiting message. Once you play the bag, move to the screen where the python script is running so you can kill it when the rosbag stops playing.

When the bag stops playing, kill the tmux session.


### Running the model 
#### Navigation
_Make sure to run this script inside the `visualnav-transformer/deployment/src/` directory._

>[!Warning] Do not forget to modify the `vel_teleop_topic` and `vel_navi_topic` in your `deployment/config/robot.yaml` file, both of them to: `vel_teleop_topic: /RosAria/cmd_vel` for the P3-AT robot and  `vel_teleop_topic: /panther/cmd_vel` for the Husarion Panther robot.

##### P3-AT
```bash
./navigate_p3.sh “--model <model_name> --dir <topomap_dir>”
```

##### Husarion Panther
```bash
./navigate_panther.sh “--model <model_name> --dir <topomap_dir>”
```

To deploy one of the models from the published results, we are releasing model checkpoints that you can download from [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing).


The `<model_name>` is the name of the model in the `visualnav-transformer/deployment/config/models.yaml` file. In this file, you specify these parameters of the model for each model (defaults used):
- `config_path` (str): path of the *.yaml file in `visualnav-transformer/train/config/` used to train the model
- `ckpt_path` (str): path of the *.pth file in `visualnav-transformer/deployment/model_weights/`


Make sure these configurations match what you used to train the model. The configurations for the models we provided the weights for are provided in yaml file for your reference.

The `<topomap_dir>` is the name of the directory in `visualnav-transformer/deployment/topomaps/images` that has the images corresponding to the nodes in the topological map. The images are ordered by name from 0 to N.

This command opens up 4 windows:

1. `roslaunch vint_p3.launch` or `roslaunch vint_panther.launch`: This launch file opens the usb_cam node for the camera, and several nodes for the robot’s mobile base for P3-AT.
2. `python navigate.py --model <model_name> -—dir <topomap_dir>`: This python script starts a node that reads in image observations from the `/usb_cam/image_raw` topic, inputs the observations and the map into the model, and publishes actions to the `/waypoint` topic.
3. `rosrun teleop_twist_keyboard teleop_twist_keyboard.py /cmd_vel:=/RosAria/cmd_vel`: This node teleoperates the robot’s base using keyboard for P3_AT. OR `rosrun teleop_twist_keyboard teleop_twist_keyboard.py /cmd_vel:=/panther/cmd_vel`: This node teleoperates the robot’s base using keyboard.
4. `python pd_controller.py`: This python script starts a node that reads messages from the `/waypoint` topic (waypoints from the model) and outputs velocities to navigate the robot’s base.
5. Using the `navigate_panther.sh` additionally creates: `ros2 run ros1_bridge dynamic_bridge`: This node passes the `cmd_vel` commands from ROS1 to ROS2

When the robot is finishing navigating, kill the `pd_controller.py` script, and then kill the tmux session.

#### Exploration
_Make sure to run this script inside the `visualnav-transformer/deployment/src/` directory._

##### P3_AT
```bash
./exploration_p3.sh “--model <model_name>”
```

##### Husarion Panther
```bash
./exploration_panther.sh “--model <model_name>”
```

To deploy one of the models from the published results, we are releasing model checkpoints that you can download from [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing).


The `<model_name>` is the name of the model in the `visualnav-transformer/deployment/config/models.yaml` file (note that only NoMaD works for exploration). In this file, you specify these parameters of the model for each model (defaults used):
- `config_path` (str): path of the *.yaml file in `visualnav-transformer/train/config/` used to train the model
- `ckpt_path` (str): path of the *.pth file in `visualnav-transformer/deployment/model_weights/`


Make sure these configurations match what you used to train the model. The configurations for the models we provided the weights for are provided in yaml file for your reference.

The `<topomap_dir>` is the name of the directory in `visualnav-transformer/deployment/topomaps/images` that has the images corresponding to the nodes in the topological map. The images are ordered by name from 0 to N.

This command opens up 4 windows:

1. `roslaunch vint_p3.launch` or `roslaunch vint_panther.launch`: This launch file opens the usb_cam node for the camera, the joy node for the joystick, and several nodes for the robot’s mobile base for P3-AT.
2. `python explore.py --model <model_name>`: This python script starts a node that reads in image observations from the `/usb_cam/image_raw` topic, inputs the observations and the map into the model, and publishes exploration actions to the `/waypoint` topic.
3. `rosrun teleop_twist_keyboard teleop_twist_keyboard.py /cmd_vel:=/RosAria/cmd_vel`: This node teleoperates the robot’s base using keyboard for P3_AT. OR `rosrun teleop_twist_keyboard teleop_twist_keyboard.py /cmd_vel:=/panther/cmd_vel`: This node teleoperates the robot’s base using keyboard.
4. `python pd_controller.py`: This python script starts a node that reads messages from the `/waypoint` topic (waypoints from the model) and outputs velocities to navigate the robot’s base.
5. Using the `navigate_panther.sh` additionally creates: `ros2 run ros1_bridge dynamic_bridge`: This node passes the `cmd_vel` commands from ROS1 to ROS2

When the robot is finishing navigating, kill the `pd_controller.py` script, and then kill the tmux session.


### Adapting this code to different robots

We hope that this codebase is general enough to allow you to deploy it to your favorite ROS-based robots. You can change the robot configuration parameters in `visualnav-transformer/deployment/config/robot.yaml`, like the max angular and linear velocities of the robot and the topics to publish to teleop and control the robot. Please feel free to create a Github Issue or reach out to the authors at shah@cs.berkeley.edu.


## Citing
```
@inproceedings{shah2022gnm,
  author    = {Dhruv Shah and Ajay Sridhar and Arjun Bhorkar and Noriaki Hirose and Sergey Levine},
  title     = {{GNM: A General Navigation Model to Drive Any Robot}},
  booktitle = {International Conference on Robotics and Automation (ICRA)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2210.03370}
}

@inproceedings{shah2023vint,
  title     = {Vi{NT}: A Foundation Model for Visual Navigation},
  author    = {Dhruv Shah and Ajay Sridhar and Nitish Dashora and Kyle Stachowicz and Kevin Black and Noriaki Hirose and Sergey Levine},
  booktitle = {7th Annual Conference on Robot Learning},
  year      = {2023},
  url       = {https://arxiv.org/abs/2306.14846}
}

@article{sridhar2023nomad,
  author  = {Ajay Sridhar and Dhruv Shah and Catherine Glossop and Sergey Levine},
  title   = {{NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration}},
  journal = {arXiv pre-print},
  year    = {2023},
  url     = {https://arxiv.org/abs/2310.xxxx}
}
```
