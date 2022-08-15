# Flex Grasp
A ROS packages for manipulating vine tomato.

## 1. Install
> :warning: This has only been tested on Ubuntu 16.04 (ros kinetic) and Ubuntu 18.04 (ros melodic)!


### 1.1 Install ROS
Install ROS [Melodic](http://wiki.ros.org/melodic/Installation) (Ubuntu 18.04) or [Kinetic](http://wiki.ros.org/kinetic/Installation) (Ubuntu 16.04). Make sure sure that you have your environment properly setup, and that you have the most up to date packages:
```
rosdep update  # No sudo
sudo apt-get update
sudo apt-get dist-upgrade
```

### 1.2 Create A Workspace
You will need to have a ROS workspace setup:
```
mkdir -p ~/flexcraft_ws/src
cd ~/flexcraft_ws/
catkin_make
```

### 1.3 Download the source code
> :warning: Only use the master branch!

clone this repository
```
cd ~/flexcraft_ws/src
git clone https://github.com/JensZuurbier/vine_tomato_grasping.git
```

### 1.4 Basic Dependencies
Some more packages need to be installed manually.

#### Calibration
For calibration easy_handeye is used and aruco_ros is needed:
```
cd ~/flexcraft_ws/src
git clone https://github.com/IFL-CAMP/easy_handeye.git
git clone https://github.com/pal-robotics/aruco_ros
```

#### Intel Realsense
For Intel Realsense support realsense-ros and realsense2_description are used. Install these packages and dependencies as explained [here](https://github.com/IntelRealSense/realsense-ros). First define your ROS version, for example:
```
export ROS_VER=melodic
```
Than install both realsense2_camera and its dependents, including librealsense2 library:

```
sudo apt-get install ros-$ROS_VER-realsense2-camera
```
Finally install the realsense2_description:

```
sudo apt-get install ros-$ROS_VER-realsense2-description
```
It includes the 3D-models of the devices and is necessary for running launch files that include these models (i.e. rs_d435_camera_with_model.launch).

> :warning: In case you run into issues with the camera install the SDK as explained [here](https://www.intelrealsense.com/sdk-2/) and select Linux.

Launch RealSense

```
roslaunch realsense2_camera rs_rgbd.launch align_depth:=true depth_width:=1280 depth_height:=720 depth_fps:=30 color_width:=1280 color_height:=720 color_fps:=30
```

#### Python packages
Not all packages could be specified in the package.xml, and need to be installled manually:
```
pip install colormath
```

### 1.5 Remaining Dependencies
Install remaining dependencies:
```
cd ~/flexcraft_ws
rosdep install --from-paths src --ignore-src -r -y
```

## 2 Working with Panda Franka Emika

### How to start the impedance controller
```
roslaunch franka_human_friendly_controllers cartesian_variable_impedance_controller.launch robot_ip:=<robot_ip> load_gripper:=True
```
For example, <robot_ip>=172.16.0.2

### How to read the current position and orientation of the end-effector?
```
rostopic echo /cartesian_pose
```

### How to connect your PC to the network and read and send commands to the controller.

1. Connect your PC to the network
2. Create a new wired network and in IPv4 set Manual and put a new ip for your computer <pc_ip>=A.B.C.F where F is different from the <computer_ip> or the <robot_ip>. Netmask is the same 255.255.255.0. Save the network. 
3. Add this to your bash file (gedit ~/.bashrc): 
```
export ROS_MASTER_URI=http://<computer_ip>:11311 
export ROS_IP=<pc_ip> 
export ROS_HOSTNAME=<pc_ip>
```
(this can give issues when launching the camera for example, an alternative is to do these exports only in the needed terminal windows)

4. source /opt/ros/<ros_version>/setup.bash
5. Test the data_streaming with rostopic list

### How to control the gripper
```
rosrun franka_human_friendly_controllers franka_gripper_online
```
To change the width of the gripper you can publish rostopic pub /gripper_online msgs/Float32 "data: 0.01"
in the data you can specify your desired gripper width in meters.

## 3 Run Own Software
1. First launch the Panda launch files (see previouse chapter). To launch the environment and controls for real hardware run in your terminal:
    ```
    roslaunch flex_grasp detection_and_planning.launch
    ```
Note: this also launches the RealSense launch files
2. An rqt graphical user interface should pop up, sometimes in initializes incorrect, if this happens hit Ctrl + C, and retry

3. You have successfully initialized the controls, and the robot is ready to go.

Note: if you get warnings that the end effector is not able to reach its targets upon closing you may consider redefining the Closed interbotix_gripper group state as stated in `/interbotix_ros_arms/interbotix_moveit/config/srdf/px150.srdf.xacro`.

4. An rqt graphical user interface should pop up, sometimes in initializes incorrect, if this happens hit Ctrl + C, and retry
    <img src="doc/rqt.png" alt="rqt" width="800"/>

### Calibrate
First we need to calibrate the robot, this will generate a yaml file, which is stored and can be reused. Simply press `calibrate` in the GUI. The manipulator should move to several poses successively. It should print something as follows in the terminal:

```
[INFO] [1606135222.288133, 1573.747000]: State machine transitioning 'Idle':'calibrate'-->'CalibrateRobot'
[INFO] [1606135228.279287, 1579.025000]: Taking a sample...
[INFO] [1606135228.405969, 1579.128000]: Got a sample
[INFO] [1606135233.904765, 1583.933000]: Taking a sample...
[INFO] [1606135234.128548, 1584.135000]: Got a sample
...
[INFO] [1606135269.247164, 1615.083000]: Computing from 8 poses...
[INFO] [1606135269.295404, 1615.128000]: Computed calibration: effector_camera:
  translation:
    x: -0.028680958287
    y: 0.0123665209654
    z: 0.572588152978
  rotation:
    x: 0.174461585153
    y: 0.615597501442
    z: 0.158824096836
    w: 0.751916070974

```
The calibration results can be found in ~/.ros/easy_handeye/calibration_eye_on_base.yaml. Now you can stop the current porces by pressing `Ctrl + C`. Now run
```
roslaunch flex_grasp interbotix_enviroment.launch
```
And the previously generated calibration file will be loaded automatically.

### Command (Virtual) Robot
To activate an action, a command needs to be published on the `ROBOT_NAME/pipeline_command`. This can be done using the GUI:
- Home: command the robot to the initial pose
- Move Right: command the robot to move ..cm to the right
- Move Left: command the robot to move ..cm to the left
- Move Forwards: command the robot to move ..cm forwards
- Move Backwards: command the robot to move ..cm backward
- Move Upwards: command the robot to move ..cm upward
- Move Downwards: command the robot to move ..cm downward
- Approach truss: command the robot to approach the truss detected by truss detection model
- Grasp: command to the robot to perform grasp with determined grasp point
- Place: command to the robot to place object at target location
- Open: command the end effector to open
- Close: command the end effector to close

- Detect Truss: command to truss detection model to detect the tomato trusses
- Detect Grasp Point: command to determine grasp point on a tomato truss

- Save Pose: command to the robot to save the current pose
- Go to Saved Pose: command to the robot to move to the saved pose

- Calibrate: determine the pose between the robot base and camera
- Experiment: Repeatedly execute Detect Truss, Approach Truss, Detect Grasp Point, Grasp, Place, Home (easy for conducting experiments)

## 4 Supported hardware

Manipulator:

- **Franka Emika Panda manipulator**

Carmera:

- **Intel RealSense D435**

## 5 Contents

### Nodes

#### Custom software
- `calibrate`: generates the calibration poses, sends them to the move robot node and computing calibration
- `move_robot`: takes commands from other nodes and moves the manipulater according to these commands
- `plan_movement`: takes commands from other nodes to plan a movement for the robot, publishes to a topic the robot reads from and moves to
- `object_detection`: identifies the tomato trusses and determines a valid grasp location
- `pipeline`: contains the statemachine, commands all other nodes
- `my_rqt_dashboard`: dashboard to press buttons to move the robot, detect grasp point etc.

#### RealSense
- `realsense2_camera`: publishes RGB and Depth data to topics

### Classes
- `communication`: this class is used by many nodes to send commands to other nodes and wait for the result

### Messages
- `ImageProcessingSettings`
- `Peduncle`
- `Tomato`
- `Truss`

### Enums
To store the state of different parts of the system, enums are used. These are defined in the messa files.
- `DynamixelErrorCodes`
- `FlexGraspCommandCodes`
- `FlexGraspErrorCodes`

### Info

All nodes run in the `robot_name` namespace to allow for multiple robots present

## 6 Trouble shooting

### libcurl: (51) SSL: no alternative certificate subject name matches target host name ‘api.ignitionfuel.org’
https://varhowto.com/how-to-fix-libcurl-51-ssl-no-alternative-certificate-subject-name-matches-target-host-name-api-ignitionfuel-org-gazebo-ubuntu-ros-melodic/

### RLException: unused args [start_sampling_gui] for include of [/home/taeke/flexcraft_ws/src/easy_handeye/easy_handeye/launch/calibrate.launch]
In older version of `calibrate.launch` the variable `start_sampling_gui` was called `start_rqt`. Thus to fix this command either update easy_handeye, of if this is not desired change `start_sampling_gui` in taeke_msc/flex_grasp/launch to `start_rqt`.

### Calibration node does not initialize, but gets stuck
If calibrate gets stuck at:
```
[INFO] [1610553811.946348, 0.001000]: Loading parameters for calibration /px150/calibration_eye_on_base/ from the parameters server
```
check this issue: https://github.com/IFL-CAMP/easy_handeye/issues/77
