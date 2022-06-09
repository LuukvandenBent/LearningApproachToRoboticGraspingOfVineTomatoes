#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import time
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

import tf2_ros
import tf2_geometry_msgs

from flex_grasp.msg import FlexGraspErrorCodes

class Planner():

    def __init__(self, NODE_NAME, playback=False):
        self.node_name = NODE_NAME
        self.playback = playback

        self.r = rospy.Rate(10)

        self.K_pos = 600
        self.K_ori = 30
        self.K_ns = 10

        self.current_pos = None
        self.current_ori = None

        self.delta_pos = np.array([0.0, 0.0, 0.0])
        self.delta_ori = np.array([0.0, 0.0, 0.0, 0.0])

        self.translation = None
        self.rotation = None
        self.to_frame = None

        self.robot_goal_pos = None
        self.robot_goal_ori = None

        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)

        self.panda_pos_sub = rospy.Subscriber("/cartesian_pose", PoseStamped, self.ee_pos_callback)
        self.truss_pos_sub = rospy.Subscriber("/truss_pose", PoseStamped, self.truss_pos_callback)
        self.trans_sub = rospy.Subscriber("/transform_matrix", Float32MultiArray, self.trans_matrix_callback)
        self.robot_goal_pos_sub = rospy.Subscriber('/panda_robot_pose', PoseStamped, self.robot_goal_pos_callback)

        self.goal_pub = rospy.Publisher('/equilibrium_pose', PoseStamped, queue_size=0)
        self.stiff_pub = rospy.Publisher('/stiffness', Float32MultiArray, queue_size=0)
        
    def ee_pos_callback(self, data):
        self.current_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.current_ori = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])

    def truss_pos_callback(self, data):
        # self.delta_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        
        self.delta_ori = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])

    def trans_matrix_callback(self,data):
        self.translation = np.array([data.translation.x, data.translation.y, data.translation.z])
        self.rotation = np.array([data.rotation.x, data.rotation.y,
                                data.rotation.z, data.rotation.w])
        self.to_frame = data.frame_id

    def robot_goal_pos_callback(self,data):
        self.robot_goal_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.robot_goal_ori = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])

        rospy.logdebug(f'goal_pos: {self.robot_goal_pos}')
        rospy.logdebug(f'goal_ori: {self.robot_goal_ori}')

    def plan_movement(self, movement=None):
        
        goal = self.find_goal_pose(movement=movement)
        
        # if movement != 'calibration_movement' and movement != 'home':
        #     transformed_pose = transformed_pose(goal,self.to_frame)
        
        is_safe = self.check_movement(goal)
        
        if is_safe: 
            result = self.go_to_pose(goal)
            
            if result == 'succes':
                return FlexGraspErrorCodes.SUCCESS
            else:
                return FlexGraspErrorCodes.FAILURE
        else:
            rospy.logwarn(f'[{self.node_name}] Movement not safe')
            rospy.logdebug(f'Goal pose: {goal}')
            return FlexGraspErrorCodes.FAILURE

    def find_goal_pose(self, movement=None):
        if movement == 'calibration_movement':
            goal_pos = self.robot_goal_pos
            goal_ori = self.robot_goal_ori

        elif movement == 'home':
            goal_pos = np.array([-0.4, 0.5, 0.4])
            goal_ori = np.array([0.70710678118, 0.70710678118, 0.0, 0.0])

        else:    
            if movement == 'approach':
                self.delta_pos[2] = 0

            elif movement == 'grasp':
                self.delta_pos[2] = -0.05

            elif movement == 'move_right':
                self.delta_pos = np.array([0.05,0.0,0.0])

            elif movement == 'move_left':
                self.delta_pos = np.array([-0.05,0.0,0.0])
            
            elif movement == 'move_forwards':
                self.delta_pos = np.array([0.0,0.05,0.0])

            elif movement == 'move_backwards':
                self.delta_pos = np.array([0.0,-0.05,0.0])

            elif movement == 'move_upwards':
                self.delta_pos = np.array([0.0,0.0,0.05])

            elif movement == 'move_downwards':
                self.delta_pos = np.array([0.0,0.0,-0.05])

            goal_pos = np.array([self.current_pos[0] + self.delta_pos[0], self.current_pos[1] + self.delta_pos[1], self.current_pos[2] + self.delta_pos[2]])
            goal_ori = self.current_ori

        goal = PoseStamped()
        
        goal.pose.position.x = goal_pos[0]
        goal.pose.position.y = goal_pos[1]
        goal.pose.position.z = goal_pos[2]

        goal.pose.orientation.x = goal_ori[0]
        goal.pose.orientation.y = goal_ori[1]
        goal.pose.orientation.z = goal_ori[2]
        goal.pose.orientation.w = goal_ori[3]

        return goal
    
    def transform_pose(self, input_pose, to_frame):
        original_frame = input_pose.header.frame_id

        try:
            transform = self.tfBuffer.lookup_transform(to_frame, original_frame, time=rospy.Time.now())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("[%s] Cannot transform pose, failed to lookup transform from %s to %s!", self.node_name,
                          original_frame, to_frame)
            return FlexGraspErrorCodes.TRANSFORM_POSE_FAILED

        return tf2_geometry_msgs.do_transform_pose(input_pose, transform)

    def check_movement(self, goal_pose):
        x = goal_pose.pose.position.x
        y = goal_pose.pose.position.y
        z = goal_pose.pose.position.z

        is_safe = True

        if abs(x) > 1.0 or abs(y) > 1.0:
            is_safe = False

        # gripper can not end below the table
        if z < 0.05:
            is_safe = False
        
        return is_safe

    # control robot to desired goal position
    def go_to_pose(self, goal_pose):
        # the goal pose should be of type PoseStamped. E.g. goal_pose=PoseStampled()
        start = self.current_pos
        start_ori = self.current_ori
        goal_= np.array([goal_pose.pose.position.x, goal_pose.pose.position.y, goal_pose.pose.position.z])
        
        # interpolate from start to goal with attractor distance of approx 1 cm
        squared_dist = np.sum(np.subtract(start, goal_)**2, axis=0)
        dist = np.sqrt(squared_dist)
        
        rospy.logdebug(f'Distance: {dist}')
        
        interp_dist = 0.01  # [m]
        step_num = math.floor(dist / interp_dist)
        
        rospy.logdebug(f'num of steps: {step_num}')
        
        x = np.linspace(start[0], goal_pose.pose.position.x, step_num)
        y = np.linspace(start[1], goal_pose.pose.position.y, step_num)
        z = np.linspace(start[2], goal_pose.pose.position.z, step_num)
        
        rot_x = np.linspace(start_ori[0], goal_pose.pose.orientation.x , step_num)
        rot_y = np.linspace(start_ori[1], goal_pose.pose.orientation.y , step_num)
        rot_z = np.linspace(start_ori[2], goal_pose.pose.orientation.z , step_num)
        rot_w = np.linspace(start_ori[3], goal_pose.pose.orientation.w, step_num)
        goal = PoseStamped()
        
        goal.pose.position.x = x[0]
        goal.pose.position.y = y[0]
        goal.pose.position.z = z[0]

        goal.pose.orientation.x = rot_x[0]
        goal.pose.orientation.y = rot_y[0]
        goal.pose.orientation.z = rot_z[0]
        goal.pose.orientation.w = rot_w[0]

        self.goal_pub.publish(goal)

        rospy.logdebug(f'Goal_pose: {goal_pose}')

        stiff_des = Float32MultiArray()

        stiff_des.data = np.array([self.K_pos, self.K_pos, self.K_pos, self.K_ori, self.K_ori, self.K_ori, self.K_ns]).astype(np.float32)
        self.stiff_pub.publish(stiff_des)
        
        goal = PoseStamped()
        
        for i in range(step_num):
            now = time.time()            # get the time
            goal.header.seq = 1
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = "map"

            goal.pose.position.x = x[i]
            goal.pose.position.y = y[i]
            goal.pose.position.z = z[i]

            goal.pose.orientation.x = rot_x[i]
            goal.pose.orientation.y = rot_y[i]
            goal.pose.orientation.z = rot_z[i]
            goal.pose.orientation.w = rot_w[i]
            
            self.goal_pub.publish(goal)
      
            self.r.sleep()
        
        return 'succes'

    def received_messages(self, command=None):
        if command == 'calibration_movement':
            is_received = {'robot_goal_pos': self.robot_goal_pos is not None,
                       'robot_goal_ori': self.robot_goal_ori is not None,
                       'current_pose': self.current_pos is not None,
                       'current_ori': self.current_ori is not None,
                       'all': True}
        else:
            #TODO: add transformation matrix to check
            """Returns a dictionary which contains information about what data has been received"""
            is_received = {'current_pose': self.current_pos is not None,
                        'current_ori': self.current_ori is not None,
                        'delta_pos': self.delta_pos is not None,
                        'delta_ori': self.delta_ori is not None,
                        'all': True}

        for key in is_received:
            is_received['all'] = is_received['all'] and is_received[key]

        return is_received

    def print_received_messages(self, is_received):
        """Prints a warning for the data which has not been received"""
        for key in is_received:
            if not is_received[key]:
                rospy.logwarn("[{0}] Did not receive {1} data yet.".format(self.node_name, key))

    def wait_for_messages(self, command=None, timeout=1):
        start_time = rospy.get_time()
        is_received = {}

        while rospy.get_time() - start_time < timeout:
            is_received = self.received_messages(command=command)
            if is_received['all']:
                rospy.logdebug(f'[{self.node_name}] Received all data')
                return True

            rospy.sleep(0.1)

        self.print_received_messages(is_received)
        return False