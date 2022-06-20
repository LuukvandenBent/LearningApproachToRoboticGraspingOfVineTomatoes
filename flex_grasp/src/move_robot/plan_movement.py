#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import time
# import quaternion # pip install numpy-quaternion
from pyquaternion import Quaternion # pip install pyquaternion
from scipy.spatial.transform import Rotation as R
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Float32

import tf
import tf2_ros

from flex_grasp.msg import FlexGraspErrorCodes

class Planner():

    def __init__(self, NODE_NAME, playback=False):
        self.node_name = NODE_NAME
        self.playback = playback

        self.r = rospy.Rate(10)

        self.K_pos = 600
        self.K_ori = 40
        self.K_ns = 10

        self.current_pos = None
        self.current_ori = None

        self.delta_pos = np.array([0.0, 0.0, 0.0])
        self.delta_ori = np.array([0.0, 0.0, 0.0, 0.0])

        self.calibration_pos = None
        self.calibration_ori = None

        self.truss_pos = None
        self.truss_ori = None

        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)

        self.panda_pos_sub = rospy.Subscriber("/cartesian_pose", PoseStamped, self.ee_pose_callback)
        self.truss_pos_sub = rospy.Subscriber("/panda/truss_pose", PoseStamped, self.truss_pose_callback)
        self.robot_goal_pos_sub = rospy.Subscriber('/panda_calibration_pose', PoseStamped, self.calibration_pose_callback)

        self.goal_pub = rospy.Publisher('/equilibrium_pose', PoseStamped, queue_size=0)
        self.gripper_pub = rospy.Publisher('/gripper_online', Float32, queue_size=0)
        self.stiff_pub = rospy.Publisher('/stiffness', Float32MultiArray, queue_size=0)

    def ee_pose_callback(self, data):
        self.current_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.current_ori = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])

    def truss_pose_callback(self, data):
        self.truss_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.truss_ori = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])

    def calibration_pose_callback(self,data):
        self.calibration_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.calibration_ori = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])

    def plan_movement(self, movement=None):

        goal = self.find_goal_pose(movement=movement)
        
        if movement == 'open' or movement == 'close':
            self.gripper_pub.publish(goal)
            return FlexGraspErrorCodes.SUCCESS
        
        else: 
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
        
        # gripper
        if movement == 'open':
            goal = 0.02
        elif movement == 'close':
            goal = 0.004

        # manipulator
        else:
            goal_ori = None
            goal_pos = None

            if movement == 'move_calibrate':
                goal_pos = self.calibration_pos
                goal_ori = self.calibration_ori

            elif movement == 'move_home':
                goal_pos = np.array([-0.4, 0.5, 0.4])
                goal_ori = np.array([0.70710678118, 0.70710678118, 0.0, 0.0])

            elif movement == 'approach' or movement == 'grasp':
                truss_pos = self.truss_pos
                truss_ori = self.truss_ori
                pos, ori = self.transform_pose(input_pos=truss_pos, input_ori=truss_ori)
                
                if movement == 'approach':
                    pos[2] = pos[2] + 0.3       #move above object

                goal_pos = pos
                goal_ori = ori
            
            else:
                if movement == 'move_right':
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

            if goal_pos is None:
                goal_pos = np.array([self.current_pos[0] + self.delta_pos[0], self.current_pos[1] + self.delta_pos[1], self.current_pos[2] + self.delta_pos[2]])

            if goal_ori is None:
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
    
    def transform_pose(self, input_pos=None, input_ori=None):
        """
        Transforms an input pose to the robot base frame
        
        - Input frame: camera
        - Output frame: base
        """
        
        # transform 1: base - gripper
        current_pos = self.current_pos
        current_ori = self.current_ori

        trans_matrix1 = tf.TransformerROS.fromTranslationRotation(self, rotation=current_ori, translation=current_pos)
        rospy.logdebug(f'Translation 1: {trans_matrix1}')

        # transform 2: camera - gripper
        from_frame = 'camera_link'
        to_frame = 'panda/ee_arm_link'
        rospy.logdebug(f'Original frame: {from_frame}')
        rospy.logdebug(f'To frame: {to_frame}')

        try:
            transform = self.tfBuffer.lookup_transform(to_frame, from_frame, time=rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("[%s] Cannot transform pose, failed to lookup transform from %s to %s!", self.node_name,
                          from_frame, to_frame)
            return FlexGraspErrorCodes.TRANSFORM_POSE_FAILED
        
        rospy.logdebug(f'Transform: {transform}')
        delta_pos = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        delta_ori = np.array([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])
        trans_matrix2 = tf.TransformerROS.fromTranslationRotation(self, rotation=delta_ori, translation=delta_pos)
        rospy.logdebug(f'Translation 2: {trans_matrix2}')

        # transform 3: camera - object
        trans_matrix3 = tf.TransformerROS.fromTranslationRotation(self, rotation=input_ori, translation=input_pos)
        rospy.logdebug(f'Translation 3: {trans_matrix3}')

        
        # total transformation
        transformation = np.matmul(trans_matrix1,trans_matrix2)
        transformation = np.matmul(transformation,trans_matrix3)
        rospy.logdebug(f'Translation base --> object: {transformation}') 

        quat = Quaternion(matrix=transformation)
        pos = np.array([transformation[0,3], transformation[1,3], transformation[2,3]])
        ori = np.array([quat[1],quat[2],quat[3],quat[0]])
        
        rospy.logdebug(f'Pos: {pos}')
        rospy.logdebug(f'Ori: {ori}')

        return pos, ori
        

    def check_movement(self, goal_pose):
        x = goal_pose.pose.position.x
        y = goal_pose.pose.position.y
        z = goal_pose.pose.position.z

        is_safe = True

        if abs(x) > 1.0 or abs(y) > 1.0:
            is_safe = False

        # gripper can not end below the table
        if z < 0.03:
            is_safe = False
        
        return is_safe

    # control robot to desired goal position
    def go_to_pose(self, goal_pose):
        # the goal pose should be of type PoseStamped. E.g. goal_pose=PoseStampled()
        start = self.current_pos
        start_ori = self.current_ori

        rospy.logdebug(f'Current position: {start}')
        rospy.logdebug(f'Current orientation: {start_ori}')

        goal_= np.array([goal_pose.pose.position.x, goal_pose.pose.position.y, goal_pose.pose.position.z])
        
        # interpolate from start to goal with attractor distance of approx 1 cm
        squared_dist = np.sum(np.subtract(start, goal_)**2, axis=0)
        dist = np.sqrt(squared_dist)
        
        rospy.logdebug(f'Distance: {dist}')
        
        interp_dist = 0.01  # [m]
        step_num_lin = math.floor(dist / interp_dist)
        rospy.logdebug(f'num of steps linear: {step_num_lin}')
        
        q_start = np.quaternion(start_ori[3], start_ori[0], start_ori[1], start_ori[2])
        q_goal = np.quaternion(goal_pose.pose.orientation.w, goal_pose.pose.orientation.x, goal_pose.pose.orientation.y, goal_pose.pose.orientation.z)
        inner_prod = q_start.x*q_goal.x+q_start.y*q_goal.y+q_start.z*q_goal.z+q_start.w*q_goal.w
        
        if inner_prod < 0:
            q_start.x = -q_start.x
            q_start.y = -q_start.y
            q_start.z = -q_start.z
            q_start.w = -q_start.w
        
        inner_prod = q_start.x*q_goal.x+q_start.y*q_goal.y+q_start.z*q_goal.z+q_start.w*q_goal.w
        theta = np.arccos(np.abs(inner_prod))
        rospy.logdebug(f'theta: {theta}')
        
        interp_dist_polar = 0.01 
        step_num_polar = math.floor(theta / interp_dist_polar)
        rospy.logdebug(f'num of steps polar: {step_num_polar}')

        step_num = np.max([step_num_polar,step_num_lin])
        rospy.logdebug(f'num of steps: {step_num}')

        x = np.linspace(start[0], goal_pose.pose.position.x, step_num)
        y = np.linspace(start[1], goal_pose.pose.position.y, step_num)
        z = np.linspace(start[2], goal_pose.pose.position.z, step_num)

        goal = PoseStamped()
        
        goal.pose.position.x = x[0]
        goal.pose.position.y = y[0]
        goal.pose.position.z = z[0]

        quat = np.slerp_vectorized(q_start, q_goal, 0)
        goal.pose.orientation.x = quat.x
        goal.pose.orientation.y = quat.y
        goal.pose.orientation.z = quat.z
        goal.pose.orientation.w = quat.w

        rospy.logdebug(f'Goal_pose: {goal_pose}')
        
        input("PRESS ENTER TO CONTINUE")
        
        self.goal_pub.publish(goal)

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
            
            quat = np.slerp_vectorized(q_start, q_goal, i/step_num)
            goal.pose.orientation.x = quat.x
            goal.pose.orientation.y = quat.y
            goal.pose.orientation.z = quat.z
            goal.pose.orientation.w = quat.w
            
            self.goal_pub.publish(goal)
      
            self.r.sleep()
        
        return 'succes'

    def received_messages(self, command=None):
        """Returns a dictionary which contains information about what data has been received"""

        if command == 'move_calibrate':
            is_received = {'calibration_goal_pos': self.calibration_pos is not None,
                       'calibration_goal_ori': self.calibration_ori is not None,
                       'current_pose': self.current_pos is not None,
                       'current_ori': self.current_ori is not None,
                       'all': True}

        elif command == 'approach' or command == 'grasp':
            is_received = {'truss_pos': self.truss_pos is not None,
                       'truss_ori': self.truss_ori is not None,
                       'current_pose': self.current_pos is not None,
                       'current_ori': self.current_ori is not None,
                       'all': True}

        else:
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