#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import time
import quaternion # pip install numpy-quaternion
from pyquaternion import Quaternion # pip install pyquaternion
from scipy.spatial.transform import Rotation as R
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Float32, String

import tf
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from flex_grasp.msg import FlexGraspErrorCodes

class Planner():

    def __init__(self, NODE_NAME, playback=False):
        self.node_name = NODE_NAME
        self.playback = playback

        self.r = rospy.Rate(10)

        self.K_pos = 2000
        self.K_ori = 50
        self.K_ori_x = 50
        self.K_ori_y = 50
        self.K_ori_z = 50
        self.K_ns = 10

        self.approach_height = 0.01     # delta_z when approaching the object
        self.grasp_height = -0.0125
        self.pre_grasp_height = 0.1
        self.current_z = 1000
        self.grasp_point_location = None

        self.current_pos = None
        self.current_ori = None

        self.delta_pos = np.array([0.0, 0.0, 0.0])
        self.delta_ori = np.array([0.0, 0.0, 0.0, 0.0])

        self.calibration_pos = None
        self.calibration_ori = None

        self.saved_pos = None
        self.saved_ori = None

        self.truss_pos = None
        self.truss_ori = None

        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)

        self.panda_pose_sub = rospy.Subscriber("/cartesian_pose", PoseStamped, self.ee_pose_callback)
        self.truss_pose_sub = rospy.Subscriber("/panda/truss_pose", PoseStamped, self.truss_pose_callback)
        self.robot_goal_pose_sub = rospy.Subscriber('/panda_calibration_pose', PoseStamped, self.calibration_pose_callback)
        self.grasp_location_sub = rospy.Subscriber('/panda/grasp_point_location', String, self.grasp_location_callback)

        self.goal_pub = rospy.Publisher('/equilibrium_pose', PoseStamped, queue_size=0)
        self.gripper_pub = rospy.Publisher('/gripper_online', Float32, queue_size=0)
        self.stiff_pub = rospy.Publisher('/stiffness', Float32MultiArray, queue_size=0)
        
        # z position required to determine the final grasp point
        self.delta_z_pub = rospy.Publisher('/panda/delta_z', Float32, queue_size=0)

        # true when grasping in middle of peduncle, false when grasping at end of peduncle
        self.com_grasp = rospy.get_param('/panda/com_grasp')

    def ee_pose_callback(self, data):
        self.current_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.current_ori = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])

    def truss_pose_callback(self, data):
        self.truss_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.truss_ori = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])

    def calibration_pose_callback(self, data):
        self.calibration_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.calibration_ori = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])

    def grasp_location_callback(self, data):
        self.grasp_point_location = data.data
        rospy.logdebug(f'Grasp point location: {self.grasp_point_location}')

    def save_pose(self):
        """
        Current pose is saved
        """
        self.saved_pos = self.current_pos
        self.saved_ori = self.current_ori

    def plan_movement(self, movement=None):

        if movement == 'save_pose':
            self.save_pose()
            rospy.logdebug(f'[{self.node_name}] Current pose is saved')
            return FlexGraspErrorCodes.SUCCESS

        goal = self.find_goal_pose(movement=movement)
        
        if movement == 'open' or movement == 'close':
            self.gripper_pub.publish(goal)
            return FlexGraspErrorCodes.SUCCESS
    
        is_safe = self.check_movement(goal)

        if movement == 'grasp':
            goals = self.adjust_grasp_movement(goal)
        elif movement == 'move_place' and not self.com_grasp:
            goals = self.adjust_move_place_movement(goal)
        elif movement == 'approach_truss':
            goals = self.adjust_approach_truss_movement(goal)
        else:
            goals = [goal]
        
        if is_safe:
            for goal in goals:
                if movement != 'approach_truss':
                    goal = self.check_orientation(goal)
            
                result = self.go_to_pose(goal)
            
        else:
            rospy.logwarn(f'[{self.node_name}] Movement not safe')
            rospy.logdebug(f'Goal pose: {goal}')
            return FlexGraspErrorCodes.FAILURE
        
        if result == 'succes':
            return FlexGraspErrorCodes.SUCCESS
        else:
            return FlexGraspErrorCodes.FAILURE

    def find_goal_pose(self, movement=None):
        
        # gripper
        if movement == 'open':
            goal = 0.02
        elif movement == 'close':
            goal = 0.002

        # manipulator
        else:
            goal_ori = None
            goal_pos = None

            if movement == 'move_calibrate':
                goal_pos = self.calibration_pos
                goal_ori = self.calibration_ori

            elif movement == 'move_home':
                goal_pos = np.array([-0.3, 0.4, 0.4])
                goal_ori = np.array([0.70710678118, 0.70710678118, 0.0, 0.0])
            
            elif movement == 'move_place':
                goal_pos = np.array([-0.53, 0.32, 0.03])
                goal_ori = np.array([0.70710678118, 0.70710678118, 0.0, 0.0])

            elif movement == 'move_saved_pose':
                goal_pos = self.saved_pos
                goal_ori = self.saved_ori
            
            elif movement == 'pickup':
                goal_pos, goal_ori = self.find_pickup_pose()

            elif movement == 'approach_truss' or movement == 'approach_truss_2' or movement == 'approach_grasp_point' or movement == 'pre_grasp' or movement == 'grasp':
                truss_pos = self.truss_pos
                truss_ori = self.truss_ori

                # adjust the truss_pose because calibration is not perfect
                pos, ori = self.adjust_pose(input_pos=truss_pos, input_ori=truss_ori)

                if movement == 'pre_grasp':
                    pos = self.adjust_pre_grasp_pos(input_pos=pos)

                if movement == 'grasp':
                    pos = self.adjust_grasp_pos(input_pos=pos)
                    if not self.com_grasp:
                        ori = self.adjust_grasp_ori(input_ori=ori)

                # transform pose with calibration transform
                pos, ori = self.transform_pose(input_pos=pos, input_ori=ori, movement=movement)

                goal_pos = pos
                goal_ori = ori

                if movement == 'approach_truss':
                    goal_pos[2] = 0.17

                    sf = np.sqrt(1/(goal_ori[0]**2 + goal_ori[1]**2))             #create unit length array
                    goal_ori = np.array([goal_ori[0]*sf, goal_ori[1]*sf, 0, 0])   #ensure the end-effector is vertically oriented

                elif movement == 'approach_truss_2':
                    goal_pos[2] = 0.17
                    goal_ori = self.current_ori

                    sf = np.sqrt(1/(goal_ori[0]**2 + goal_ori[1]**2))             #create unit length array
                    goal_ori = np.array([goal_ori[0]*sf, goal_ori[1]*sf, 0, 0])   #ensure the end-effector is vertically oriented
                
                elif movement == 'approach_grasp_point':
                    goal_pos[2] = goal_pos[2] + self.approach_height
                    rospy.loginfo(f'[{self.node_name}] Moving {self.approach_height*100}cm above object')
                
                elif movement == 'pre_grasp':
                    goal_pos[2] = goal_pos[2] + self.pre_grasp_height
                    goal_ori = self.current_ori
                    
                    self.current_z = self.current_pos[2]

                elif movement == 'grasp':
                    goal_pos[2] = goal_pos[2] + self.grasp_height

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

    def adjust_pose(self, input_pos=None, input_ori=None):
        """
        Adjust pose slightly because calibration is not perfect
        """

        delta_pos = np.array([-0.005, 0.02, 0.003])
        delta_ori = np.array([-0.00376818,  0.04237262, -0.00156235, -0.07763563])

        pos = input_pos + delta_pos
        ori = input_ori + delta_ori
        
        return pos, ori

    def adjust_pre_grasp_pos(self, input_pos=None):
        """
        Adjust pre_grasp pose because of calibration-gripper offset
        """

        delta_pos = np.array([0.0, -0.04, -0.05])
        pos = input_pos + delta_pos
        return pos

    def adjust_grasp_movement(self, goal):
        """
        When grasping, first move to x cm above object and then move down
        """
        z = 0.02
        step_size = 0.005
        num_steps = int(z/step_size)
        
        goals = []
        for i in range(num_steps):
            _goal = PoseStamped()
                
            _goal.pose.position.x = goal.pose.position.x
            _goal.pose.position.y = goal.pose.position.y
            _goal.pose.position.z = goal.pose.position.z + (z - (step_size * i))

            _goal.pose.orientation.x = goal.pose.orientation.x
            _goal.pose.orientation.y = goal.pose.orientation.y
            _goal.pose.orientation.z = goal.pose.orientation.z
            _goal.pose.orientation.w = goal.pose.orientation.w

            goals.append(_goal)

        goals.append(goal)

        return goals
    
    def adjust_move_place_movement(self, goal):
        """
        Placing a truss grasped at the end of the peduncle requires
        a different placing movement
        """
        
        current_pos = self.current_pos
        current_ori = self.current_ori
        
        goals = []

        # Goal 1
        _goal1 = PoseStamped()
        _goal1.pose.position.x = goal.pose.position.x
        _goal1.pose.position.y = goal.pose.position.y
        _goal1.pose.position.z = current_pos[2]

        _goal1.pose.orientation.x = current_ori[0]
        _goal1.pose.orientation.y = current_ori[1]
        _goal1.pose.orientation.z = current_ori[2]
        _goal1.pose.orientation.w = current_ori[3]

        # Goal 2
        _goal2 = PoseStamped()
        _goal2.pose.position.x = goal.pose.position.x
        _goal2.pose.position.y = goal.pose.position.y + 0.2
        _goal2.pose.position.z = goal.pose.position.z

        _goal2.pose.orientation.x = 0.707
        _goal2.pose.orientation.y = 0.707
        _goal2.pose.orientation.z = 0
        _goal2.pose.orientation.w = 0

        # euler_angles = euler_from_quaternion(current_ori, 'sxyz')
        # # angle (between -135 and -180) or (between 135 and 180 deg)
        # if euler_angles[1] < 0:
        #     _goal2.pose.orientation.x = 0.707
        # else:
        #     _goal2.pose.orientation.x = -0.707
        
        # _goal2.pose.orientation.y = 0.707
        # _goal2.pose.orientation.z = 0
        # _goal2.pose.orientation.w = 0

        goals.append(_goal1)
        goals.append(_goal2)

        return goals

    def adjust_approach_truss_movement(self, goal):
        """
        Adjust the movement so the gripper is more horizontal
        First move x cm above the object
        Then move down, this ensure a more horizontal orientation of the Panda
        """

        goals = []

        _goal1 = PoseStamped()

        if goal.pose.orientation.x > 0.6 and goal.pose.orientation.y > 0.6:
            # gripper faces forward
            _goal1.pose.position.x = goal.pose.position.x
            _goal1.pose.position.y = goal.pose.position.y - 0.2
        else:
            # gripper faces to the right
            _goal1.pose.position.x = goal.pose.position.x - 0.2
            _goal1.pose.position.y = goal.pose.position.y

        _goal1.pose.position.z = goal.pose.position.z + 0.2
        _goal1.pose.orientation = goal.pose.orientation

        goals.append(_goal1)
        goals.append(goal)

        return goals

    def adjust_grasp_pos(self, input_pos=None):
        """
        Adjust grasp pose because of calibration-gripper offset
        and because of the gripper orientation differing across the table
        """

        if self.current_ori[0] > 0.6 and self.current_ori[1] > 0.6:
            # gripper faces forward
            gripper_orientation = 'forwards'
            if self.current_pos[1] > 0.48:
                truss_location = 'front'
            elif self.current_pos[1] < 0.48 and self.current_pos[1] > 0.38:
                truss_location = 'middle'
            elif self.current_pos[1] < 0.38:
                truss_location = 'back'
        else:
            # gripper faces to the right
            gripper_orientation = 'sideways'
            if self.current_pos[0] > -0.32:
                truss_location = 'front'
            elif self.current_pos[0] < -0.32 and self.current_pos[0] > -0.44:
                truss_location = 'middle'
            elif self.current_pos[0] < -0.44:
                truss_location = 'back'

        rospy.logdebug(f'Gripper orientation: {gripper_orientation}')
        rospy.logdebug(f'Truss location: {truss_location}')

        # correction for tilting of gripper
        if truss_location == 'front':
            if gripper_orientation == 'forwards':
                delta_pos = np.array([0.0, -0.015, -0.005])
            else:
                delta_pos = np.array([0.005, -0.015, -0.005])
        
        elif truss_location == 'middle':
            if gripper_orientation == 'forwards':
                delta_pos = np.array([0.0, -0.015, -0.005])
            else:
                delta_pos = np.array([0.005, -0.013, -0.005])
        
        elif truss_location == 'back':
            if gripper_orientation == 'forwards':
                delta_pos = np.array([0.0, -0.015, -0.0075])
            else:
                delta_pos = np.array([0.0, -0.015, -0.015])

        pos = input_pos + delta_pos
        return pos

    def adjust_grasp_ori(self, input_ori=None):
        """
        When grasping at the end of peduncle, the gripper orientation
        must sometimes be flipped in order to not run into joint limits 
        when picking up the truss 
        """

        if self.current_ori[0] > 0.6 and self.current_ori[1] > 0.6:
            gripper_orientation = 'forwards'
        else:
            gripper_orientation = 'sideways'
        
        rospy.logdebug(f'Gripper orientation: {gripper_orientation}')
        rospy.logdebug(f'Grasp point location: {self.grasp_point_location}')

        flip = False

        if gripper_orientation == 'sideways':
            if input_ori[0] > 0 and input_ori[1] > 0:
                if self.grasp_point_location == 'right':
                    flip = True
            if input_ori[0] > 0 and input_ori[1] < 0:
                if self.grasp_point_location == 'left':
                    flip = True
        else:
            if input_ori[0] > 0.6 and -0.8 < input_ori[1] < 0.8:
                if self.grasp_point_location == 'right':
                    flip = True
            if 0 < input_ori[0] < 0.8 and (input_ori[1] < -0.707 or input_ori[1] > 0.707):
                if self.grasp_point_location == 'left':
                    flip = True

        if flip:
            x = input_ori[0]
            y = input_ori[1]
            z = input_ori[2]
            w = input_ori[3]
            quat = (x,y,z,w)
            rospy.logdebug(f'[{self.node_name}] Orientation is flipped 180 degrees because of joint limits')
            rospy.logdebug(f'Quaternion before flip: {quat}')

            euler_angles = euler_from_quaternion(quat, 'sxyz')
            
            # this somehow flips the sign of the angle
            euler_angles = (euler_angles[0], euler_angles[1], euler_angles[2]+np.pi)
            quat = quaternion_from_euler(euler_angles[0], euler_angles[1], -euler_angles[2], 'rxyz')
            ori = np.array(quat)
            rospy.logdebug(f'Quaternion after flip: {ori}')
        else:
            ori = input_ori

        return ori
    
    def transform_pose(self, input_pos=None, input_ori=None, movement=None):
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
        
        if movement == 'approach_truss':
            delta_pos = np.array([0., 0., 0.])
        else:
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

    def find_pickup_pose(self):
        """
        Finds the pose to pickup the truss when end of peduncle grasping is used
        """

        
        # current_ori = self.current_ori
        # grasp_point_location = self.grasp_point_location

        # if grasp_point_location is None:
        #     grasp_point_location = 'left'
        #     rospy.logwarn(f'[{self.node}] Grasp point location is unknown, setting it to {grasp_point_location}')

        current_pos = self.current_pos
        delta_pos = np.array([0.0, 0.0, 0.3])

        goal_pos = current_pos + delta_pos
        goal_ori = np.array([-0.5, -0.5, -0.5, 0.5]) 

        # euler_angles = euler_from_quaternion(current_ori, 'sxyz')

        # # angle (between -135 and -180) or (between 135 and 180 deg)
        # if euler_angles[2] > 2.356 or euler_angles[2] < -2.356:
        #     if grasp_point_location == 'right':
        #         goal_ori = np.array([-0.5, -0.5, -0.5, 0.5])
        #     elif grasp_point_location == 'left':
        #         goal_ori = np.array([-0.5, 0.5, 0.5, 0.5])

        # else:
        #     if grasp_point_location == 'right':
        #         goal_ori = np.array([-0.5, 0.5, 0.5, 0.5])
        #     elif grasp_point_location == 'left':
        #         goal_ori = np.array([-0.5, -0.5, -0.5, 0.5])
        
        return goal_pos, goal_ori

    def check_movement(self, goal_pose):
        """
        Check for collisions with table or end points that are far away
        """
        
        x = goal_pose.pose.position.x
        y = goal_pose.pose.position.y
        z = goal_pose.pose.position.z

        is_safe = True

        if abs(x) > 1.0 or abs(y) > 1.0:
            is_safe = False

        # gripper can not end below the table
        if z < 0.005:
            is_safe = False
        
        return is_safe

    def check_orientation(self,goal_pose):
        """
        Ensure gripper orientation is within joint limits
        """

        x = goal_pose.pose.orientation.x
        y = goal_pose.pose.orientation.y
        z = goal_pose.pose.orientation.z
        w = goal_pose.pose.orientation.w
        quat = (x,y,z,w)

        euler_angles = euler_from_quaternion(quat, 'sxyz')

        # angle between 0 and -90 deg
        if euler_angles[2] < 0 and euler_angles[2] > -1.57:
            rospy.logdebug(f'[{self.node_name}] Orientation is flipped 180 degrees because of joint limits')
            
            # this somehow flips the sign of the angle
            euler_angles = (euler_angles[0], euler_angles[1], euler_angles[2]+np.pi)
            quat = quaternion_from_euler(euler_angles[0], euler_angles[1], -euler_angles[2], 'rxyz')
            goal_pose.pose.orientation.x = quat[0]
            goal_pose.pose.orientation.y = quat[1]
            goal_pose.pose.orientation.z = quat[2]
            goal_pose.pose.orientation.w = quat[3]

        return goal_pose

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

        stiff_des.data = np.array([self.K_pos, self.K_pos, self.K_pos, self.K_ori_x, self.K_ori_y, self.K_ori_z, self.K_ns]).astype(np.float32)
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
     
        # publish delta_z of pre_grasp movement (needed for determining final grasp point)
        delta_z = self.current_z - self.current_pos[2]
        self.delta_z_pub.publish(delta_z)
        
        return 'succes'

    def received_messages(self, command=None):
        """Returns a dictionary which contains information about what data has been received"""

        if command == 'move_calibrate':
            is_received = {'calibration_goal_pos': self.calibration_pos is not None,
                       'calibration_goal_ori': self.calibration_ori is not None,
                       'current_pose': self.current_pos is not None,
                       'current_ori': self.current_ori is not None,
                       'all': True}

        elif command == 'approach_truss' or command == 'approach_grasp_point' or command == 'pre_grasp' or command == 'grasp':
            is_received = {'truss_pos': self.truss_pos is not None,
                       'truss_ori': self.truss_ori is not None,
                       'current_pose': self.current_pos is not None,
                       'current_ori': self.current_ori is not None,
                       'all': True}

        elif command == 'move_saved_pose':
            is_received = {'saved_pos': self.saved_pos is not None,
                       'saved_ori': self.saved_ori is not None,
                       'all': True}

        elif command == 'open' or command == 'close':
            is_received = {'all': True}

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