#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:28:49 2020

@author: taeke

edited by: Jens Zuurbier
"""

import rospy
import numpy as np
import tf2_ros # for error messages
import tf2_geometry_msgs
import geometry_msgs

# messages
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, TransformStamped
from flex_grasp.msg import FlexGraspErrorCodes

from easy_handeye.handeye_client import HandeyeClient
from flex_shared_resources.errors.flex_grasp_error import flex_grasp_error_log
from flex_shared_resources.utils.conversions import list_to_position, list_to_orientation
from flex_shared_resources.utils.communication import Communication
from flex_shared_resources.data_logger import DataLogger
from flex_shared_resources.experiment_info import ExperimentInfo

class Calibration(object):
    """Calibration"""
    node_name = "CALIBRATION"

    def __init__(self, node_name, playback=False):
        # TODO: whe shouldn't have to run this node in a seperate calibration namespace, it would make things much better
        self.robot_ns = 'panda'

        self.node_name = node_name
        self.playback = playback
        self.command = None
        self.experiment_path = None

        if self.playback:
            rospy.loginfo(f"[{self.node_name}] Calibration launched in playback mode!")

        rospy.sleep(5)
        rospy.logdebug(f"[{self.node_name}] initializing hand eye client")
        self.client = HandeyeClient()
        self.experiment_info = ExperimentInfo(self.node_name, namespace=self.robot_ns, id="initial_calibration")

        # Listen
        rospy.logdebug(f"[{self.node_name}] initializing tf2_ros buffer")
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.calibration_frame = rospy.get_param('/' + self.robot_ns + '/calibration_eye_on_hand' + '/' + 'robot_base_frame')
        self.planning_frame = rospy.get_param('/' + self.robot_ns + '/calibration_eye_on_hand' + '/' + 'tracking_base_frame')
        self.end_effector_frame = rospy.get_param('/' + self.robot_ns + '/calibration_eye_on_hand' + '/' + 'robot_effector_frame')
        self.pose_array = None

        self.pub_e_out = rospy.Publisher("~e_out", FlexGraspErrorCodes, queue_size=10, latch=True)
        
        move_robot_topic = '/' + self.robot_ns + '/' + 'plan_movement'
        pose_array_topic = '/' + self.robot_ns + '/' + 'pose_array'
        calibration_pose_topic = '/panda_calibration_pose'
        
        self.move_robot_communication = Communication(move_robot_topic, timeout=15, node_name=self.node_name)

        self.pose_array_publisher = rospy.Publisher(pose_array_topic, PoseArray, queue_size=5, latch=True)
        self.calibration_pose_publisher = rospy.Publisher(calibration_pose_topic, PoseStamped, queue_size=10, latch=False)

        self.output_logger = DataLogger(self.node_name, {"calibration": "calibration_transform"},
                                        {"calibration": TransformStamped}, bag_name=self.node_name)
        
        # Subscribe to path where results are stored
        experiment_pwd_topic = '/' + self.robot_ns + '/' + 'experiment_pwd'
        rospy.Subscriber(experiment_pwd_topic, String, self.experiment_pwd_cb)

        # Subscribe
        rospy.Subscriber("~e_in", String, self.e_in_cb)

        # Subscribe to cartesian pose to broadcast end_effector tf
        rospy.Subscriber('/cartesian_pose', PoseStamped, self.cartesian_pose_cb)

        # Subscribe to aruco tracker for it to publish the tf
        rospy.Subscriber('/' + self.robot_ns + '/' + 'aruco_tracker/pose', PoseStamped, self.aruco_tracker_cb)

        # Subscribe to calibration transform to broadcast to '/tf' topic
        calibration_tf_topic = '/' + self.robot_ns + '/calibration_eye_on_hand' + '/' + 'calibration_transform'
        rospy.Subscriber(calibration_tf_topic, TransformStamped, self.broadcast_tf_cb)

    def e_in_cb(self, msg):
        if self.command is None:
            self.command = msg.data
            rospy.logdebug("[{0}] Received event message: {1}".format(self.node_name, self.command))

            # reset outputting message
            msg = FlexGraspErrorCodes()
            msg.val = FlexGraspErrorCodes.NONE
            self.pub_e_out.publish(msg)

    def aruco_tracker_cb(self, msg):
        pass

    def cartesian_pose_cb(self, data):
        '''
        Creates and broadcasts transform:
            from    'panda/base_link' 
            to      'panda/ee_arm_link'
        '''
        broadcaster = tf2_ros.StaticTransformBroadcaster()

        position = data.pose.position
        orientation = data.pose.orientation

        transform = geometry_msgs.msg.TransformStamped()

        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.calibration_frame
        transform.child_frame_id = self.end_effector_frame

        transform.transform.translation.x = position.x
        transform.transform.translation.y = position.y
        transform.transform.translation.z = position.z
        
        transform.transform.rotation.x = orientation.x
        transform.transform.rotation.y = orientation.y
        transform.transform.rotation.z = orientation.z
        transform.transform.rotation.w = orientation.w
        
        broadcaster.sendTransform(transform)

    def experiment_pwd_cb(self, msg):
        """callback to find the path where results are stored"""
        path = msg.data
        self.experiment_path = path

    def broadcast_tf_cb(self, data):
        rospy.logdebug(f'[{self.node_name}] Found calibration tf and broadcasting it')
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        transform = data
        broadcaster.sendTransform(transform)

    def init_poses_1(self):
        r_amplitude = 0.0
        z_amplitude = -0.05

        r_min = 0.24
        z_min = 0.28 # 0.05

        pos_intervals = 2
        if abs(r_amplitude) < 0.0001:
            r_vec = [r_min]
        else:
            r_vec = np.linspace(r_min, r_min + 2*r_amplitude, pos_intervals)
        
        if abs(z_amplitude) < 0.0001:
            z_vec = [z_min]
        else:
            z_vec = np.linspace(z_min, z_min + 2*z_amplitude, pos_intervals)

        ai_amplitude = np.deg2rad(38.0) / 2.0     #rot around y-axis in base frame
        aj_amplitude = np.deg2rad(38.0) / 2.1     #rot around x-axis in base frame
        ak_amplitude = np.deg2rad(38.0) / 2.0     #rot around z-axis in base frame

        rot_intervals = 2
        ai_vec = np.linspace(-ai_amplitude, ai_amplitude, rot_intervals)
        aj_vec = np.linspace(-aj_amplitude, aj_amplitude, rot_intervals)
        ak_vec = np.linspace(-ak_amplitude, ak_amplitude, rot_intervals)

        rospy.logdebug(f'ai_vec: {ai_vec}')
        rospy.logdebug(f'aj_vec: {aj_vec}')
        rospy.logdebug(f'ak_vec: {ak_vec}')

        return self.generate_poses(r_vec, z_vec, ai_vec, aj_vec, ak_vec)

    def init_poses_2(self):

        surface_height = 0.019
        height_finger = 0.040  # [m]
        finger_link2ee_link = 0.023  # [m]
        grasp_height = height_finger + finger_link2ee_link - surface_height


        sag_angle = np.deg2rad(6.0) # [deg]
        r_amplitude = 0.08
        z_amplitude = 0.00

        r_min = 0.10
        z_min = grasp_height  # 0.05

        pos_intervals = 3
        if pos_intervals == 1:
            r_vec = [r_min + r_amplitude]  # np.linspace(x_min, x_min + 2*x_amplitude, 2) #
            z_vec = [z_min + z_amplitude]
        else:
            r_vec = np.linspace(r_min, r_min + 2 * r_amplitude, pos_intervals)
            z_vec = np.linspace(z_min, z_min + 2 * z_amplitude, pos_intervals) + np.tan(sag_angle)*r_vec

        ak_amplitude = np.deg2rad(15.0)

        rot_intervals = 2
        ai_vec = [np.deg2rad(0)]
        aj_vec = [np.deg2rad(90)]
        ak_vec = [-ak_amplitude, ak_amplitude]
        return self.generate_poses_2(r_vec, z_vec, ai_vec, aj_vec, ak_vec)

    def generate_poses(self, r_vec, z_vec, ai_vec, aj_vec, ak_vec):
        pose_array = PoseArray()
        pose_array.header.frame_id = self.calibration_frame
        pose_array.header.stamp = rospy.Time.now()

        poses = []
        
        for r in r_vec:
            for z in z_vec:
                for ak in ak_vec:
                    for aj in aj_vec:
                        for ai in ai_vec:
                            pose = Pose()

                            x = abs(r * np.cos(ak))
                            y = abs(r * np.sin(ak))
                            pose.position = list_to_position([x, y, z])

                            pose.orientation = list_to_orientation([ai, aj, ak])

                            poses.append(pose)

        pose_array.poses = poses

        self.pose_array_publisher.publish(pose_array)
        self.pose_array = pose_array
        return FlexGraspErrorCodes.SUCCESS

    def generate_poses_2(self, r_vec, z_vec, ai_vec, aj_vec, ak_vec):
        pose_array = PoseArray()
        pose_array.header.frame_id = self.calibration_frame
        pose_array.header.stamp = rospy.Time.now()

        poses = []
        for ak in ak_vec:
            for r, z in zip(r_vec, z_vec):
                for aj in aj_vec:
                    for ai in ai_vec:
                        pose = Pose()

                        x = r * np.cos(ak)
                        y = r * np.sin(ak)
                        pose.position = list_to_position([x, y, z])

                        pose.orientation = list_to_orientation([ai, aj, ak])

                        poses.append(pose)

        pose_array.poses = poses

        self.pose_array_publisher.publish(pose_array)
        self.pose_array = pose_array
        return FlexGraspErrorCodes.SUCCESS

    def get_initial_base_cam_trans(self):
        '''
        Rough estimate of the starting position of the camera 
        with respect to the base
        
        This determines the calibration poses
        '''
        trans = TransformStamped()

        trans
        trans.transform.translation.x = -0.3
        trans.transform.translation.y = 0.2
        trans.transform.translation.z = 0.7
        
        trans.transform.rotation.x = 0.707
        trans.transform.rotation.y = 0.707
        trans.transform.rotation.z = 0
        trans.transform.rotation.w = 0

        trans.header.frame_id = 'panda/base_link'
        trans.child_frame_id = 'camera_link'

        return trans

    def calibrate(self, track_marker=True):

        sample_list = self.client.get_sample_list().camera_marker_samples
        n_samples = len(sample_list)
        if n_samples > 0:
            rospy.loginfo("[{0}] Found {1} old calibration samples, deleting them before recalibrating!".format(self.node_name, n_samples))
            for i in reversed(range(n_samples)):
                rospy.loginfo("[{0}] Deleting sample {1}".format(self.node_name, i))
                self.client.remove_sample(i)

            sample_list = self.client.get_sample_list().camera_marker_samples
            n_samples = len(sample_list)

            if n_samples > 0:
                rospy.logwarn("[{0}] Failed to remove all old samples, cannot calibrate".format(self.node_name))
                print(sample_list)
                return FlexGraspErrorCodes.FAILURE
            else:
                rospy.loginfo("[{0}] All old samples have been removed".format(self.node_name))

        if self.playback:
            rospy.loginfo("[{0}] Playback is active: publishing messages from bag!".format(self.node_name))
            messages = self.output_logger.load_messages_from_bag(self.experiment_info.path, self.experiment_info.id)
            if messages is not None:
                self.broadcast(messages['calibration'])
                return FlexGraspErrorCodes.SUCCESS
            else:
                return FlexGraspErrorCodes.FAILURE

        if self.pose_array is None:
            rospy.logwarn(f"[{self.node_name}] pose_array is still empty")
            return FlexGraspErrorCodes.REQUIRED_DATA_MISSING
        
        trans = self.get_initial_base_cam_trans()
        
        rospy.logdebug(f'Transform: {trans}')

        # go to home
        result = self.move_robot_communication.wait_for_result("move_home")

        if result != FlexGraspErrorCodes.SUCCESS:
            return result
        
        rospy.logdebug(f'{len(self.pose_array.poses)} calibration poses')
        rospy.logdebug(f'Calibration poses: {self.pose_array.poses}')

        for i, pose in enumerate(self.pose_array.poses):
            rospy.logdebug(f'Calibration pose {i+1}/{len(self.pose_array.poses)}')

            if rospy.is_shutdown():
                return FlexGraspErrorCodes.SHUTDOWN

            pose_stamped = PoseStamped()
            pose_stamped.header = self.pose_array.header
            pose_stamped.pose = pose

            # transform to planning frame
            pose_trans = tf2_geometry_msgs.do_transform_pose(pose_stamped, trans)

            rospy.logdebug(f'Pose Transformerd: {pose_trans}')

            self.calibration_pose_publisher.publish(pose_trans)
            result = self.move_robot_communication.wait_for_result("move_calibrate") # timout = 5?

            if result == FlexGraspErrorCodes.SUCCESS:
                if track_marker:
                    # camera delay + wait a small amount of time for vibrations to stop
                    rospy.sleep(3.0)
                    try:
                        input("PRESS ENTER TO TAKE SAMPLE")
                        self.client.take_sample()
                    except:
                        rospy.logwarn(f"[{self.node_name}] Failed to take sample, marker might not be visible.")
                        return FlexGraspErrorCodes.TAKE_SAMPLE_ERROR

            elif result == FlexGraspErrorCodes.DYNAMIXEL_ERROR:
                rospy.logwarn(f"[{self.node_name}] Dynamixel error triggered, returning error")
                return result
            elif result == FlexGraspErrorCodes.DYNAMIXEL_SEVERE_ERROR:
                rospy.logwarn(f"[{self.node_name}] Dynamixel error triggered, returning error")
                return result

        # return to home
        result = self.move_robot_communication.wait_for_result("move_home")
        
        if result != FlexGraspErrorCodes.SUCCESS:
            return result

        # compute calibration transform
        if not track_marker:
            return FlexGraspErrorCodes.SUCCESS
        else:
            calibration_transform = self.client.compute_calibration()
            rospy.logdebug(f"[{self.node_name}] Calibration transform: {calibration_transform.calibration.transform}")

            if calibration_transform.valid:
                rospy.loginfo(f"[{self.node_name}] Found valid transfrom")
                self.broadcast(calibration_transform.calibration.transform)
                self.client.save()
                return FlexGraspErrorCodes.SUCCESS
            else:
                rospy.logwarn(f"[{self.node_name}] Computed calibration is invalid")
                return FlexGraspErrorCodes.FAILURE

    def calibrate_manually(self):
        '''
        Calibrate by moving the manipulator manually 
        '''

        value = int(input("ENTER THE NUMBER OF CALIBRATION POSES: "))
        
        if value < 3:
            rospy.logdebug(f'[{self.node_name}] Exiting calibration')
            return FlexGraspErrorCodes.SUCCESS

        rospy.logdebug(f'{value} calibration poses')

        for i in range(value):
            rospy.logdebug(f'Calibration pose {i+1}/{value}')

            if rospy.is_shutdown():
                return FlexGraspErrorCodes.SHUTDOWN

            try:
                input("PRESS ENTER TO TAKE SAMPLE")
                self.client.take_sample()
            except:
                rospy.logwarn(f"[{self.node_name}] Failed to take sample, marker might not be visible.")
                return FlexGraspErrorCodes.TAKE_SAMPLE_ERROR
        
        calibration_transform = self.client.compute_calibration()
        rospy.logdebug(f"[{self.node_name}] Calibration transform: {calibration_transform.calibration.transform}")

        response = input("DO YOU WANT TO BROADCAST AND SAVE THIS TRANSFORM? ('y' is yes)")
        
        if response == 'y':
            if calibration_transform.valid:
                rospy.loginfo(f"[{self.node_name}] Found valid transfrom")
                self.broadcast(calibration_transform.calibration.transform)
                self.client.save()
            else:
                rospy.logwarn(f"[{self.node_name}] Computed calibration is invalid")
                return FlexGraspErrorCodes.FAILURE

        return FlexGraspErrorCodes.SUCCESS

    def broadcast(self, transform):
        rospy.loginfo("[{0}] Broadcasting result".format(self.node_name))
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        broadcaster.sendTransform(transform)

        if not self.playback:
            self.output_logger.write_messages_to_bag({"calibration": transform},
                                                     self.experiment_info.path, self.experiment_info.id)

    def publish_calibration_tf(self):
        '''
        Broadcasts calibration transform 
        if it exists locally on computer
        '''
        bag_path = self.experiment_path + '/initial_calibration'
        bag_id = ''
        rospy.logdebug(f'bag path: {bag_path}')
        self.output_logger.publish_messages_from_bag(bag_path, bag_id)

    def take_action(self):
        msg = FlexGraspErrorCodes()
        result = None

        if self.command == 'e_init':
            result = FlexGraspErrorCodes.SUCCESS

        # elif self.command == 'calibrate':
        #     result = self.init_poses_1()

        #     if result == FlexGraspErrorCodes.SUCCESS:
        #         result = self.calibrate()

        elif self.command == 'calibrate':
            self.publish_calibration_tf()
            result = self.calibrate_manually()

        elif self.command == 'calibrate_height':
            result = self.init_poses_2()

            if result == FlexGraspErrorCodes.SUCCESS:
                result = self.calibrate(track_marker=False)

        elif self.command is not None:
            rospy.logwarn("[CALIBRATE] Can not take an action: received unknown command %s!", self.command)
            result = FlexGraspErrorCodes.UNKNOWN_COMMAND

        # publish success
        if result is not None:
            msg.val = result
            flex_grasp_error_log(result, self.node_name)
            self.pub_e_out.publish(msg)
            self.command = None
