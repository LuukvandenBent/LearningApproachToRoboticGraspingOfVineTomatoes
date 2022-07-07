import os
import rospy
import rospkg

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget, QMenu
from std_msgs.msg import String, Bool
from flex_shared_resources.msg import GazeboInstruction
from rqt_user_interface.util import initialize_drop_down_button

from rqt_user_interface.experiment_path_interface import ExperimentPathInterface

class RqtFlexGrasp(Plugin):

    def __init__(self, context):
        super(RqtFlexGrasp, self).__init__(context)
        # Give QObjects reasonable names
        self.setObjectName('RqtFlexGrasp')

        # Process standalone plugin command-line arguments
        from argparse import ArgumentParser
        parser = ArgumentParser()
        # Add argument(s) to the parser.
        parser.add_argument("-q", "--quiet", action="store_true",
                      dest="quiet",
                      help="Put plugin in silent mode")
        args, unknowns = parser.parse_known_args(context.argv())
        if not args.quiet:
            print('arguments: ', args)
            print('unknowns: ', unknowns)

        # Create QWidget
        self._widget = QWidget()
        # Get path to UI file which should be in the "resource" folder of this package
        ui_file = os.path.join(rospkg.RosPack().get_path('rqt_user_interface'), 'resource', 'flex_grasp.ui')
        # Extend the widget with all attributes and children from UI file
        loadUi(ui_file, self._widget)
        # Give QObjects reasonable names
        self._widget.setObjectName('RqtFlexGraspUi')
        # Show _widget.windowTitle on left-top of each plugin (when
        # it's set in _widget). This is useful when you open multiple
        # plugins at once. Also if you open multiple instances of your
        # plugin at once, these lines add number to make it easy to
        # tell from pane to pane.
        if context.serial_number() > 1:
            self._widget.setWindowTitle(self._widget.windowTitle() + (' (%d)' % context.serial_number()))

        # Add widget to the user interface
        context.add_widget(self._widget)

        self.pub_command = rospy.Publisher("pipeline_command", String, queue_size=10, latch=False)
        self.pub_experiment = rospy.Publisher("experiment", Bool, queue_size=1, latch=True)
        self.pub_gazebo_interface = rospy.Publisher("gazebo_interface/e_in", GazeboInstruction, queue_size=1, latch=True)

        self.experiment = False
        self._widget.ExperimentButton.setCheckable(True)
        
        # basic commands
        self._widget.OpenButton.clicked[bool].connect(lambda: self.pub_command.publish("open"))
        self._widget.CloseButton.clicked[bool].connect(lambda: self.pub_command.publish("close"))
        self._widget.CalibrateButton.clicked[bool].connect(lambda: self.pub_command.publish("calibrate"))
        self._widget.CalibrateHeightButton.clicked[bool].connect(lambda: self.pub_command.publish("calibrate_height"))

        self._widget.DetectTrussButton.clicked[bool].connect(lambda: self.pub_command.publish("detect_truss"))
        self._widget.DetectGraspPointButton.clicked[bool].connect(lambda: self.pub_command.publish("detect_grasp_point"))
        self._widget.DetectGraspPointCloseButton.clicked[bool].connect(lambda: self.pub_command.publish("detect_grasp_point_close"))
        self._widget.DetectGraspPointNNButton.clicked[bool].connect(lambda: self.pub_command.publish("detect_grasp_point_NN"))

        self._widget.PickButton.clicked[bool].connect(lambda: self.pub_command.publish("pick"))
        self._widget.MovePlaceButton.clicked[bool].connect(lambda: self.pub_command.publish("move_place"))

        self._widget.ApproachGraspPointButton.clicked[bool].connect(lambda: self.pub_command.publish("approach_grasp_point"))
        self._widget.ApproachTrussButton.clicked[bool].connect(lambda: self.pub_command.publish("approach_truss"))
        self._widget.PreGraspButton.clicked[bool].connect(lambda: self.pub_command.publish("pre_grasp"))
        self._widget.GraspButton.clicked[bool].connect(lambda: self.pub_command.publish("grasp"))
        self._widget.MoveRightButton.clicked[bool].connect(lambda: self.pub_command.publish("move_right"))
        self._widget.MoveLeftButton.clicked[bool].connect(lambda: self.pub_command.publish("move_left"))
        self._widget.MoveForwardsButton.clicked[bool].connect(lambda: self.pub_command.publish("move_forwards"))
        self._widget.MoveBackwardsButton.clicked[bool].connect(lambda: self.pub_command.publish("move_backwards"))
        self._widget.MoveUpwardsButton.clicked[bool].connect(lambda: self.pub_command.publish("move_upwards"))
        self._widget.MoveDownwardsButton.clicked[bool].connect(lambda: self.pub_command.publish("move_downwards"))
        self._widget.MoveHomeButton.clicked[bool].connect(lambda: self.pub_command.publish("move_home"))
        self._widget.SavePoseButton.clicked[bool].connect(lambda: self.pub_command.publish("save_pose"))
        self._widget.MoveSavedPoseButton.clicked[bool].connect(lambda: self.pub_command.publish("move_saved_pose"))

        self._widget.ExperimentButton.clicked.connect(self.handle_experiment)

        self.experiment_path_interface = ExperimentPathInterface(self._widget.ExperimentNameButton, self._widget.ExperimentIDButton)

    def shutdown_plugin(self):
        self.pub_command.unregister()

    def save_settings(self, plugin_settings, instance_settings):
        # TODO save intrinsic configuration, usually using:
        # instance_settings.set_value(k, v)
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        # TODO restore intrinsic configuration, usually using:
        # v = instance_settings.value(k)
        pass

    #def trigger_configuration(self):
        # Comment in to signal that the plugin has a way to configure
        # This will enable a setting button (gear icon) in each dock widget title bar
        # Usually used to open a modal configuration dialog

    def handle_spawn_type(self):
        self.spawn_type = str(self._widget.SelectSpawnTypeButton.currentText())

    def handle_set_pose_truss(self):
        spawn_instruction = GazeboInstruction(command=GazeboInstruction.SETPOSE)
        self.pub_gazebo_interface.publish(spawn_instruction)

    def handle_spawn_truss(self):
        spawn_instruction = GazeboInstruction(command=GazeboInstruction.SPAWN, model_type=self.spawn_type)
        self.pub_gazebo_interface.publish(spawn_instruction)

    def handle_experiment(self):
        self.experiment = self._widget.ExperimentButton.isChecked()
        self.pub_experiment.publish(self.experiment)
