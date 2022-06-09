import rospy
from flex_grasp.msg import FlexGraspErrorCodes

class PlannerStateMachine(object):

    def __init__(self, planner, state_input, update_rate, node_name):
        self.node_name = node_name
        self._update_rate = update_rate
        self._input = state_input
        self._planner = planner

        self._is_idle = True
        self._command = None
        self._shutdown_requested = False
        self.possible_commands = ['approach', 'grasp', 
                                'move_right', 'move_left',
                                'move_forwards', 'move_backwards',
                                'move_upwards', 'move_downwards',
                                'calibration_movement',
                                'home']

    def run(self):
        rate = rospy.Rate(self._update_rate)
        while not self._shutdown_requested:
            if self._is_idle:
                self._process_idle_state()
            else:
                self._process_plan_state()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                return
            except KeyboardInterrupt:
                return

    def _process_idle_state(self):
        command = self._input.command
        
        if command is None:
            return

        elif command in self.possible_commands:
            self._is_idle = False
            self.command = command
            self._input.command_accepted()

        elif command == "e_init":
            self._input.command_accepted()
            self._input.command_completed()

        else:
            self._input.command_rejected()

    def _process_plan_state(self):
        if self.command in self.possible_commands:
            if self._planner.wait_for_messages(command=self.command):
                result = self._planner.plan_movement(movement=self.command)
            else:
                result = FlexGraspErrorCodes.REQUIRED_DATA_MISSING

        else:
            result = FlexGraspErrorCodes.Failure

        self._input.command_completed(result)
        self._transition_to_idle_state()

    def _transition_to_idle_state(self):
        self._is_idle = True
        self._command = None
        rospy.logdebug("[{0}] Transitioned to idle".format(self.node_name))

    def request_shutdown(self):
        rospy.loginfo("[{0}] Shutdown requested".format(self.node_name))
        self._shutdown_requested = True