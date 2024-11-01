
from motor_control import MotorControl
from udp_communication import UDPCommunication
from gripper_pose import GripperPose

class MainProgram:
    def __init__(self):
        self.motor_control = MotorControl()
        self.udp_communication = UDPCommunication("192.168.1.1", 8888)
        self.gripper_pose = GripperPose()
        self.state = "INIT"

    def run(self):
        while True:
            if self.state == "INIT":
                pass
            elif self.state == "RUNNING":
                pass
            elif self.state == "STOP":
                break

if __name__ == "__main__":
    program = MainProgram()
    program.run()
