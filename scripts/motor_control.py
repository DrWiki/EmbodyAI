import serial
import time
import numpy as np
from .DM_CAN import *

class MotorControl:
    """
    统一的电机控制类，包含机械臂、身体和夹持器的控制
    """
    def __init__(self):
        # 机械臂控制器
        self.left_arm = None
        self.right_arm = None
        # 身体控制器
        self.body = None
        # 夹持器控制器
        self.gripper = None
        
        # 默认速度设置
        self.default_arm_speed = 0.3
        self.default_body_speed = 0.2
        self.default_gripper_speed = 20

    def init_arm(self, port: str, rate: int, arm_type: str):
        """
        初始化机械臂
        :param port: 串口号
        :param rate: 波特率
        :param arm_type: 'left' 或 'right'
        """
        if arm_type == 'left':
            self.left_arm = ArmController()
            self.left_arm.Motor_Init(port, rate, 'left')
            print("左臂初始化成功")
        elif arm_type == 'right':
            self.right_arm = ArmController()
            self.right_arm.Motor_Init(port, rate, 'right')
            print("右臂初始化成功")

    def init_body(self, port: str, rate: int):
        """
        初始化身体控制器
        :param port: 串口号
        :param rate: 波特率
        """
        self.body = BodyController()
        self.body.Motor_Init(port, rate, 'body')
        print("身体控制器初始化成功")

    def init_gripper(self, port: str, rate: int = 115200):
        """
        初始化夹持器
        :param port: 串口号
        :param rate: 波特率，默认115200
        """
        try:
            self.gripper = serial.Serial(port, rate)
            print("夹持器初始化成功")
        except Exception as e:
            print(f"夹持器初始化失败: {e}")

    def move_arm(self, positions, arm_type: str, speed: float = None):
        """
        控制机械臂运动
        :param positions: 关节角度列表 [7个值]
        :param arm_type: 'left' 或 'right'
        :param speed: 可选的速度参数
        """
        arm = self.left_arm if arm_type == 'left' else self.right_arm
        if arm is None:
            print(f"{arm_type}臂未初始化")
            return False

        if speed is not None:
            arm.set_speed(speed)

        try:
            arm.motor_safe_control(positions)
            return True
        except Exception as e:
            print(f"机械臂控制错误: {e}")
            return False

    def move_body(self, positions, speed: float = None):
        """
        控制身体运动
        :param positions: 关节角度列表 [4个值]
        :param speed: 可选的速度参数
        """
        if self.body is None:
            print("身体控制器未初始化")
            return False

        if speed is not None:
            self.body.set_speed(speed)

        try:
            self.body.motor_safe_control(positions)
            return True
        except Exception as e:
            print(f"身体控制错误: {e}")
            return False

    def control_gripper(self, position: float, speed: float = None, direction: int = 1):
        """
        控制夹持器
        :param position: 目标位置（角度）
        :param speed: 运动速度（可选）
        :param direction: 运动方向（1顺时针，0逆时针）
        :return: 当前位置
        """
        if self.gripper is None:
            print("夹持器未初始化")
            return None

        speed = speed or self.default_gripper_speed
        try:
            position_read = Motor_control(position, speed, direction, 32, self.gripper)
            return position_read
        except Exception as e:
            print(f"夹持器控制错误: {e}")
            return None

    def get_arm_positions(self, arm_type: str):
        """
        获取机械臂当前位置
        :param arm_type: 'left' 或 'right'
        :return: 位置数组
        """
        arm = self.left_arm if arm_type == 'left' else self.right_arm
        if arm is None:
            print(f"{arm_type}臂未初始化")
            return None
        return arm.get_all_positions()

    def get_body_positions(self):
        """
        获取身体当前位置
        :return: 位置数组
        """
        if self.body is None:
            print("身体控制器未初始化")
            return None
        return self.body.get_all_positions()

    def disable_all(self):
        """
        禁用所有电机
        """
        if self.left_arm:
            self.left_arm.Disable_all_motors()
        if self.right_arm:
            self.right_arm.Disable_all_motors()
        if self.body:
            self.body.Disable_all_motors()
        if self.gripper:
            self.gripper.close()

    def __del__(self):
        """
        析构函数，确保所有设备正确关闭
        """
        self.disable_all()

if __name__ == "__main__":
    # 初始化控制器
    controller = MotorControl()

    # 初始化各个部分
    controller.init_arm(port="COM1", rate=115200, arm_type="left")
    controller.init_arm(port="COM2", rate=115200, arm_type="right")
    controller.init_body(port="COM3", rate=115200)
    controller.init_gripper(port="COM4")

    # 控制左臂运动
    left_arm_positions = [0, 0, 0, 0, 0, 0, 0]  # 7个关节的目标位置
    controller.move_arm(left_arm_positions, arm_type="left", speed=0.3)

    # 控制身体运动
    body_positions = [0, 0, 0, 0]  # 4个关节的目标位置
    controller.move_body(body_positions, speed=0.2)

    # 控制夹持器
    gripper_position = controller.control_gripper(position=600, speed=20, direction=1)

    # 获取当前位置
    left_arm_pos = controller.get_arm_positions("left")
    body_pos = controller.get_body_positions()

    # 程序结束时关闭所有设备
    controller.disable_all()
