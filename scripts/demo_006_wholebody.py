import numpy as np
import keyboard
from DM_Control.DM_CAN import *
from DM_Control.Robot_Control import ArmController
from DM_Control.Robot_Control import BodyController
from Kinematics.Retarget import RetargetClass
from Kinematics.IK_solver import quat2Rot, compound_Rot_p_2_T, pose_to_homogeneous_matrix
from glove_remote import GloveRemote
from vr_remote import VrRemote
import time
import serial



def demo_main():
    # 创建串口连接
    serial_device = serial.Serial("COM4", 921600, timeout=0.5)  # 假设使用COM4作为统一的串口
    motor_control = MotorControl(serial_device)

    # 创建左臂、右臂和躯干控制器
    left_motorcontroller = ArmController(motor_control)
    right_motorcontroller = ArmController(motor_control)
    body_controller = BodyController(motor_control)

    # 初始化各个控制器
    left_motorcontroller.Motor_Init(motor_control, 'left')
    right_motorcontroller.Motor_Init(motor_control, 'right')
    body_controller.Motor_Init(motor_control, 'body')

    # 设置速度
    left_motorcontroller.set_speed(0.8)
    right_motorcontroller.set_speed(0.8)
    body_controller.set_speed(0.5)

    left_controller = RetargetClass(arm_type='left')
    right_controller = RetargetClass(arm_type='right')
    left_init_angle = np.zeros(8)
    right_init_angle = np.zeros(8)
    left_init_angle[4] = -1.57
    right_init_angle[4] = 1.57

    # 电机控制初始化位置
    left_motorcontroller.motor_safe_control(left_init_angle)
    right_motorcontroller.motor_safe_control(right_init_angle)
    left_now_angle = np.zeros(8)
    right_now_angle = np.zeros(8)

    while True:
        if keyboard.is_pressed('enter'):
            break
#     # 后续逻辑...
    

            
    left_now_angle[1:] = left_motorcontroller.Cmd_Limit_Clip(left_motorcontroller.get_all_positions())
    right_now_angle[1:] = right_motorcontroller.Cmd_Limit_Clip(right_motorcontroller.get_all_positions())
    glove_left = GloveRemote(which_hand='l')
    glove_right = GloveRemote(which_hand='r')

    while True:
        left_delta_glove = glove_left.calculate_delta_mat_glove()
        right_delta_glove = glove_right.calculate_delta_mat_glove()

        left_delta_glove_world = glove_left.retarget_delta_glove(left_delta_glove)
        right_delta_glove_world = glove_right.retarget_delta_glove(right_delta_glove)

        print(left_delta_glove_world)
        print(right_delta_glove_world)

        left_motor_cmd_pos = left_controller.get_retarget_joint_angle(left_delta_glove_world,
                                                                      init_angles=left_now_angle,
                                                                      accumulate_mode=True)
        right_motor_cmd_pos = right_controller.get_retarget_joint_angle(right_delta_glove_world,
                                                                        init_angles=right_now_angle,
                                                                        accumulate_mode=True)
        left_motorcontroller.motor_safe_control(left_motor_cmd_pos)
        right_motorcontroller.motor_safe_control(right_motor_cmd_pos)
        left_now_angle[1:] = left_motorcontroller.Cmd_Limit_Clip(left_motorcontroller.get_all_positions())
        right_now_angle[1:] = right_motorcontroller.Cmd_Limit_Clip(right_motorcontroller.get_all_positions())

        if keyboard.is_pressed('space'):
            left_motorcontroller.motor_safe_control(left_init_angle)
            right_motorcontroller.motor_safe_control(right_init_angle)
            left_controller.reset()
            right_controller.reset()
            time.sleep(20.)


def demo_vr():
    # 创建串口连接
    serial_device = serial.Serial("COM4", 921600, timeout=0.5)  # 假设使用COM4作为统一的串口
    motor_control = MotorControl(serial_device)

    # 创建左臂、右臂和躯干控制器
    left_motorcontroller = ArmController(motor_control)
    right_motorcontroller = ArmController(motor_control)
    body_controller = BodyController(motor_control)

    # 初始化各个控制器
    left_motorcontroller.Motor_Init(motor_control, 'left')
    right_motorcontroller.Motor_Init(motor_control, 'right')
    body_controller.Motor_Init(motor_control, 'body')

    # 设置速度
    left_motorcontroller.set_speed(0.8)
    right_motorcontroller.set_speed(0.8)
    body_controller.set_speed(0.5)

    left_controller = RetargetClass(arm_type='left')
    right_controller = RetargetClass(arm_type='right')
    left_init_angle = np.zeros(8)
    right_init_angle = np.zeros(8)
    left_init_angle[4] = -1.57
    right_init_angle[4] = 1.57

    # 电机控制初始化位置
    left_motorcontroller.motor_safe_control(left_init_angle)
    right_motorcontroller.motor_safe_control(right_init_angle)
    left_now_angle = np.zeros(8)
    right_now_angle = np.zeros(8)

    while True:
        if keyboard.is_pressed('enter'):
            break

            
    left_now_angle[1:] = left_motorcontroller.Cmd_Limit_Clip(left_motorcontroller.get_all_positions())
    right_now_angle[1:] = right_motorcontroller.Cmd_Limit_Clip(right_motorcontroller.get_all_positions())
    vr_left = VrRemote(which_hand='l')
    vr_right = VrRemote(which_hand='r')

    while True:
        left_delta_glove_world = vr_left.get_handvr_world()
        right_delta_glove_world = vr_right.get_handvr_world()

        print(left_delta_glove_world)
        print(right_delta_glove_world)


        left_motor_cmd_pos = left_controller.get_retarget_joint_angle(left_delta_glove_world,
                                                                      init_angles=left_now_angle,
                                                                      accumulate_mode=True)
        
        right_motor_cmd_pos = right_controller.get_retarget_joint_angle(right_delta_glove_world,
                                                                        init_angles=right_now_angle,
                                                                        accumulate_mode=True)
        left_motorcontroller.motor_safe_control(left_motor_cmd_pos)
        right_motorcontroller.motor_safe_control(right_motor_cmd_pos)
        left_now_angle[1:] = left_motorcontroller.Cmd_Limit_Clip(left_motorcontroller.get_all_positions())
        right_now_angle[1:] = right_motorcontroller.Cmd_Limit_Clip(right_motorcontroller.get_all_positions())

        if keyboard.is_pressed('space'):
            left_motorcontroller.motor_safe_control(left_init_angle)
            right_motorcontroller.motor_safe_control(right_init_angle)
            left_controller.reset()
            right_controller.reset()
            time.sleep(20.)
#测试缩放值用
def demo_test():
    # 创建串口连接
    serial_device = serial.Serial("COM4", 921600, timeout=0.5)  # 假设使用COM4作为统一的串口
    motor_control = MotorControl(serial_device)

    # 创建左臂、右臂和躯干控制器
    left_motorcontroller = ArmController(motor_control)
    right_motorcontroller = ArmController(motor_control)
    body_controller = BodyController(motor_control)

    # 初始化各个控制器
    left_motorcontroller.Motor_Init(motor_control, 'left')
    right_motorcontroller.Motor_Init(motor_control, 'right')
    body_controller.Motor_Init(motor_control, 'body')

    # 设置速度
    left_motorcontroller.set_speed(0.8)
    right_motorcontroller.set_speed(0.8)
    body_controller.set_speed(0.5)

    left_controller = RetargetClass(arm_type='left')
    right_controller = RetargetClass(arm_type='right')
    left_init_angle = np.zeros(8)
    right_init_angle = np.zeros(8)
    left_init_angle[4] = -1.57
    right_init_angle[4] = 1.57

    # 电机控制初始化位置
    left_motorcontroller.motor_safe_control(left_init_angle)
    right_motorcontroller.motor_safe_control(right_init_angle)
    left_now_angle = np.zeros(8)
    right_now_angle = np.zeros(8)

    while True:
        if keyboard.is_pressed('enter'):
            break

            
    left_now_angle[1:] = left_motorcontroller.Cmd_Limit_Clip(left_motorcontroller.get_all_positions())
    right_now_angle[1:] = right_motorcontroller.Cmd_Limit_Clip(right_motorcontroller.get_all_positions())
    vr_left = VrRemote(which_hand='l')
    vr_right = VrRemote(which_hand='r')

    while True:
        left_delta_glove_world = vr_left.get_handvr_world()
        right_delta_glove_world = vr_right.get_handvr_world()
    
        # print(left_delta_glove_world)
        # print(right_delta_glove_world)
        start_time = time.time()

        left_motor_cmd_pos = left_controller.get_retarget_joint_angle(left_delta_glove_world,
                                                                      init_angles=left_now_angle,
                                                                      accumulate_mode=True)
        
        right_motor_cmd_pos = right_controller.get_retarget_joint_angle(right_delta_glove_world,
                                                                        init_angles=right_now_angle,
                                                                        accumulate_mode=True)
        left_motorcontroller.motor_safe_control(left_motor_cmd_pos)
        right_motorcontroller.motor_safe_control(right_motor_cmd_pos)
        left_now_angle[1:] = left_motorcontroller.Cmd_Limit_Clip(left_motorcontroller.get_all_positions())
        right_now_angle[1:] = right_motorcontroller.Cmd_Limit_Clip(right_motorcontroller.get_all_positions())
        # 获取结束时间
        end_time = time.time()
            
            # 计算时延
        latency = end_time - start_time
        print(f"机器人控制时延: {latency:.6f} 秒")

        if keyboard.is_pressed('space'):
            left_motorcontroller.motor_safe_control(left_init_angle)
            right_motorcontroller.motor_safe_control(right_init_angle)
            left_controller.reset()
            right_controller.reset()
            time.sleep(20.)
if __name__ == "__main__":
    # demo_test()
    demo_main()
