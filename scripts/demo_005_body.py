import numpy as np
import keyboard

from DM_Control.Motor_Control import BodyController
from Kinematics.Retarget import RetargetClass
from Kinematics.IK_solver import quat2Rot, compound_Rot_p_2_T, pose_to_homogeneous_matrix
from glove_remote import GloveRemote
import time

def demo_main():
    body_motorcontroller = BodyController()
    body_motorcontroller.Motor_Init("COM4", 921600, 'body')
    body_init_angle = np.zeros(4)
    now_angle = np.zeros(4)

    while True:
            if keyboard.is_pressed('a'):
                motor_cmd_positions =  [0.3,0.3,0,0]
                body_motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle = body_motorcontroller.Cmd_Limit_Clip(body_motorcontroller.get_all_positions())
                
            if keyboard.is_pressed('d'):
                motor_cmd_positions =  [-0.3,-0.3,0,0]
                body_motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle = body_motorcontroller.Cmd_Limit_Clip(body_motorcontroller.get_all_positions())\
                
            if keyboard.is_pressed('w'):
                motor_cmd_positions =  [0,0,0.1,0]
                body_motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle = body_motorcontroller.Cmd_Limit_Clip(body_motorcontroller.get_all_positions())

            if keyboard.is_pressed('s'):
                motor_cmd_positions =  [0,0,-0.1,0]
                body_motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle = body_motorcontroller.Cmd_Limit_Clip(body_motorcontroller.get_all_positions())

            if keyboard.is_pressed('u'):
                motor_cmd_positions =  [0,0,0,0.4]
                body_motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle = body_motorcontroller.Cmd_Limit_Clip(body_motorcontroller.get_all_positions())

            if keyboard.is_pressed('j'):
                motor_cmd_positions =  [0,0,0,-0.2]
                body_motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle = body_motorcontroller.Cmd_Limit_Clip(body_motorcontroller.get_all_positions())

def demo():
    body_motorcontroller = BodyController()
    body_motorcontroller.Motor_Init("COM4", 921600, 'body')
    body_init_angle = np.zeros(4)
    now_angle = np.zeros(4)
    body_motorcontroller.set_speed(0.4)

    motor_cmd_positions =  [0.8,0.8,0,0]
    body_motorcontroller.motor_safe_control(motor_cmd_positions)
    time.sleep(3)

    motor_cmd_positions =  [0,0,0,0]
    body_motorcontroller.motor_safe_control(motor_cmd_positions)
    time.sleep(3)
    
    motor_cmd_positions =  [-0.8,-0.8,0,0]
    body_motorcontroller.motor_safe_control(motor_cmd_positions)
    time.sleep(3)

    motor_cmd_positions =  [0,0,0.5,0]
    body_motorcontroller.motor_safe_control(motor_cmd_positions)
    time.sleep(3)

    motor_cmd_positions =  [0,0,-0.5,0]
    body_motorcontroller.motor_safe_control(motor_cmd_positions)
    time.sleep(3)

    motor_cmd_positions =  [0,0,0,0.3]
    body_motorcontroller.motor_safe_control(motor_cmd_positions)
    time.sleep(3)

    motor_cmd_positions =  [0,0,0,-0.3]
    body_motorcontroller.motor_safe_control(motor_cmd_positions)
    time.sleep(3)




if __name__=="__main__":
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(demo_main())
    # loop.close()
    # demo_main()
    demo()