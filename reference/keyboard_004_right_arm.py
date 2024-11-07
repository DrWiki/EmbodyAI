import numpy as np
# import keyboard
import cv2

from DM_Control.Motor_Control import ArmController
from Kinematics.Retarget import RetargetClass
from Kinematics.IK_solver import quat2Rot, compound_Rot_p_2_T, pose_to_homogeneous_matrix
import asyncio, time

def demo_main():
    motorcontroller = ArmController()
    motorcontroller.Motor_Init("/dev/ttyACM0", 921600, arm_type='right')
    controller = RetargetClass(arm_type='right')
    init_angle = np.zeros(8)
    init_angle[4] = 1.57
    # init_angle[5] = -1.57    ADW S UJ 12 
    motorcontroller.set_speed(0.5)
    # print(motor_cmd_positions)
    motorcontroller.motor_safe_control(init_angle)
    now_angle = np.zeros(8)
    now_angle[1:] = motorcontroller.Cmd_Limit_Clip(motorcontroller.get_all_positions())
    i = 0
    motor_cmd_positions =  [0,0,0,0,1.57,0,0,0]
 
    while True:
        while True:
            # 显示空白窗口
            cv2.imshow('Keyboard Input', 255 * np.ones((100, 100, 3), dtype=np.uint8))
            # 等待键盘输入，单位为毫秒
            key = cv2.waitKey(1) & 0xFF

            if key == ord('a'):
                target_pose = np.array([0., 0.05, 0., 0., 0., 0.])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = controller.get_retarget_joint_angle(delta_pose, init_angles=now_angle)
                motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle[1:] = motorcontroller.Cmd_Limit_Clip(motorcontroller.get_all_positions())

            if key == ord('d'):
                target_pose = np.array([0., -0.05, 0.0, 0., 0., 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = controller.get_retarget_joint_angle(delta_pose, init_angles=now_angle)
                motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle[1:] = motorcontroller.Cmd_Limit_Clip(motorcontroller.get_all_positions())

            if key == ord('w'):
                target_pose = np.array([0.05, 0, 0.0, 0., 0., 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = controller.get_retarget_joint_angle(delta_pose, init_angles=now_angle)
                motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle[1:] = motorcontroller.Cmd_Limit_Clip(motorcontroller.get_all_positions())

            if key == ord('s'):
                target_pose = np.array([-0.05, 0, 0.0, 0., 0., 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = controller.get_retarget_joint_angle(delta_pose, init_angles=now_angle)
                motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle[1:] = motorcontroller.Cmd_Limit_Clip(motorcontroller.get_all_positions())

            if key == ord('u'):
                target_pose = np.array([0.0, 0, 0.05, 0., 0., 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = controller.get_retarget_joint_angle(delta_pose, init_angles=now_angle)
                motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle[1:] = motorcontroller.Cmd_Limit_Clip(motorcontroller.get_all_positions())

            if key == ord('j'):
                target_pose = np.array([0, 0.0, -0.05, 0., 0., 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = controller.get_retarget_joint_angle(delta_pose, init_angles=now_angle)
                motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle[1:] = motorcontroller.Cmd_Limit_Clip(motorcontroller.get_all_positions())

            if key == ord('1'):
                target_pose = np.array([0, 0.0, 0, 0.2, 0., 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = controller.get_retarget_joint_angle(delta_pose, init_angles=now_angle)
                motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle[1:] = motorcontroller.Cmd_Limit_Clip(motorcontroller.get_all_positions())

            if key == ord('2'):
                target_pose = np.array([0, 0.0, 0, -0.2, 0., 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = controller.get_retarget_joint_angle(delta_pose, init_angles=now_angle)
                motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle[1:] = motorcontroller.Cmd_Limit_Clip(motorcontroller.get_all_positions())

            if key == ord('3'):
                target_pose = np.array([0, 0.0, 0, 0, 0.2, 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = controller.get_retarget_joint_angle(delta_pose, init_angles=now_angle)
                motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle[1:] = motorcontroller.Cmd_Limit_Clip(motorcontroller.get_all_positions())

            if key == ord('4'):
                target_pose = np.array([0, 0.0, 0, 0, -0.2, 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = controller.get_retarget_joint_angle(delta_pose, init_angles=now_angle)
                motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle[1:] = motorcontroller.Cmd_Limit_Clip(motorcontroller.get_all_positions())

            if key == ord('5'):
                target_pose = np.array([0, 0.0, 0, 0, 0.0, 0.2])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = controller.get_retarget_joint_angle(delta_pose, init_angles=now_angle)
                motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle[1:] = motorcontroller.Cmd_Limit_Clip(motorcontroller.get_all_positions())

            if key == ord('6'):
                target_pose = np.array([0, 0.0, 0, 0, 0.0, -0.2])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = controller.get_retarget_joint_angle(delta_pose, init_angles=now_angle)
                motorcontroller.motor_safe_control(motor_cmd_positions)
                now_angle[1:] = motorcontroller.Cmd_Limit_Clip(motorcontroller.get_all_positions())

            if key == ord('b'):
                motorcontroller.motor_safe_control(init_angle)
                controller.reset()

            if key == ord('v'):
                print(motor_cmd_positions)

            if key == 27:
                print("Exiting...")
                break



if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(demo_main())
    # loop.close()
    demo_main()
