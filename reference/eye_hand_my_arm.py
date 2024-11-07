# 导入必要的库
import cv2
import numpy as np
import pyrealsense2 as rs
import os
from scipy.spatial.transform import Rotation as R
import serial
from DM_Control.DM_CAN import MotorControl
from DM_Control.Robot_Control import ArmController
from arm_get import RetargetClass

class RoboticArmController:
    """
    机械臂控制类，整合RetargetClass和ArmController
    """
    def __init__(self, arm_type='left', com_port="COM4"):
        # 初始化串口连接
        self.serial_device = serial.Serial(com_port, 921600, timeout=0.5)
        self.motor_control = MotorControl(self.serial_device)
        
        # 初始化机械臂控制器
        self.arm_controller = ArmController(self.motor_control)
        self.arm_controller.Motor_Init(self.motor_control, arm_type)
        self.arm_controller.set_speed(0.8)
        
        # 初始化运动学控制器
        self.retarget_controller = RetargetClass(arm_type=arm_type)
        
        # 设置初始角度
        self.init_angle = np.zeros(8)
        self.init_angle[4] = -1.57 if arm_type == 'left' else 1.57
        self.current_angle = np.zeros(8)
        
        # 初始化位置
        self.arm_controller.motor_safe_control(self.init_angle)
        self.current_angle[1:] = self.arm_controller.Cmd_Limit_Clip(
            self.arm_controller.get_all_positions())

    def get_end_pose(self):
        """
        获取机械臂末端6D位姿
        返回: [x, y, z, rx, ry, rz]
        """
        # 获取当前关节角度
        current_angles = np.zeros(8)
        current_angles = self.arm_controller.get_all_positions()
        clipped_angles = self.arm_controller.Cmd_Limit_Clip(current_angles)
        
        # 使用RetargetClass计算末端位姿
        return self.retarget_controller.get_end_pose(clipped_angles)

    def reset_position(self):
        """
        重置机械臂到初始位置
        """
        self.arm_controller.motor_safe_control(self.init_angle)
        self.retarget_controller.reset()
        self.current_angle = np.zeros(8)
        self.current_angle[1:] = self.arm_controller.Cmd_Limit_Clip(
            self.arm_controller.get_all_positions())

    def cleanup(self):
        """
        清理资源
        """
        self.serial_device.close()

class EyeHandCalibrator:
    """
    眼手标定类
    """
    def __init__(self, robotic_arm):
        """
        初始化标定器
        :param robotic_arm: RoboticArmController实例
        """
        self.robotic_arm = robotic_arm
        
        # 初始化相机
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        # 获取相机内参
        profile = self.pipeline.get_active_profile()
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                     [0, intrinsics.fy, intrinsics.ppy],
                                     [0, 0, 1]])
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # 初始化AprilTag检测器
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.arucoDict, self.parameters)
        self.tag_size = 0.147  # AprilTag尺寸（米）

        # 标定数据存储
        self.step = 4  # 标定位姿数量
        self.R_t2c = []  # 标定板到相机的旋转
        self.T_t2c = []  # 标定板到相机的平移
        self.g2b_R_seq = []  # 机械臂末端到基座的旋转
        self.g2b_T_seq = []  # 机械臂末端到基座的平移
        
        # 创建数据保存目录
        self.output_dir1 = 'calibration_data/origin_img'
        self.output_dir2 = 'calibration_data/final_img'
        os.makedirs(self.output_dir1, exist_ok=True)
        os.makedirs(self.output_dir2, exist_ok=True)

    def collect_calibration_data(self):
        """收集标定数据"""
        frame_count = 0
        exit_loop = False

        while not exit_loop:
            # 获取相机图像
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 处理图像
            color_image = np.asanyarray(color_frame.get_data())
            color_frame_save = color_image.copy()
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # 检测AprilTag
            corners, ids, _ = self.detector.detectMarkers(gray)

            if ids is not None and len(ids) == 1:
                # 估计AprilTag位姿
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.tag_size, self.camera_matrix, self.dist_coeffs)
                
                # 绘制检测结果
                cv2.aruco.drawDetectedMarkers(color_image, corners)
                self.draw_axis(color_image, rvecs[0], tvecs[0])

                # 处理键盘输入
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    exit_loop = True
                elif key == ord('s'):
                    print("保存当前标定数据...")
                    
                    # 保存相机数据
                    self.R_t2c.append(rvecs[0])
                    self.T_t2c.append(tvecs[0].T)

                    # 获取机械臂位姿
                    pose = self.robotic_arm.get_end_pose()
                    print("当前机械臂位姿:", pose)
                    
                    # 转换并保存机械臂数据
                    self.g2b_R_seq.append([pose[3:6]])
                    self.g2b_T_seq.append([pose[0:3]])

                    # 保存图像
                    frame_count += 1
                    cv2.imwrite(os.path.join(self.output_dir1, f'origin_frame_{frame_count}.jpg'), 
                              color_frame_save)
                    cv2.imwrite(os.path.join(self.output_dir2, f'final_frame_{frame_count}.jpg'), 
                              color_image)

                    if len(self.R_t2c) == self.step:
                        exit_loop = True

            # 显示结果
            cv2.imshow('AprilTag Detection', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def calibrate(self):
        """执行眼手标定"""
        # 数据预处理和标定过程代码保持不变
        ...

    def cleanup(self):
        """清理资源"""
        self.pipeline.stop()
        cv2.destroyAllWindows()

def main():
    """
    主函数 - 使用示例
    """
    try:
        # 创建机械臂控制器
        robotic_arm = RoboticArmController(arm_type='left', com_port="COM4")
        
        # 创建标定器
        calibrator = EyeHandCalibrator(robotic_arm)

        print("开始眼手标定程序...")
        print("1. 请将AprilTag标定板放置在相机视野内")
        print("2. 移动机械臂到不同位置")
        print("3. 当看到清晰的AprilTag时，按's'保存当前位姿")
        print("4. 收集完所需数据后，程序将自动进行标定")
        print("5. 随时按'q'退出程序")
        
        # 收集标定数据
        calibrator.collect_calibration_data()
        
        # 执行标定
        R, t = calibrator.calibrate()
        
        # 打印结果
        print("\n标定结果:")
        print("旋转矩阵:\n", R)
        print("平移向量:\n", t)
        
        # 保存结果
        np.save('calibration_results/rotation_matrix.npy', R)
        np.save('calibration_results/translation_vector.npy', t)
        
    finally:
        # 清理资源
        calibrator.cleanup()
        robotic_arm.cleanup()

if __name__ == "__main__":
    main()
