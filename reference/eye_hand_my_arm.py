# 导入必要的库
import cv2
import numpy as np
import pyrealsense2 as rs
import os
from scipy.spatial.transform import Rotation as R
import serial
from DM_Control.DM_CAN import MotorControl
from DM_Control.Robot_Control import ArmController
from IK_solver import quat2Rot, compound_Rot_p_2_T, pose_to_homogeneous_matrix
from arm_get import RetargetClass


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([
        [marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, marker_size / 2, 0]
    ], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash


class RoboticArmController:
    """
    机械臂控制类，整合RetargetClass和ArmController
    """

    def __init__(self, arm_type='left', com_port="/dev/ACM0"):
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
        current_angles = np.zeros(7)
        clipped_angles = np.zeros(8)
        current_angles = self.arm_controller.get_all_positions()
        clipped_angles[1:] = self.arm_controller.Cmd_Limit_Clip(current_angles)

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
        self.tag_size = 0.080  # AprilTag尺寸（米）

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

        # 添加标定板到末端执行器的固定变换
        # 这个变换矩阵描述了从机械臂末端到AprilTag的固定变换
        # 需要根据实际测量值来设置
        self.tag_to_gripper = np.eye(4)  # 初始化为单位矩阵
        
        # 例如，如果标定板在末端执行器前方5cm，上方3cm的位置：
        self.tag_to_gripper[0:3, 3] = np.array([0.05, 0.0, 0.126])  # 平移部分
        # 如果标定板还有旋转，需要设置旋转矩阵部分
        # self.tag_to_gripper[0:3, 0:3] = R.from_euler('xyz', [rx, ry, rz]).as_matrix()

    def collect_calibration_data(self, robotic_arm):
        """收集标定数据"""

        print("collect_calibration_data")
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
            key = cv2.waitKey(1)

            if ids is not None and len(ids) == 1:
                # 估计AprilTag位姿
                rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
                    corners, self.tag_size, self.camera_matrix, self.dist_coeffs)
                # 绘制检测结果
                cv2.aruco.drawDetectedMarkers(color_image, corners)
                self.draw_axis(color_image, rvecs[0], tvecs[0])

                if key & 0xFF == ord('v'):
                    print("保存当前标定数据...")
                    # 保存相机数据
                    self.R_t2c.append(rvecs[0].T)
                    self.T_t2c.append(tvecs[0].T)

                    # 获取机械臂位姿
                    end_pose = self.robotic_arm.get_end_pose()
                    print("当前机械臂位姿:", end_pose)

                    # 创建末端执行器的齐次变换矩阵
                    gripper_matrix = pose_to_homogeneous_matrix(end_pose)
                    
                    # 考虑标定板相对于末端执行器的偏移
                    tag_matrix = gripper_matrix @ self.tag_to_gripper
                    
                    # 提取修正后的旋转和平移
                    corrected_R = tag_matrix[0:3, 0:3]
                    corrected_t = tag_matrix[0:3, 3]

                    # 转换并保存机械臂数据（使用修正后的值）
                    rot = R.from_matrix(corrected_R)
                    self.g2b_R_seq.append([rot.as_rotvec()])  # 保存为旋转向量
                    self.g2b_T_seq.append([corrected_t])

                    # 保存图像
                    frame_count += 1
                    cv2.imwrite(os.path.join(self.output_dir1, f'origin_frame_{frame_count}.jpg'),
                                color_frame_save)
                    cv2.imwrite(os.path.join(self.output_dir2, f'final_frame_{frame_count}.jpg'),
                                color_image)

                    # if len(self.R_t2c) == self.step:
                    #     exit_loop = True

            # 显示结果
            cv2.imshow('AprilTag Detection', color_image)
            if key & 0xFF == ord('q'):
                break
            elif key == ord('a'):
                print("111")
                target_pose = np.array([0., 0.05, 0., 0., 0., 0.])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = robotic_arm.retarget_controller.get_retarget_joint_angle(delta_pose,
                                                                                               init_angles=robotic_arm.current_angle)
                robotic_arm.arm_controller.motor_safe_control(motor_cmd_positions)
                robotic_arm.current_angle[1:] = robotic_arm.arm_controller.Cmd_Limit_Clip(
                    robotic_arm.arm_controller.get_all_positions())
            elif key == ord('d'):
                target_pose = np.array([0., -0.05, 0.0, 0., 0., 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = robotic_arm.retarget_controller.get_retarget_joint_angle(delta_pose,
                                                                                               init_angles=robotic_arm.current_angle)
                robotic_arm.arm_controller.motor_safe_control(motor_cmd_positions)
                robotic_arm.current_angle[1:] = robotic_arm.arm_controller.Cmd_Limit_Clip(
                    robotic_arm.arm_controller.get_all_positions())
            elif key == ord('w'):
                target_pose = np.array([0.05, 0, 0.0, 0., 0., 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = robotic_arm.retarget_controller.get_retarget_joint_angle(delta_pose,
                                                                                               init_angles=robotic_arm.current_angle)
                robotic_arm.arm_controller.motor_safe_control(motor_cmd_positions)
                robotic_arm.current_angle[1:] = robotic_arm.arm_controller.Cmd_Limit_Clip(
                    robotic_arm.arm_controller.get_all_positions())
            elif key == ord('s'):
                target_pose = np.array([-0.05, 0, 0.0, 0., 0., 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = robotic_arm.retarget_controller.get_retarget_joint_angle(delta_pose,
                                                                                               init_angles=robotic_arm.current_angle)
                robotic_arm.arm_controller.motor_safe_control(motor_cmd_positions)
                robotic_arm.current_angle[1:] = robotic_arm.arm_controller.Cmd_Limit_Clip(
                    robotic_arm.arm_controller.get_all_positions())
            elif key == ord('u'):
                target_pose = np.array([0.0, 0, 0.05, 0., 0., 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = robotic_arm.retarget_controller.get_retarget_joint_angle(delta_pose,
                                                                                               init_angles=robotic_arm.current_angle)
                robotic_arm.arm_controller.motor_safe_control(motor_cmd_positions)
                robotic_arm.current_angle[1:] = robotic_arm.arm_controller.Cmd_Limit_Clip(
                    robotic_arm.arm_controller.get_all_positions())
            elif key == ord('j'):
                target_pose = np.array([0, 0.0, -0.05, 0., 0., 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = robotic_arm.retarget_controller.get_retarget_joint_angle(delta_pose,
                                                                                               init_angles=robotic_arm.current_angle)
                robotic_arm.arm_controller.motor_safe_control(motor_cmd_positions)
                robotic_arm.current_angle[1:] = robotic_arm.arm_controller.Cmd_Limit_Clip(
                    robotic_arm.arm_controller.get_all_positions())
            elif key == ord('1'):
                target_pose = np.array([0, 0.0, 0, 0.2, 0., 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = robotic_arm.retarget_controller.get_retarget_joint_angle(delta_pose,
                                                                                               init_angles=robotic_arm.current_angle)
                robotic_arm.arm_controller.motor_safe_control(motor_cmd_positions)
                robotic_arm.current_angle[1:] = robotic_arm.arm_controller.Cmd_Limit_Clip(
                    robotic_arm.arm_controller.get_all_positions())
            elif key == ord('2'):
                target_pose = np.array([0, 0.0, 0, -0.2, 0., 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = robotic_arm.retarget_controller.get_retarget_joint_angle(delta_pose,
                                                                                               init_angles=robotic_arm.current_angle)
                robotic_arm.arm_controller.motor_safe_control(motor_cmd_positions)
                robotic_arm.current_angle[1:] = robotic_arm.arm_controller.Cmd_Limit_Clip(
                    robotic_arm.arm_controller.get_all_positions())
            elif key == ord('3'):
                target_pose = np.array([0, 0.0, 0, 0, 0.2, 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = robotic_arm.retarget_controller.get_retarget_joint_angle(delta_pose,
                                                                                               init_angles=robotic_arm.current_angle)
                robotic_arm.arm_controller.motor_safe_control(motor_cmd_positions)
                robotic_arm.current_angle[1:] = robotic_arm.arm_controller.Cmd_Limit_Clip(
                    robotic_arm.arm_controller.get_all_positions())
            elif key == ord('4'):
                target_pose = np.array([0, 0.0, 0, 0, -0.2, 0.0])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = robotic_arm.retarget_controller.get_retarget_joint_angle(delta_pose,
                                                                                               init_angles=robotic_arm.current_angle)
                robotic_arm.arm_controller.motor_safe_control(motor_cmd_positions)
                robotic_arm.current_angle[1:] = robotic_arm.arm_controller.Cmd_Limit_Clip(
                    robotic_arm.arm_controller.get_all_positions())
            elif key == ord('5'):
                target_pose = np.array([0, 0.0, 0, 0, 0.0, 0.2])
                delta_pose = pose_to_homogeneous_matrix(target_pose)
                motor_cmd_positions = robotic_arm.retarget_controller.get_retarget_joint_angle(delta_pose,
                                                                                               init_angles=robotic_arm.current_angle)
                robotic_arm.arm_controller.motor_safe_control(motor_cmd_positions)
                robotic_arm.current_angle[1:] = robotic_arm.arm_controller.Cmd_Limit_Clip(
                    robotic_arm.arm_controller.get_all_positions())
            elif key == ord('6'):
                target_pose = np.array([0, 0.0, 0, 0, 0.0, -0.2])
                motor_cmd_positions = robotic_arm.retarget_controller.get_retarget_joint_angle(delta_pose,
                                                                                               init_angles=robotic_arm.current_angle)
                robotic_arm.arm_controller.motor_safe_control(motor_cmd_positions)
                robotic_arm.current_angle[1:] = robotic_arm.arm_controller.Cmd_Limit_Clip(
                    robotic_arm.arm_controller.get_all_positions())
            elif key == ord('b'):
                robotic_arm.arm_controller.motor_safe_control(robotic_arm.init_angle)
                robotic_arm.retarget_controller.reset()

    def calibrate(self):
        """
        执行眼手标定
        返回: (R, t) 相机到机械臂末端的旋转矩阵和平移向量
        """
        if len(self.R_t2c) < 3:
            print("数据点不足，无法进行标定")
            return None, None

        try:
            # 转换数据格式
            R_gripper2base = np.array(self.g2b_R_seq).astype(np.float64)
            t_gripper2base = np.array(self.g2b_T_seq).astype(np.float64)
            R_target2cam = np.array(self.R_t2c).astype(np.float64)
            t_target2cam = np.array(self.T_t2c).astype(np.float64)

            print("数据形状检查:")
            print(f"R_gripper2base shape: {R_gripper2base.shape}")
            print(f"t_gripper2base shape: {t_gripper2base.shape}")
            print(f"R_target2cam shape: {R_target2cam.shape}")
            print(f"t_target2cam shape: {t_target2cam.shape}")

            # 将旋转向量转换为旋转矩阵
            R_g2b_matrices = []
            for rvec in R_gripper2base:
                rot = R.from_rotvec(rvec[0])
                R_g2b_matrices.append(rot.as_matrix())
            R_g2b_matrices = np.array(R_g2b_matrices)

            R_t2c_matrices = []
            for rvec in R_target2cam:
                R_mat, _ = cv2.Rodrigues(rvec)
                R_t2c_matrices.append(R_mat)
            R_t2c_matrices = np.array(R_t2c_matrices)
            print("数据形状检查2:")
            print(f"R_gripper2base shape: {R_g2b_matrices[0].shape}")
            print(f"t_gripper2base shape: {t_gripper2base[0].shape}")
            print(f"R_target2cam shape: {R_t2c_matrices[0].shape}")
            print(f"t_target2cam shape: {t_target2cam[0].shape}")
            # 执行标定
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_g2b_matrices,
                t_gripper2base,
                R_t2c_matrices,
                t_target2cam,
                method=cv2.CALIB_HAND_EYE_PARK
            )

            print("\n标定完成!")
            print("使用的数据点数量:", len(self.R_t2c))
            print("标定方法: CALIB_HAND_EYE_PARK")

            # 计算重投影误差
            # error = self.calculate_reprojection_error(
            #     R_cam2gripper, t_cam2gripper,
            #     R_g2b_matrices, t_gripper2base,
            #     R_t2c_matrices, t_target2cam
            # )
            # print(f"重投影误差: {error}")

            # 标定完成后，需要考虑将结果转换回机械臂末端坐标系
            if R_cam2gripper is not None:
                # 创建相机到标定板的变换矩阵
                H_cam2tag = np.eye(4)
                H_cam2tag[:3, :3] = R_cam2gripper
                H_cam2tag[:3, 3] = t_cam2gripper.flatten()
                
                # 计算相机到机械臂末端的变换
                H_cam2gripper = H_cam2tag @ np.linalg.inv(self.tag_to_gripper)
                
                # 提取最终的旋转矩阵和平移向量
                R_final = H_cam2gripper[:3, :3]
                t_final = H_cam2gripper[:3, 3].reshape(3, 1)
                
                return R_final, t_final

            return R_cam2gripper, t_cam2gripper

        except Exception as e:
            print(f"标定过程出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def calculate_reprojection_error(self, R_cam2gripper, t_cam2gripper,
                                     R_g2b, t_g2b, R_t2c, t_t2c):
        """
        计算标定结果的重投影误差
        """
        error = 0
        n_points = len(R_g2b)

        for i in range(n_points):
            # 计算预测的标定板位置
            H_pred = np.eye(4)
            H_pred[:3, :3] = R_g2b[i] @ R_cam2gripper
            H_pred[:3, 3] = (R_g2b[i] @ t_cam2gripper + t_g2b[i].flatten())

            # 计算实际的标定板位置
            H_actual = np.eye(4)
            H_actual[:3, :3] = R_t2c[i]
            H_actual[:3, 3] = t_t2c[i].flatten()

            # 计算误差
            error += np.linalg.norm(H_pred[:3, 3] - H_actual[:3, 3])

        return error / n_points

    def draw_axis(self, img, rvec, tvec):
        """
        在图像上绘制3D坐标轴
        """
        axis_length = 0.1  # 坐标轴长度（米）
        imgpts, _ = cv2.projectPoints(
            np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]),
            rvec, tvec, self.camera_matrix, self.dist_coeffs
        )

        origin = tuple(map(int, imgpts[0].ravel()))
        x_end = tuple(map(int, imgpts[1].ravel()))
        y_end = tuple(map(int, imgpts[2].ravel()))
        z_end = tuple(map(int, imgpts[3].ravel()))

        cv2.line(img, origin, x_end, (0, 0, 255), 3)  # X轴为红色
        cv2.line(img, origin, y_end, (0, 255, 0), 3)  # Y轴为绿色
        cv2.line(img, origin, z_end, (255, 0, 0), 3)  # Z轴为蓝色

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
        robotic_arm = RoboticArmController(arm_type='left', com_port="/dev/ttyACM0")

        # 创建标定器
        calibrator = EyeHandCalibrator(robotic_arm)

        print("开始眼手标定程序...")
        print("1. 请将AprilTag标定板放置在相机视野内")
        print("2. 移动机械臂到不同位置")
        print("3. 当看到清晰的AprilTag时，按's'保存当前位姿")
        print("4. 收集完所需数据后，程序将自动进行标定")
        print("5. 随时按'q'退出程序")

        # 收集标定数据
        calibrator.collect_calibration_data(robotic_arm)

        # 执行标定
        R, t = calibrator.calibrate()

        # 打印结果
        print("\n标定结果:")
        print("旋转矩阵:\n", R)
        print("平移向量:\n", t)

        # 保存结果
        # np.save('calibration_results/rotation_matrix.npy', R)
        # np.save('calibration_results/translation_vector.npy', t)

    finally:
        # 清理资源
        calibrator.cleanup()
        robotic_arm.cleanup()


if __name__ == "__main__":
    main()
