# 导入必要的库
import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_control      # UR机器人控制接口
import rtde_receive      # UR机器人数据接收接口
import os
from scipy.spatial.transform import Rotation as R  # 用于处理旋转矩阵、欧拉角等

# 初始化realsense相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置彩色图像流
pipeline.start(config)

# 设置标定所需的图像数量
step = 4

# 获取相机内参
profile = pipeline.get_active_profile()
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                         [0, intrinsics.fy, intrinsics.ppy],
                         [0, 0, 1]])
print("camera_matrix", camera_matrix)

# 设置畸变系数（假设相机已经校正，因此设为0）
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# 创建AprilTag检测器
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)  # 使用36h11编码的AprilTag
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, parameters)
tag_size = 0.147  # AprilTag的实际尺寸（米）

# 连接UR机器人
robot = rtde_control.RTDEControlInterface("192.168.4.101")
info = rtde_receive.RTDEReceiveInterface("192.168.4.101")
# arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

def calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True):
    """
    执行眼手标定计算
    :param R_gripper2base: 机器人末端到基座的旋转矩阵序列
    :param t_gripper2base: 机器人末端到基座的平移向量序列
    :param R_target2cam: 标定板到相机的旋转矩阵序列
    :param t_target2cam: 标定板到相机的平移向量序列
    :param eye_to_hand: 是否为眼在手外配置
    :return: 相机到机器人末端的旋转矩阵和平移向量
    """
    if eye_to_hand:
        # 对于眼在手外配置，需要将gripper2base转换为base2gripper
        R_base2gripper, t_base2gripper = [], []

        print("R_gripper2base", R_gripper2base.shape)
        print("t_gripper2base", t_gripper2base.shape)

        # 计算逆变换
        for R, t in zip(R_gripper2base, t_gripper2base):
            print("R",R.shape)
            print("T",t.shape)
            R_b2g = R.T  # 旋转矩阵的逆是其转置
            t = t.T
            t_b2g = -R_b2g @ t  # 计算平移向量的逆变换
            R_base2gripper.append(R_b2g)
            t_base2gripper.append(t_b2g)

        # 更新变换矩阵
        R_gripper2base = np.array(R_base2gripper)
        t_gripper2base = np.array(t_base2gripper)
        R_target2cam = np.array(R_target2cam)
        t_target2cam = np.array(t_target2cam)

    # 使用OpenCV的calibrateHandEye函数进行标定
    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
    )

    return R, t

def rodrigues_to_matrix(rvecs):
    """
    将旋转向量转换为旋转矩阵
    :param rvecs: 旋转向量列表
    :return: 旋转矩阵列表
    """
    return [cv2.Rodrigues(rvec)[0] for rvec in rvecs]

# 定义绘制3D坐标轴的函数
def draw_axis(img, rvec, tvec, camera_matrix, dist_coeffs, length=5):
    """
    在图像上绘制3D坐标轴
    :param img: 输入图像
    :param rvec: 旋转向量
    :param tvec: 平移向量
    :param camera_matrix: 相机内参矩阵
    :param dist_coeffs: 畸变系数
    :param length: 坐标轴长度
    :return: 绘制了坐标轴的图像
    """
    # 定义坐标轴的端点
    axis = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).reshape(-1, 3)

    # 将3D点投影到图像平面
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # 绘制坐标轴（红色X轴，绿色Y轴，蓝色Z轴）
    cv2.arrowedLine(img, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 2)  # X轴为红色
    cv2.arrowedLine(img, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 2)  # Y轴为绿色
    cv2.arrowedLine(img, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 2)  # Z轴为蓝色
    return img

def compute_transform_properties(rotation_matrix, translation_vector):
    # 创建旋转对象
    rotation = R.from_matrix(rotation_matrix)

    # 计算欧拉角（采用XYZ顺序）
    euler_angles = rotation.as_euler('xyz', degrees=True)

    # 计算四元数
    quaternion = rotation.as_quat()  # 输出顺序为 x, y, z, w

    # 计算旋转矢量
    rotvec = rotation.as_rotvec()

    return euler_angles, quaternion, rotvec

R_t2c = []
T_t2c = []
R_t2cPnP = []
T_t2cPnP = []
UR_TCP = []
# info = rtde_receive.RTDEReceiveInterface("192.168.4.100")

g2b_R_seq = []
g2b_T_seq = []

# 初始化帧计数器
frame_count = 0

output_dir1 = r'E:\eye_hand\origin_img'
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)

output_dir2 = r'E:\eye_hand\final_img'
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

exit_loop = False
while not exit_loop:

    # 读取realsense相机帧
    # print("------------------")
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("Frame not available.")
        continue


    color_image = np.asanyarray(color_frame.get_data())
    color_frame = color_image.copy()
    # 转换图像为灰度
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # 检测AprilTag
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(ids)==1:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, tag_size, camera_matrix, dist_coeffs)
        rvecPnP = None
        tvecPnP = None
        for i in range(len(ids)):
            # 绘制AprilTag的包围框
            cv2.aruco.drawDetectedMarkers(color_image, corners)

            # 绘制坐标轴
            color_image = draw_axis(color_image, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)

            # 将3D点投影到图像上
            imgpts, _ = cv2.projectPoints(np.array([(-tag_size / 2, -tag_size/2, 0),
                                                    (tag_size/ 2, -tag_size/2, 0),
                                                    (tag_size / 2, tag_size / 2, 0),
                                                    (-tag_size / 2, tag_size / 2, 0)]),
                                          rvecs[i], tvecs[i], camera_matrix, dist_coeffs)

            imgpts = np.int32(imgpts).reshape(-1, 2).tolist()
        #    print("imgpts", imgpts)
            for corner in imgpts:
                # print( tuple(corner[0]))
                cv2.circle(color_image, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), -1)

            # cv2.polylines(color_image, [imgpts[:4]], True, (0, 255, 0), thickness=2, lineType=cv2.LINE_DASH)
            # cv2.drawContours(color_image, [imgpts[:4]], -1, (0, 255, 0), thickness=2)
            # cv2.drawContours(color_image, [imgpts[:4]], -1, (0, 255, 0), thickness=2)
            _, rvecPnP, tvecPnP = cv2.solvePnP(
                tag_size * np.array([[-0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0]]),
                corners[i], camera_matrix, dist_coeffs)
         #   print("rvecPnP, tvecPnP", rvecPnP, tvecPnP)
         #   print("rvecs[i], tvecs[i]", rvecs[i], tvecs[i])
            imgptsPnP, _ = cv2.projectPoints(np.array([(-tag_size / 2, -tag_size/2, 0),
                                                    (tag_size/ 2, -tag_size/2, 0),
                                                    (tag_size / 2, tag_size / 2, 0),
                                                    (-tag_size / 2, tag_size / 2, 0)]),
                                          rvecPnP, tvecPnP, camera_matrix, dist_coeffs)
            imgptsPnP = np.int32(imgptsPnP).reshape(-1, 2).tolist()
      #      print("imgptsPnP", imgptsPnP)
            for corner in imgptsPnP:
                # print( tuple(corner[0]))
                cv2.circle(color_image, (int(corner[0]), int(corner[1])), 8, (255, 0, 0), 2)
                cv2.circle(color_image, (320,240), 5, (255, 255, 255), -1)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            exit_loop = True
        elif key == ord('s'):
            # 保存图片
            print("Saving data...")
            R_t2c.append(rvecs[0])
            T_t2c.append(tvecs[0].T)

            R_t2cPnP.append(rvecPnP)
            T_t2cPnP.append(tvecPnP)

            frame_count += 1
            # 构建输出文件名
            #先不保存，以免破坏已有数据集
            output_filename1 = os.path.join(output_dir1, f'origin_frame_{frame_count}.jpg')
            cv2.imwrite(output_filename1, color_frame)  # 保存帧
            output_filename2 = os.path.join(output_dir2, f'final_frame_{frame_count}.jpg')
            cv2.imwrite(output_filename2, color_image)  # 保存帧
            # UR_TCP.append(info.getActualTCPPose())
            pose = info.getActualTCPPose()

            print("getActualTCPPose", pose)
            g2b_R_seq.append([pose[3:6]])
            g2b_T_seq.append([pose[0:3]])
            print("R_t2c", R_t2c[-1])
            print("T_t2c", T_t2c[-1])

            # print("R_t2cPnP", R_t2cPnP[-1])
            # print("T_t2cPnP", T_t2cPnP[-1])

            print("g2b_R_seq", g2b_R_seq[-1])
            print("g2b_T_seq", g2b_T_seq[-1])
            # UR_pos = [1,1,1, 1,0,0] # xyz. rx, ry, rz (旋转矢量) T_g2b, R_g2b

            if R_t2c.__len__() == step:
                exit_loop = True


    # 显示结果
    cv2.imshow('AprilTag Detection', color_image)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break


# #机器人抓取器（或手臂末端）相对于机器人基座的旋转矩阵列表
R_gripper2base = np.array(g2b_R_seq).astype(np.float64)
#机器人抓取器（或手臂末端）相对于机器人基座的平移向量列表
t_gripper2base = np.array(g2b_T_seq).astype(np.float64)
#标定目标（如标定板）相对于摄像头的旋转矩阵列表
R_target2cam =np.array(R_t2c).astype(np.float64)
#标定目标（如标定板）相对于摄像头的平移向量列表
t_target2cam =np.array(T_t2c).astype(np.float64)


print(R_gripper2base, type(R_gripper2base), R_gripper2base.shape)
print(t_gripper2base, type(t_gripper2base), t_gripper2base.shape)
print(R_target2cam, type(R_target2cam), R_target2cam.shape)
print(t_target2cam, type(t_target2cam), t_target2cam.shape)

R_g2b_list = []
t_g2b_list = []
R_t2c_list = []
t_t2c_list = []


for i in range(step):
    print("R_gripper2base[i,0,:]",R_gripper2base[i,:,:].shape)
    R_g2b_list.append(rodrigues_to_matrix(R_gripper2base[i,:,:]))
    t_g2b_list.append(t_gripper2base[i,:,:])
    R_t2c_list.append(rodrigues_to_matrix(R_target2cam[i,:,:]))
    t_t2c_list.append(t_target2cam[i,:,:])
R_g2b_list = np.array(R_g2b_list)[:,0,:,:]
t_g2b_list = np.array(t_g2b_list)
R_t2c_list = np.array(R_t2c_list)[:,0,:,:]
t_t2c_list = np.array(t_t2c_list)

print(R_g2b_list, type(R_g2b_list), R_g2b_list.shape)
print(t_g2b_list, type(t_g2b_list), t_g2b_list.shape)
print(R_t2c_list, type(R_t2c_list), R_t2c_list.shape)
print(t_t2c_list, type(t_t2c_list), t_t2c_list.shape)

if len(R_gripper2base) == len(t_gripper2base) == len(R_target2cam) == len(t_target2cam):

    R_target2cam_matrices = rodrigues_to_matrix(R_target2cam.reshape(-1, 3, 1))
    R_gripper2base_matrices = rodrigues_to_matrix(R_gripper2base.reshape(-1, 3, 1))
    R_cam2gripper,t_cam2gripper = calibrate_eye_hand(R_g2b_list,t_g2b_list,R_t2c_list,t_t2c_list,eye_to_hand=True)
    for i in range(step):
        print("Hand-eye calibration successful.")
    print("R_cam2gripper:::",R_cam2gripper)
    print("t_cam2gripper:::",t_cam2gripper)
else:
    print("Error: Array sizes do not match.")

# 释放资源
pipeline.stop()
cv2.destroyAllWindows()
