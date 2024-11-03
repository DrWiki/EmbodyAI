import argparse
from opt import *
import numpy as np
import torch
import open3d as o3d
from graspnet_baseline_main.models.graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup, Grasp
from graspnet_baseline_main.utils.collision_detector import ModelFreeCollisionDetector
import sys
import json

def backproject(depth_cv, intrinsic_matrix, return_finite_depth=True, return_selection=False):
    """
    将深度图反投影到3D空间，用于生成点云
    :param depth_cv: OpenCV格式的深度图
    :param intrinsic_matrix: 相机内参矩阵，包含焦距和主点坐标
    :param return_finite_depth: 是否只返回有效深度点
    :param return_selection: 是否返回有效点的选择掩码
    :return: 3D点云坐标，可选返回选择掩码
    """
    # 将深度图转换为float32类型
    depth = depth_cv.astype(np.float32, copy=True)

    # 获取相机内参矩阵的逆矩阵，用于反投影计算
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # 获取图像尺寸
    width = depth.shape[1]
    height = depth.shape[0]

    # 构建像素坐标网格（u,v坐标）
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    # 将像素坐标转换为齐次坐标
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # 使用内参矩阵的逆矩阵进行反投影
    R = np.dot(Kinv, x2d.transpose())

    # 计算3D点坐标：将深度值与反投影射线相乘
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    X = np.array(X).transpose()

    # 如果需要，只保留有效深度点（非无穷大的点）
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection

    return X

def get_grasps_camera(grasp):
    """
    将抓取对象转换为相机坐标系下的变换矩阵
    :param grasp: 抓取对象，包含位置和旋转信息
    :return: 4x4的齐次变换矩阵
    """
    # 获取抓取位置和旋转矩阵
    translation = grasp.translation  # 平移向量
    rot = grasp.rotation_matrix     # 旋转矩阵

    # 构建4x4齐次变换矩阵
    grasp_trans = np.eye(4)        # 初始化为单位矩阵
    grasp_trans[:3, :3] = rot      # 设置旋转部分
    grasp_trans[:3, -1] = translation  # 设置平移部分

    return grasp_trans

class GraspNetModule():
    """
    抓取检测模块主类
    """
    def __init__(self, cfg, device) -> None:
        """
        初始化抓取检测模块
        :param cfg: 配置参数，包含网络参数、路径等设置
        :param device: 运行设备（CPU/GPU）
        """
        self.cfg = cfg.GRASPNET_BASELINE
        self.device = device

        # 初始化抓取网络，设置网络参数
        self.net = GraspNet(
            input_feature_dim=0,     # 输入特征维度
            num_view=self.cfg.num_view,  # 视角数量
            num_angle=12,            # 角度采样数
            num_depth=4,             # 深度层级数
            cylinder_radius=0.05,    # 抓取圆柱体半径
            hmin=-0.02,             # 最小高度
            hmax_list=[0.01,0.02,0.03,0.04],  # 最大高度列表
            is_training=False        # 设置为推理模式
        )
        self.net.to(device)  # 将网络移至指定设备

        # 加载预训练模型权重
        checkpoint = torch.load(self.cfg.checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()  # 设置为评估模式
        print("成功加载GraspNet模型!")

        # 加载相机到机器人基座的标定结果
        self.R_cam2base = np.load("/home/ssr/transparent/calibrate/R_cam2gripper.npy")  # 旋转矩阵
        self.T_cam2base = np.load("/home/ssr/transparent/calibrate/t_cam2gripper.npy")  # 平移向量
        self.trans_camera2base = np.eye(4)
        self.trans_camera2base[:3, :3] = self.R_cam2base
        self.trans_camera2base[:3, 3] = self.T_cam2base[:, 0]
        self.trans_base2camera = np.linalg.inv(self.trans_camera2base)

        # Rotation matrix for 45 degrees clockwise
        theta = np.deg2rad(45)
        R_base2world = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        # Create the 4x4 transformation matrix from base to world
        self.trans_base2world = np.eye(4)
        self.trans_base2world[:3, :3] = R_base2world

        self.trans_world2base = np.linalg.inv(self.trans_base2world)

        self.trans_tool = np.array([
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],

        ])



    def generate_pc(self, depth, image, K, mask=None):
        """
        生成点云数据
        :param depth: 深度图，包含场景中每个像素的深度值
        :param image: RGB图像，用于获取点云颜色信息
        :param K: 相机内参矩阵
        :param mask: 可选的掩码，用于过滤不需要的区域
        :return: (点云坐标数组, 对应的颜色数组)
        """
        # 如果提供了掩码，将被掩码的深度值设为无效
        if mask is not None:
            depth[mask] = np.nan
        
        # 生成场景点云，同时获取有效点的选择掩码
        pc, selection = backproject(depth,
                                  K,
                                  return_finite_depth=True,
                                  return_selection=True)
        
        # 处理点云对应的颜色信息
        pc_colors = image.copy()
        pc_colors = np.reshape(pc_colors, [-1, 3])  # 展平为Nx3数组
        pc_colors = pc_colors[selection, :]  # 只保留有效点的颜色
        
        return pc, pc_colors

    def process_data(self, pc, pc_colors):
        """
        处理点云数据，进行采样和格式转换
        :param pc: 原始点云坐标
        :param pc_colors: 原始点云颜色
        :return: (处理后的数据字典, 点云对象)
        """
        # 将颜色值归一化到[0,1]范围
        pc_colors = pc_colors / 255.0
        cloud_masked = pc
        color_masked = pc_colors

        # 采样点云，确保点数符合要求
        if len(cloud_masked) >= self.cfg.num_point:
            # 如果点数过多，随机采样到指定数量
            idxs = np.random.choice(len(cloud_masked), self.cfg.num_point, replace=False)
        else:
            # 如果点数不足，进行重复采样
            idxs1 = np.arange(len(cloud_masked))  # 保留所有原始点
            idxs2 = np.random.choice(len(cloud_masked), self.cfg.num_point-len(cloud_masked), replace=True)  # 重复采样补充
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        
        # 获取采样后的点云和颜色
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # 转换为Open3D点云格式，用于可视化
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

        # 准备网络输入数据
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))  # 添加batch维度
        cloud_sampled = cloud_sampled.to(self.device)  # 移至指定设备
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud

    def get_grasps(self, end_points):
        """
        使用神经网络预测抓取位姿
        :param end_points: 预处理后的点云数据字典，包含：
            - point_clouds: 采样后的点云坐标
            - cloud_colors: 对应的颜色信息
        :return: GraspGroup对象，包含多个预测的抓取位姿
        """
        with torch.no_grad():
            # 通过网络前向传播获取预测结果
            ret = self.net(end_points)
            # 解码网络输出，生成具体的抓取参数
            grasp_preds = pred_decode(ret)
            
            # 将预测结果转换为GraspGroup对象
            gg = GraspGroup().from_numpy(grasp_preds)
            
            # 根据分数对抓取进行排序
            gg.sort_by_score()
            # 移除重复的抓取
            gg.nms()
            # 移除低分抓取
            gg = gg.remove_invisible(thresh=0.9)
            
            return gg

    def collision_detection(self, gg, cloud):
        """
        进行碰撞检测，过滤掉可能发生碰撞的抓取
        :param gg: GraspGroup对象，包含候选抓取
        :param cloud: 场景点云数组
        :return: 经过碰撞检测后的GraspGroup对象
        """
        # 初始化无模型碰撞检测器
        collision_detector = ModelFreeCollisionDetector(cloud)
        # 检测每个抓取是否会发生碰撞
        collision_mask = collision_detector.detect(gg, approach_dist=0.05, collision_thresh=self.cfg.collision_thresh)
        # 返回未发生碰撞的抓取
        gg = gg[~collision_mask]
        return gg

    def grasp2pixel(self, grasp_poses, K):
        grasp_points = []
        for grasp in grasp_poses['grasp_camera']:
            # Reshape and accumulate grasp points
            grasp_point = grasp[:, -1].reshape((-1, 4))
            grasp_points.append(grasp_point)

        # Concatenate all grasp points into a single array
        grasp_points = np.concatenate(grasp_points, axis=0)
        grasp_points = grasp_points[:, :3] # (N, 3)
        grasp_points = grasp_points / grasp_points[:, 2:3]

        pixel_coordinate = K @ grasp_points.T
        pixel_coordinate = pixel_coordinate.T
        pixel_coordinate = pixel_coordinate[:, :2]

        return pixel_coordinate

    def filter_obj_grasps(self, grasp_poses, obj_mask, K):
        """
        根据物体掩码过滤抓取位姿
        :param grasp_poses: 抓取位姿字典，包含分数和位姿信息
        :param obj_mask: 目标物体的掩码图像
        :param K: 相机内参矩阵
        :return: (过滤掩码, 过滤后的抓取位姿)
        """
        # 初始化过滤掩码
        filter_mask = np.zeros(len(grasp_poses['score']), dtype=bool)
        
        # 遍历所有抓取位姿
        for i in range(len(grasp_poses['score'])):
            # 将抓取位置投影到图像平面
            pixel_x, pixel_y = self.grasp2pixel(grasp_poses['grasp_camera'][i], K)
            
            # 检查投影点是否在图像范围内
            if 0 <= pixel_x < obj_mask.shape[1] and 0 <= pixel_y < obj_mask.shape[0]:
                # 检查投影点是否在物体掩码内
                if obj_mask[pixel_y, pixel_x]:
                    filter_mask[i] = True
        
        # 更新抓取位姿列表
        grasp_poses['score'] = list(np.array(grasp_poses['score'])[filter_mask])
        grasp_poses['grasp_camera'] = list(np.array(grasp_poses['grasp_camera'])[filter_mask])
        grasp_poses['grasp_base'] = list(np.array(grasp_poses['grasp_base'])[filter_mask])
        
        return filter_mask, grasp_poses

    def filer_direction_grasps(self, grasp_poses):
        """
        根据抓取方向过滤抓取位姿，确保抓取方向合理
        :param grasp_poses: 抓取位姿字典
        :return: (过滤掩码, 过滤后的抓取位姿)
        """
        # 初始化过滤掩码
        filter_mask = np.zeros(len(grasp_poses['score']), dtype=bool)
        
        # 遍历所有抓取位姿
        for i in range(len(grasp_poses['score'])):
            # 获取抓取位姿在基座坐标系下的表示
            grasp_base = grasp_poses['grasp_base'][i]
            # 获取抓取方向（z轴方向）
            direction = grasp_base[:3, 2]
            
            # 检查抓取方向是否满足要求（与垂直方向的夹角）
            angle = np.arccos(np.dot(direction, np.array([0, 0, 1])))
            if angle < np.pi / 3:  # 夹角小于60度
                filter_mask[i] = True
        
        # 更新抓取位姿列表
        grasp_poses['score'] = list(np.array(grasp_poses['score'])[filter_mask])
        grasp_poses['grasp_camera'] = list(np.array(grasp_poses['grasp_camera'])[filter_mask])
        grasp_poses['grasp_base'] = list(np.array(grasp_poses['grasp_base'])[filter_mask])
        
        return filter_mask, grasp_poses

    def find_max_score(self, grasp_poses):
        """
        找出得分最高的抓取位姿
        :param grasp_poses: 抓取位姿字典
        :return: 得分最高的抓取位姿的索引
        """
        scores = np.array(grasp_poses['score'])
        if len(scores) == 0:
            return None
        return np.argmax(scores)

    def vis_grasps(self, gg, cloud, num_grasp_vis=None):
        """
        可视化抓取位姿
        :param gg: GraspGroup对象
        :param cloud: 场景点云
        :param num_grasp_vis: 可视化的抓取数量，None表示全部显示
        """
        # 将抓取转换为Open3D可视化对象
        grippers = gg.to_open3d_geometry_list()
        # 如果指定了显示数量，则只显示部分抓取
        if num_grasp_vis is not None:
            grippers = grippers[:num_grasp_vis]
        # 使用Open3D进行可视化
        o3d.visualization.draw_geometries([cloud, *grippers])

            

    def generate_grasps(self, data, visualize=False, num_grasps=None):
        """
        生成抓取位姿的主函数
        :param data: 输入数据字典，包含：
            - pc: 场景点云
            - pc_colors: 点云颜色
            - obj_mask: 目标物体掩码
        :param visualize: 是否进行可视化
        :param num_grasps: 生成的抓取数量
        :return: (相机坐标系下的最佳抓取位姿, 可视化信息)
        """
        # 提取输入数据
        pc = data['pc']
        pc_colors = data['pc_colors']
        obj_mask = data['obj_mask']

        # 设置相机参数（焦距和主点）
        camera_params = (213.2162658691406, 284.3765096028646, 111.82499999999999, 111.76666666666667)
        K = np.array([
            [camera_params[0], 0, camera_params[2]],
            [0, camera_params[1], camera_params[3]],
            [0, 0, 1]
        ])

        # 创建场景点云对象
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
        scene_cloud.colors = o3d.utility.Vector3dVector((pc_colors / 255.0).astype(np.float32))

        # 处理点云数据并生成抓取
        end_points, cloud = self.process_data(pc, pc_colors)
        self.cloud = cloud
        gg = self.get_grasps(end_points)

        # 进行碰撞检测
        if self.cfg.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(scene_cloud.points))

        # 整理抓取位姿信息
        grasp_poses = {'score':[], 'grasp_camera':[], 'grasp_base': []}
        for grasp in gg:
            grasp_camera = get_grasps_camera(grasp)
            grasp_poses['score'].append(grasp.score)
            grasp_poses['grasp_camera'].append(grasp_camera)
            grasp_poses['grasp_base'].append(self.trans_camera2base @ grasp_camera)

        self.gg = gg
        
        # 可视化原始抓取结果
        if self.cfg.vis_grasp:
            self.vis_grasps(gg, scene_cloud)

        # 根据方向过滤抓取
        filter_mask, grasp_poses = self.filer_direction_grasps(grasp_poses)
        gg = gg[filter_mask]

        # 可视化过滤后的抓取结果
        if self.cfg.vis_grasp:
            self.vis_grasps(gg, scene_cloud)

        # 选择最佳抓取
        best_id = self.find_max_score(grasp_poses)

        # 可视化最佳抓取
        if self.cfg.vis_grasp:
            grippers = gg[best_id:best_id+1].to_open3d_geometry_list()
            o3d.visualization.draw_geometries([cloud, *grippers])
            gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            gripper_frame.transform(grasp_poses['grasp_camera'][best_id] @ self.trans_tool)

        # 准备返回结果
        grasp_pose_camera = get_grasps_camera(gg[best_id])
        grasp_vis = [gg[best_id].translation, gg[best_id].rotation_matrix, gg[best_id].width, gg[best_id].depth]
        grasp_vis = np.array(grasp_vis, dtype=object)
        
        return grasp_pose_camera, grasp_vis

    def generate_grasps_more(self, data, visualize=False, num_grasps=None):
        """
        生成更多抓取位姿的增强版函数，专门用于处理分割后的目标物体点云
        :param data: 输入数据字典，包含：
            - pc: 完整场景的点云数据
            - pc_colors: 完整场景的点云颜色
            - object_pc: 目标物体的点云数据
            - object_pc_colors: 目标物体的点云颜色
        :param visualize: 是否进行可视化
        :param num_grasps: 生成的抓取数量
        :return: (相机坐标系下的最佳抓取位姿, 可视化信息)
        """
        # 提取输入数据
        pc = data['pc']                    # 完整场景点云
        pc_colors = data['pc_colors']      # 场景点云颜色
        obj_pc = data['object_pc']         # 目标物体点云
        obj_pc_colors = data['object_pc_colors']  # 目标物体点云颜色
        
        # 设置相机内参矩阵
        # fx, fy: 焦距, cx, cy: 主点坐标
        camera_params = (213.2162658691406, 284.3765096028646, 111.82499999999999, 111.76666666666667)
        K = np.array([
            [camera_params[0], 0, camera_params[2]],
            [0, camera_params[1], camera_params[3]],
            [0, 0, 1]
        ])

        # 创建完整场景的点云对象（用于碰撞检测）
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
        scene_cloud.colors = o3d.utility.Vector3dVector((pc_colors / 255.0).astype(np.float32))

        # 处理目标物体的点云数据
        end_points, cloud = self.process_data(obj_pc, obj_pc_colors)
        self.cloud = scene_cloud  # 保存场景点云用于后续处理
        
        # 使用神经网络生成抓取位姿
        gg = self.get_grasps(end_points)
        
        # 如果启用碰撞检测
        if self.cfg.collision_thresh > 0:
            # 使用完整场景点云进行碰撞检测，过滤掉可能碰撞的抓取
            gg = self.collision_detection(gg, np.array(scene_cloud.points))

        # 初始化抓取位姿字典，存储不同坐标系下的位姿信息
        grasp_poses = {'score':[], 'grasp_camera':[], 'grasp_base': []}
        
        # 遍历所有预测的抓取位姿
        for grasp in gg:
            # 获取相机坐标系下的抓取位姿
            grasp_camera = get_grasps_camera(grasp)
            
            # 保存抓取信息
            grasp_poses['score'].append(grasp.score)  # 抓取评分
            grasp_poses['grasp_camera'].append(grasp_camera)  # 相机坐标系下的位姿
            grasp_poses['grasp_base'].append(self.trans_camera2base @ grasp_camera)  # 机器人基座坐标系下的位姿
        
        # 可视化原始抓取结果
        if self.cfg.vis_grasp:
            self.vis_grasps(gg, scene_cloud)

        # 根据抓取方向进行过滤（确保抓取方向合理）
        filter_mask, grasp_poses = self.filer_direction_grasps(grasp_poses)
        gg = gg[filter_mask]  # 更新抓取组

        # 可视化过滤后的抓取结果
        if self.cfg.vis_grasp:
            self.vis_grasps(gg, scene_cloud)

        # 选择得分最高的抓取
        best_id = self.find_max_score(grasp_poses)

        # 可视化最佳抓取位姿
        if self.cfg.vis_grasp:
            # 显示抓取器模型
            grippers = gg[best_id:best_id+1].to_open3d_geometry_list()
            o3d.visualization.draw_geometries([cloud, *grippers])
            # 显示坐标系
            gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            gripper_frame.transform(grasp_poses['grasp_camera'][best_id] @ self.trans_tool)

        # 准备返回结果
        grasp_pose_camera = get_grasps_camera(gg[best_id])  # 最佳抓取位姿（相机坐标系）
        # 保存可视化所需信息：位置、旋转矩阵、抓取宽度和深度
        grasp_vis = [gg[best_id].translation, gg[best_id].rotation_matrix, gg[best_id].width, gg[best_id].depth]
        grasp_vis = np.array(grasp_vis, dtype=object)
        
        return grasp_pose_camera, grasp_vis

def wait_input():
    """
    等待用户输入控制命令
    :return: True表示继续，False表示退出
    """
    print("等待输入命令...")
    user_input = sys.stdin.read().strip()
    
    if user_input == "g":
        return True
    elif user_input == "q":
        return False

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LIDF Training')
    # 设置默认配置文件路径
    parser.add_argument('--default_cfg_path', 
                       default = '/home/ssr/transparent/src/experiments/implicit_depth/default_config.yaml', 
                       help='默认配置文件路径，包含基本参数设置')
    # 设置更新配置文件路径
    parser.add_argument("--cfg_path", 
                       type=str, 
                       default = '/home/ssr/transparent/src/experiments/implicit_depth/real_time.yaml', 
                       help="更新配置文件路径，用于覆盖默认配置")
    args = parser.parse_args()

    # 加载和设置配置参数
    if args.default_cfg_path is None:
        raise ValueError('未找到默认配置文件路径，必须指定一个默认配置文件')
    # 加载默认配置
    opt = Params(args.default_cfg_path)
    # 如果提供了更新配置，则更新参数
    if args.cfg_path is not None:
        opt.update(args.cfg_path)

    # 设置运行设备（GPU）
    device = torch.device('cuda:{}'.format(opt.gpu_id))

    # 初始化抓取网络模块
    GNM = GraspNetModule(opt, device)

    # 等待用户输入命令（'g'继续，'q'退出）
    flag = wait_input()

    if flag:  # 如果用户输入'g'，执行抓取检测
        # 根据配置选择不同的抓取生成模式
        if not opt.GRASPNET_BASELINE.generate_grasps_more:
            # 标准抓取生成模式
            # 创建数据字典，加载必要的点云数据
            data_dict_grasp = {}
            # 加载场景点云
            data_dict_grasp['pc'] = np.load(os.path.join("/home/ssr/transparent/logs/graspnet/", "pc.npy"))
            # 加载点云颜色
            data_dict_grasp['pc_colors'] = np.load(os.path.join("/home/ssr/transparent/logs/graspnet/", "pc_colors.npy"))
            # 加载物体掩码
            data_dict_grasp['obj_mask'] = np.load(os.path.join("/home/ssr/transparent/logs/graspnet/", "obj_mask.npy"))

            # 生成抓取位姿
            grasp_pose_camera, grasp_vis = GNM.generate_grasps(data_dict_grasp)

            # 保存结果
            # 保存相机坐标系下的抓取位姿
            np.save(os.path.join("/home/ssr/transparent/logs/graspnet", "grasp_pose_camera.npy"), grasp_pose_camera)
            # 保存可视化数据
            np.save(os.path.join("/home/ssr/transparent/logs/graspnet", "grasp_vis.npy"), grasp_vis)
        
        else:
            # 增强抓取生成模式（使用分割后的目标点云）
            data_dict_grasp = {}
            # 加载完整场景点云
            data_dict_grasp['pc'] = np.load(os.path.join("/home/ssr/transparent/logs/graspnet/", "pc.npy"))
            # 加载场景点云颜色
            data_dict_grasp['pc_colors'] = np.load(os.path.join("/home/ssr/transparent/logs/graspnet/", "pc_colors.npy"))
            # 加载目标物体点云
            data_dict_grasp['object_pc'] = np.load(os.path.join("/home/ssr/transparent/logs/graspnet/", "object_pc.npy"))
            # 加载目标物体点云颜色
            data_dict_grasp['object_pc_colors'] = np.load(os.path.join("/home/ssr/transparent/logs/graspnet/", "object_pc_colors.npy"))

            # 使用增强模式生成抓取位姿
            grasp_pose_camera, grasp_vis = GNM.generate_grasps_more(data_dict_grasp)

            # 保存结果
            # 保存相机坐标系下的抓取位姿
            np.save(os.path.join("/home/ssr/transparent/logs/graspnet", "grasp_pose_camera.npy"), grasp_pose_camera)
            # 保存可视化数据
            np.save(os.path.join("/home/ssr/transparent/logs/graspnet", "grasp_vis.npy"), grasp_vis)

    else:  # 如果用户输入'q'，退出程序
        print("关闭抓取检测模块")
