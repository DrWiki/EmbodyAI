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

def backproject(depth_cv,
                intrinsic_matrix,
                return_finite_depth=True,
                return_selection=False):

    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    X = np.array(X).transpose()
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection

    return X

def get_grasps_camera(grasp):
    translation = grasp.translation
    rot = grasp.rotation_matrix

    grasp_trans = np.eye(4)
    grasp_trans[:3, :3] = rot
    grasp_trans[:3, -1] = translation

    return grasp_trans

class GraspNetModule():
    def __init__(self, cfg, device) -> None:
        self.cfg = cfg.GRASPNET_BASELINE
        self.device = device

        self.net = GraspNet(input_feature_dim=0, num_view=self.cfg.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        self.net.to(device)

        checkpoint = torch.load(self.cfg.checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        print("Load GraspNet successfully!")

        self.R_cam2base = np.load("/home/ssr/transparent/calibrate/R_cam2gripper.npy") # (3, 3)
        self.T_cam2base = np.load("/home/ssr/transparent/calibrate/t_cam2gripper.npy") # (3, 1)
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
        if mask is not None:
            depth[mask] = np.nan
        
        # generate whole obervation point cloud
        pc, selection = backproject(depth,
                                    K,
                                    return_finite_depth=True,
                                    return_selection=True)
        
        pc_colors = image.copy()
        pc_colors = np.reshape(pc_colors, [-1, 3])
        pc_colors = pc_colors[selection, :]
        
        return pc, pc_colors

    def process_data(self, pc, pc_colors):
        pc_colors = pc_colors / 255.0
        cloud_masked = pc
        color_masked = pc_colors

        # sample points
        if len(cloud_masked) >= self.cfg.num_point:
            idxs = np.random.choice(len(cloud_masked), self.cfg.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.cfg.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        cloud_sampled = cloud_sampled.to(self.device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud

    def get_grasps(self, end_points):
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfg.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfg.collision_thresh)
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
        grasp_points_pixel = self.grasp2pixel(grasp_poses, K)
        # print(grasp_poses.keys())

        grasp_poses_filtered = {'score':[], 'grasp_camera':[], 'grasp_base':[]}
        filter_mask = np.zeros((len(grasp_poses['score']),), dtype=bool)
        for i, grasp_point in enumerate(grasp_points_pixel):
            if obj_mask[int(grasp_point[1]), int(grasp_point[0])]:
                grasp_poses_filtered['score'].append(grasp_poses['score'][i])
                grasp_poses_filtered['grasp_camera'].append(grasp_poses['grasp_camera'][i])
                grasp_poses_filtered['grasp_base'].append(grasp_poses['grasp_base'][i])
                filter_mask[i] = True

        return filter_mask, grasp_poses_filtered

    def filer_direction_grasps(self, grasp_poses):
        filter_mask = np.zeros((len(grasp_poses['score']),), dtype=bool)
        y_min = self.cfg.r2h_thre_min
        z_max = self.cfg.verticle_thre_max
        
        grasp_poses_filtered = {'score':[], 'grasp_camera':[], 'grasp_base':[]}
        for i, gb in enumerate(grasp_poses['grasp_camera']):
            grasp_world = self.trans_base2world @ self.trans_camera2base @ gb
            unit_vector_world = np.dot(grasp_world[:3, :3], np.array([1, 0, 0]))

            if unit_vector_world[1] > y_min and unit_vector_world[2] < z_max:
                grasp_poses_filtered['score'].append(grasp_poses['score'][i])
                grasp_poses_filtered['grasp_camera'].append(grasp_poses['grasp_camera'][i])
                grasp_poses_filtered['grasp_base'].append(grasp_poses['grasp_base'][i])
                filter_mask[i] = True

                # print(f"vector: {unit_vector_world}")
                # grippers = self.gg[i:i+1].to_open3d_geometry_list()
                # world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                # camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                # gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                # # world_frame.transform(self.trans_world2base)
                # world_frame.transform(self.trans_base2camera)
                # # gripper_frame.transform(self.trans_tool)
                # gripper_frame.transform(grasp_poses['grasp_camera'][i] @ self.trans_tool)
                
                # o3d.visualization.draw_geometries([self.cloud, *grippers, gripper_frame, world_frame, camera_frame])


        return filter_mask, grasp_poses_filtered

    def find_max_score(self, grasp_poses):
        # grasps_rule_score = np.array([abs(grasp_poses['grasp_base'][i][2, 0]) for i in range(len(grasp_poses['grasp_base']))])
        final_score = grasp_poses['score']
        max_id = np.argmax(final_score)

        return int(max_id)

    def vis_grasps(self, gg, cloud, num_grasp_vis=None):
        if num_grasp_vis is not None:
            gg = gg[:num_grasp_vis]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

            

    def generate_grasps(self, data, visualize=False, num_grasps=None):
        pc = data['pc']
        pc_colors = data['pc_colors']
        obj_mask = data['obj_mask']
        camera_params = (213.2162658691406, 284.3765096028646, 111.82499999999999, 111.76666666666667)
        K = np.array([
            [camera_params[0], 0, camera_params[2]],
            [0, camera_params[1], camera_params[3]],
            [0, 0, 1]
        ])

        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
        scene_cloud.colors = o3d.utility.Vector3dVector((pc_colors / 255.0).astype(np.float32))

        end_points, cloud = self.process_data(pc, pc_colors)
        self.cloud = cloud
        gg = self.get_grasps(end_points)
        if self.cfg.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(scene_cloud.points))

        grasp_poses = {'score':[], 'grasp_camera':[], 'grasp_base': []}
        for grasp in gg:
            grasp_camera = get_grasps_camera(grasp)

            grasp_poses['score'].append(grasp.score)
            grasp_poses['grasp_camera'].append(grasp_camera)
            grasp_poses['grasp_base'].append(self.trans_camera2base @ grasp_camera)

        # filter_mask, grasp_poses = self.filter_obj_grasps(grasp_poses, obj_mask, K)
        # gg = gg[filter_mask]
        self.gg = gg
        
        if self.cfg.vis_grasp:
            self.vis_grasps(gg, scene_cloud)

        filter_mask, grasp_poses = self.filer_direction_grasps(grasp_poses)
        gg = gg[filter_mask]

        if self.cfg.vis_grasp:
            self.vis_grasps(gg, scene_cloud)

        best_id = self.find_max_score(grasp_poses)

        if self.cfg.vis_grasp:
            grippers = gg[best_id:best_id+1].to_open3d_geometry_list()
            o3d.visualization.draw_geometries([cloud, *grippers])
            gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            gripper_frame.transform(grasp_poses['grasp_camera'][best_id] @ self.trans_tool)

        grasp_pose_camera = get_grasps_camera(gg[best_id])
        grasp_vis = [gg[best_id].translation, gg[best_id].rotation_matrix, gg[best_id].width, gg[best_id].depth]
        grasp_vis= np.array(grasp_vis, dtype=object)
        
        return grasp_pose_camera, grasp_vis

    def generate_grasps_more(self, data, visualize=False, num_grasps=None):
        pc = data['pc']
        pc_colors = data['pc_colors']
        obj_pc = data['object_pc']
        obj_pc_colors = data['object_pc_colors']
        
        camera_params = (213.2162658691406, 284.3765096028646, 111.82499999999999, 111.76666666666667)
        K = np.array([
            [camera_params[0], 0, camera_params[2]],
            [0, camera_params[1], camera_params[3]],
            [0, 0, 1]
        ])

        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
        scene_cloud.colors = o3d.utility.Vector3dVector((pc_colors / 255.0).astype(np.float32))

        end_points, cloud = self.process_data(obj_pc, obj_pc_colors)
        self.cloud = scene_cloud
        gg = self.get_grasps(end_points)
        if self.cfg.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(scene_cloud.points))

        grasp_poses = {'score':[], 'grasp_camera':[], 'grasp_base': []}
        for grasp in gg:
            grasp_camera = get_grasps_camera(grasp)

            grasp_poses['score'].append(grasp.score)
            grasp_poses['grasp_camera'].append(grasp_camera)
            grasp_poses['grasp_base'].append(self.trans_camera2base @ grasp_camera)

        # filter_mask, grasp_poses = self.filter_obj_grasps(grasp_poses, obj_mask, K)
        # gg = gg[filter_mask]
        # self.gg = gg
        
        if self.cfg.vis_grasp:
            self.vis_grasps(gg, scene_cloud)

        filter_mask, grasp_poses = self.filer_direction_grasps(grasp_poses)
        gg = gg[filter_mask]

        if self.cfg.vis_grasp:
            self.vis_grasps(gg, scene_cloud)

        best_id = self.find_max_score(grasp_poses)

        if self.cfg.vis_grasp:
            grippers = gg[best_id:best_id+1].to_open3d_geometry_list()
            o3d.visualization.draw_geometries([cloud, *grippers])
            gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            gripper_frame.transform(grasp_poses['grasp_camera'][best_id] @ self.trans_tool)

        grasp_pose_camera = get_grasps_camera(gg[best_id])
        grasp_vis = [gg[best_id].translation, gg[best_id].rotation_matrix, gg[best_id].width, gg[best_id].depth]
        grasp_vis= np.array(grasp_vis, dtype=object)
        
        return grasp_pose_camera, grasp_vis

def wait_input():
    print("Waiting for input...")
    # Read input from standard input
    user_input = sys.stdin.read().strip()
    
    if user_input == "g":
        return True
    elif user_input == "q":
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LIDF Training')
    parser.add_argument('--default_cfg_path', default = '/home/ssr/transparent/src/experiments/implicit_depth/default_config.yaml', help='default config file')
    parser.add_argument("--cfg_path", type=str, default = '/home/ssr/transparent/src/experiments/implicit_depth/real_time.yaml', help="List of updated config file")
    args = parser.parse_args()

    # setup opt
    if args.default_cfg_path is None:
        raise ValueError('default config path not found, should define one')
    opt = Params(args.default_cfg_path)
    if args.cfg_path is not None:
        opt.update(args.cfg_path)

    device = torch.device('cuda:{}'.format(opt.gpu_id))

    GNM = GraspNetModule(opt, device)

    flag = wait_input()

    if flag:
        if not opt.GRASPNET_BASELINE.generate_grasps_more:
            data_dict_grasp = {}
            data_dict_grasp['pc'] = np.load(os.path.join("/home/ssr/transparent/logs/graspnet/", "pc.npy"))
            data_dict_grasp['pc_colors'] = np.load(os.path.join("/home/ssr/transparent/logs/graspnet/", "pc_colors.npy"))
            data_dict_grasp['obj_mask'] = np.load(os.path.join("/home/ssr/transparent/logs/graspnet/", "obj_mask.npy"))

            grasp_pose_camera, grasp_vis = GNM.generate_grasps(data_dict_grasp)

            np.save(os.path.join("/home/ssr/transparent/logs/graspnet", "grasp_pose_camera.npy"), grasp_pose_camera)
            np.save(os.path.join("/home/ssr/transparent/logs/graspnet", "grasp_vis.npy"), grasp_vis)
        else:
            data_dict_grasp = {}
            data_dict_grasp['pc'] = np.load(os.path.join("/home/ssr/transparent/logs/graspnet/", "pc.npy"))
            data_dict_grasp['pc_colors'] = np.load(os.path.join("/home/ssr/transparent/logs/graspnet/", "pc_colors.npy"))
            data_dict_grasp['object_pc'] = np.load(os.path.join("/home/ssr/transparent/logs/graspnet/", "object_pc.npy"))
            data_dict_grasp['object_pc_colors'] = np.load(os.path.join("/home/ssr/transparent/logs/graspnet/", "object_pc_colors.npy"))

            grasp_pose_camera, grasp_vis = GNM.generate_grasps_more(data_dict_grasp)

            np.save(os.path.join("/home/ssr/transparent/logs/graspnet", "grasp_pose_camera.npy"), grasp_pose_camera)
            np.save(os.path.join("/home/ssr/transparent/logs/graspnet", "grasp_vis.npy"), grasp_vis)
            

    else:
        print("close graspnet module")