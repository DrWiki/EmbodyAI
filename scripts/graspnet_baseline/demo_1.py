""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import matplotlib.pyplot as plt

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import GraspNetDataset
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    print(cfgs.checkpoint_path)
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(result_dir):
    # load data

    obj_mask = np.load(os.path.join(result_dir, "segmap.npy"))
    obj_mask = np.squeeze(obj_mask).astype(bool)
    obj_mask = obj_mask.reshape((-1, 1))
    color = np.load(os.path.join(result_dir, "rgb_img.npy"))
    color = color / 255.0

    depth = None
    # cloud = np.load(os.path.join(result_dir, "pc_full.npy"))
    # pc_colors = np.load(os.path.join(result_dir, "pc_colors.npy"))
    cloud = np.load(os.path.join(result_dir, "pc_wo_hand.npy"))
    pc_colors = np.load(os.path.join(result_dir, "pc_wo_hand_colors.npy"))
    pc_colors = pc_colors / 255.0

    # generate cloud
    # camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    # cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    # mask = (workspace_mask & (depth > 0))
    # mask = workspace_mask
    # print(cloud.shape, obj_mask.shape)
    cloud_masked = cloud
    color_masked = pc_colors

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def get_grasps_camera(grasp):
    translation = grasp.translation
    rot = grasp.rotation_matrix

    grasp_trans = np.eye(4)
    grasp_trans[:3, :3] = rot
    grasp_trans[:3, -1] = translation

    return grasp_trans

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def grasp2pixel(grasp_poses):
    # Camera parameters
    camera_params = (213.2162658691406, 284.3765096028646, 111.82499999999999, 111.76666666666667)
    width, height = 224, 224
    K = np.array([
        [camera_params[0], 0, camera_params[2]],
        [0, camera_params[1], camera_params[3]],
        [0, 0, 1]
    ])
    
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

    # Visualization
    color = np.load(os.path.join("/home/ssr/transparent/logs/real_time", "rgb_img.npy"))
    plt.figure(figsize=(8, 8))
    plt.imshow(color)
    plt.scatter(pixel_coordinate[:, 0], pixel_coordinate[:, 1], c='r', marker='o')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.title('Grasp Points in Image Coordinates')
    plt.show()

    return pixel_coordinate

def filter_obj_grasps(grasp_poses):
    grasp_points_pixel = grasp2pixel(grasp_poses)
    obj_mask = np.load(os.path.join("/home/ssr/transparent/logs/real_time", "segmap.npy"))
    obj_mask = np.squeeze(obj_mask).astype(bool)
    
    # plt.imshow(obj_mask)
    # plt.scatter(grasp_points_pixel[:, 0], grasp_points_pixel[:, 1], c='r', marker='o')
    # plt.show()

    grasp_candidate = []
    for i, grasp_point in enumerate(grasp_points_pixel):
        if obj_mask[int(grasp_point[1]), int(grasp_point[0])]:
            grasp_candidate.append(i)

    if len(grasp_candidate):
        scores = [grasp_poses['score'][i] for i in grasp_candidate]
        grasp_points_pixel = [grasp_points_pixel[i] for i in grasp_candidate]
        grasps = [grasp_poses['grasp_camera'][i] for i in grasp_candidate]
    print(len(grasp_points_pixel))

    # Visualization
    color = np.load(os.path.join("/home/ssr/transparent/logs/real_time", "rgb_img.npy"))
    plt.figure(figsize=(8, 8))
    plt.imshow(color)
    for grasp_point_pixel in grasp_points_pixel:
        plt.scatter(grasp_point_pixel[0], grasp_point_pixel[1], c='r', marker='o')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.title('Grasp Points in Image Coordinates')
    plt.show()


def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:10]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)

    grasp_poses = {'score':[], 'grasp_camera':[]}
    for grasp in gg:
        grasp_camera = get_grasps_camera(grasp)

        grasp_poses['score'].append(grasp.score)
        grasp_poses['grasp_camera'].append(grasp_camera)
    filter_obj_grasps(grasp_poses)

if __name__=='__main__':
    data_dir = "/home/ssr/transparent/logs/real_time"
    demo(data_dir)
