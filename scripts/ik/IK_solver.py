import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.optimize

def matrix_inverse(matrix:np.array)-> np.array:
    new_matrix = np.copy(matrix)
    new_matrix[:3,:3] = np.transpose(matrix[:3,:3])
    new_matrix[:3,-1] = - new_matrix[:3,:3]@matrix[:3,-1]
    return new_matrix

def quat2Rot(quaternion):
    r = R.from_quat(quaternion)
    rot = r.as_matrix()
    return rot

def compound_Rot_p_2_T(Rot,pos):
    T = np.eye(4)
    T[:3,:3] = Rot
    T[:3,-1] = np.transpose(pos)
    return T

def pose_to_homogeneous_matrix(pose):
    # 解构输入的位姿
    x, y, z, roll, pitch, yaw = pose

    # 将欧拉角转换为旋转矩阵
    rotation_matrix = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    # 创建齐次变换矩阵
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = [x, y, z]

    return homogeneous_matrix

def pose_to_homogeneous_matrix(pose):
    # 解构输入的位姿
    x, y, z, roll, pitch, yaw = pose

    # 将欧拉角转换为旋转矩阵
    rotation_matrix = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    # 创建齐次变换矩阵
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = [x, y, z]

    return homogeneous_matrix


def solver(chain,target,starting_nodes_angles):
    def optimize_basis(x):
        # y = np.append(starting_nodes_angles[:chain.first_active_joint], x)
        y = chain.active_to_full(x, starting_nodes_angles)
        fk = chain.forward_kinematics(y)
        return fk

    def optimize_target_function(x):
        fk = optimize_basis(x)
        target_error = (fk[:3, -1] - target[:3,-1])
        orientation_error = (fk[:3,:3] - target[:3, :3]).ravel()
        # We need to return the fk, it will be used in a later function
        # This way, we don't have to recompute it
        return np.concatenate([target_error, 0.01 * orientation_error])

    f = optimize_target_function(starting_nodes_angles)

    real_bounds = [link.bounds for link in chain.links]
    real_bounds = chain.active_from_full(real_bounds)
    real_bounds = np.moveaxis(real_bounds, -1, 0)
    res = scipy.optimize.least_squares(optimize_target_function, chain.active_from_full(starting_nodes_angles),bounds=real_bounds,ftol=1e-15,verbose=0)

    if res.status == -1:
        raise ValueError("improper input parameters status returned from MINPACK")
    if res.cost > 1:
        print("unable to solve")
        return np.zeros(8)
    return chain.active_to_full(res.x, starting_nodes_angles)
