from ikpy.chain import Chain
import numpy as np
from .IK_solver import solver


class RetargetClass():
    def __init__(self, arm_type):
        if arm_type == 'left':
            urdf_path = "robot/left_arm_7/left_arm_7.urdf"
            init_angle = np.zeros([8])
            init_angle[4] = -1.57
            base_world = np.array([[-1.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0]])

        elif arm_type == 'right':
            urdf_path = "D:/RemoteControl/robot/right_arm_2/right_arm_7.urdf"
            init_angle = np.zeros([8])
            init_angle[4] = 1.57  # 右臂可能需要不同的初始角度
            base_world = np.array([[1.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, -1.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0]])
        else:
            raise ValueError("Invalid arm type. Please choose 'left' or 'right'.")
        self.chain = Chain.from_urdf_file(urdf_path,
                                          active_links_mask=[False, True, True, True, True, True, True, True],base_elements=["base_link"])
        self.init_pose_base = self.chain.forward_kinematics(init_angle)
        self.base_world = base_world
        self.pose_init_world = self.base_world @ self.init_pose_base
        self.world_base = np.linalg.inv(self.base_world)
        self.current_pose_world = self.pose_init_world  # 现在的位姿在世界坐标系下的表示


    def get_end_pose_world(self, init_pos, delta_pose):
        std0_world = np.eye(4)
        std0_world[:3, -1] = init_pos[:3, -1]
        e0_std0 = np.linalg.inv(std0_world) @ init_pos
        return std0_world @ delta_pose @ e0_std0

    def get_retarget_joint_angle(self, delta_pose, init_angles=np.zeros([8]), accumulate_mode=False):
        """
        @param accumulate_mode False代表给的glove delta是单步的误差
        """
        if accumulate_mode:
            self.current_pose_world = self.get_end_pose_world(self.pose_init_world, delta_pose)
            print("current_pose_world:", self.current_pose_world)
        else:
            self.current_pose_world = self.get_end_pose_world(self.current_pose_world, delta_pose)
        new_pose_base = self.world_base @ self.current_pose_world
        return solver(self.chain, new_pose_base, init_angles)  #

    def reset(self):
        self.current_pose_world = self.pose_init_world  # 现在的位姿在世界坐标系下的表示
