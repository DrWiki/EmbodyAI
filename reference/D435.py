import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import threading
import time

class D435Camera:
    """
    Intel RealSense D435相机驱动类
    用于获取RGB图像、深度图和点云数据，并提供可视化功能
    """
    def __init__(self):
        """
        初始化D435相机
        """
        # 初始化相机参数
        self.width = 640
        self.height = 480
        self.fps = 30
        
        # 初始化RealSense流水线
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 配置数据流
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        # 初始化对齐对象
        self.align = None
        
        # 点云可视化器
        self.vis = o3d.visualization.Visualizer()
        self.is_running = False
        self.point_cloud = o3d.geometry.PointCloud()

    def start(self):
        """
        启动相机
        """
        try:
            # 启动流水线
            profile = self.pipeline.start(self.config)
            
            # 创建对齐对象（将深度图对齐到彩色图）
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            # 获取深度传感器的深度标尺
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            
            print(f"相机启动成功! 深度标尺: {depth_scale}")
            return True
            
        except Exception as e:
            print(f"相机启动失败: {e}")
            return False

    def stop(self):
        """
        停止相机
        """
        self.pipeline.stop()
        self.is_running = False
        self.vis.destroy_window()
        cv2.destroyAllWindows()
        print("相机已停止!")

    def get_frames(self):
        """
        获取对齐后的RGB图像和深度图
        :return: (RGB图像, 深度图, 相机内参)
        """
        try:
            # 等待新的帧
            frames = self.pipeline.wait_for_frames()
            
            # 对齐深度帧和彩色帧
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None, None

            # 获取相机内参
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            camera_matrix = np.array([
                [color_intrin.fx, 0, color_intrin.ppx],
                [0, color_intrin.fy, color_intrin.ppy],
                [0, 0, 1]
            ])

            # 转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return color_image, depth_image, camera_matrix
            
        except Exception as e:
            print(f"获取帧失败: {e}")
            return None, None, None

    def create_point_cloud(self, depth_image, color_image, camera_matrix):
        """
        从深度图和彩色图创建点云
        """
        # 创建点云
        depth = o3d.geometry.Image(depth_image)
        color = o3d.geometry.Image(color_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=1000.0,  # D435深度单位是毫米
            depth_trunc=5.0,     # 最大深度5米
            convert_rgb_to_intensity=False)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height,
            camera_matrix[0,0], camera_matrix[1,1],  # fx, fy
            camera_matrix[0,2], camera_matrix[1,2])  # cx, cy

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsic)

        # 将点云上下翻转以匹配真实世界的坐标系统
        pcd.transform([[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])
        
        return pcd

    def show_images(self, color_image, depth_image):
        """
        在同一窗口中显示RGB图像和深度图
        """
        # 将深度图转换为伪彩色图像
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET)
        
        # 水平拼接两张图像
        images = np.hstack((color_image, depth_colormap))
        
        # 显示图像
        cv2.imshow('RGB and Depth Images', images)

    def update_point_cloud_visualization(self, pcd):
        """
        更新点云可视化
        """
        if not self.is_running:
            # 首次运行时初始化可视化器
            self.vis.create_window("Point Cloud", width=1024, height=768)
            self.vis.add_geometry(pcd)
            self.is_running = True
        else:
            # 更新点云数据
            self.vis.update_geometry(pcd)
        
        # 更新可视化
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):
        """
        运行相机并显示实时数据
        """
        try:
            while True:
                # 获取图像帧
                color_image, depth_image, camera_matrix = self.get_frames()
                if color_image is None:
                    continue

                # 显示RGB图像和深度图
                self.show_images(color_image, depth_image)

                # 创建和更新点云
                pcd = self.create_point_cloud(depth_image, color_image, camera_matrix)
                self.update_point_cloud_visualization(pcd)

                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.stop()

def main():
    """
    主函数
    """
    # 创建相机对象
    camera = D435Camera()
    
    # 启动相机
    if not camera.start():
        return

    # 运行相机
    camera.run()

if __name__ == "__main__":
    main()