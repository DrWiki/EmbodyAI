import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import threading
import time
from datetime import datetime
import os

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
        self.width = 1920
        self.height = 1080
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
        
        # 创建保存文件夹
        self.save_dir = "camera_data"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 打印可用分辨率
        self.print_available_resolutions()

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

    def save_data(self, color_image, depth_image, pcd):
        """
        保存RGB图像、深度图和点云数据
        :param color_image: RGB图像
        :param depth_image: 深度图
        :param pcd: 点云数据
        """
        try:
            # 生成时间戳作为文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 为当前时间创建子文件夹
            save_path = os.path.join(self.save_dir, timestamp)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # 保存RGB图像
            color_file = os.path.join(save_path, "color.png")
            cv2.imwrite(color_file, color_image)
            
            # 保存深度图（以16位PNG格式保存以保持精度）
            depth_file = os.path.join(save_path, "depth.png")
            cv2.imwrite(depth_file, depth_image.astype(np.uint16))
            
            # 保存点云（以PCD格式保存）
            pcd_file = os.path.join(save_path, "pointcloud.pcd")
            o3d.io.write_point_cloud(pcd_file, pcd)
            
            print(f"数据已保存到文件夹: {save_path}")
            return True
            
        except Exception as e:
            print(f"保存数据失败: {e}")
            return False

    def run(self):
        """
        运行相机并显示实时数据
        按'q'退出，按's'保存当前帧数据
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

                # 检测键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # 按'q'退出
                    break
                elif key == ord('s'):  # 按's'保存数据
                    self.save_data(color_image, depth_image, pcd)

        finally:
            self.stop()

    def print_available_resolutions(self):
        """
        打印D435相机支持的所有分辨率
        """
        try:
            # 获取相机设备
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                print("未找到RealSense设备!")
                return

            device = devices[0]
            
            # 获取深度和彩色传感器
            depth_sensor = device.first_depth_sensor()
            color_sensor = device.first_color_sensor()

            print("\n=== D435相机支持的分辨率 ===")
            
            # 打印深度传感器支持的分辨率
            print("\n深度传感器支持的分辨率:")
            for depth_stream in depth_sensor.get_stream_profiles():
                if depth_stream.stream_type() == rs.stream.depth:
                    video_stream = depth_stream.as_video_stream_profile()
                    fps = video_stream.fps()
                    width = video_stream.width()
                    height = video_stream.height()
                    format = video_stream.format()
                    print(f"- {width}x{height} @ {fps}fps ({format})")

            # 打印彩色传感器支持的分辨率
            print("\n彩色传感器支持的分辨率:")
            for color_stream in color_sensor.get_stream_profiles():
                if color_stream.stream_type() == rs.stream.color:
                    video_stream = color_stream.as_video_stream_profile()
                    fps = video_stream.fps()
                    width = video_stream.width()
                    height = video_stream.height()
                    format = video_stream.format()
                    print(f"- {width}x{height} @ {fps}fps ({format})")

        except Exception as e:
            print(f"获取分辨率信息失败: {e}")

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
