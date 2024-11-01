import socket
import struct
import threading
import numpy as np
import pyrealsense2 as rs
import redis
import time

# Initialize Redis connection
redis_host = "192.168.4.4"
redis_port = 6379
redis_password = ""  # If your Redis server has no password, keep it as an empty string.
r = redis.StrictRedis(
    host=redis_host, port=redis_port, password=redis_password, decode_responses=True
)

# Global variables
camera_pose = None
hand_data = []

def quaternion_to_euler(x, y, z, w):
    # Convert quaternion to Euler angles in radians
    pitch_y = -np.arcsin(2.0 * (x * z - w * y))
    roll_x = np.arctan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
    yaw_z = np.arctan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
    return roll_x, pitch_y, yaw_z

class CameraController:
    def __init__(self):
        self.pipe = rs.pipeline()
        self.configure_camera()

    def configure_camera(self):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.pose)
        self.pipe.start(cfg)
        self.serial_number = self.get_serial_number()

    def get_serial_number(self):
        profiles = self.pipe.get_active_profile()
        device = profiles.get_device()
        return device.get_info(rs.camera_info.serial_number)

    def get_camera_pose(self):
        global camera_pose
        while True:
            frames = self.pipe.wait_for_frames()
            pose_frame = frames.get_pose_frame()
            if pose_frame:
                pose_data = pose_frame.get_pose_data()
                euler_angles = quaternion_to_euler(pose_data.rotation.x, pose_data.rotation.y, pose_data.rotation.z, pose_data.rotation.w)
                camera_pose = np.array([
                    pose_data.translation.x, pose_data.translation.y, pose_data.translation.z,
                    euler_angles[0], euler_angles[1], euler_angles[2]
                ], dtype=np.float64)

                # Save camera pose to Redis based on serial number
                if self.serial_number == '133122110781':
                    r.set('right_camera_pose', camera_pose.tobytes())
                    print(camera_pose)
                if self.serial_number == '230222110115':
                    r.set('left_camera_pose', camera_pose.tobytes())
                    print(camera_pose)

                # time.sleep(0.5)
                print(f"Camera Pose ({self.serial_number}): {camera_pose}")
                # time.sleep(0.2)

def run_hand():
    global hand_data
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('127.0.0.1', 8082)
    udp_socket.bind(server_address)
    print("Listening for data on {} port {}...".format(*server_address))

    while True:
        data, address = udp_socket.recvfrom(1024)
        floats = struct.unpack('26f', data)
        hand_index = floats[0]
        thumb_angles = floats[2:5]
        index_angles = floats[7:10]
        middle_angles = floats[12:15]
        ring_angles = floats[17:20]
        pinky_angles = floats[22:25]

        hand_data = {
            'thumb': thumb_angles,
            'index': index_angles,
            'middle': middle_angles,
            'ring': ring_angles,
            'pinky': pinky_angles
        }

        # Save finger bend angles to Redis based on hand index
        finger_bend_angles = np.array([
            thumb_angles[0], thumb_angles[1], thumb_angles[2],
            index_angles[0], index_angles[1], index_angles[2],
            middle_angles[0], middle_angles[1], middle_angles[2],
            ring_angles[0], ring_angles[1], ring_angles[2],
            pinky_angles[0], pinky_angles[1], pinky_angles[2]
        ], dtype=np.float64)

        if hand_index == 0:
            r.set('left_bend_angles', finger_bend_angles.tobytes())
            # print(finger_bend_angles)
        elif hand_index == 1:
            r.set('right_bend_angles', finger_bend_angles.tobytes())

        # print(f"Hand Data: {hand_data}")
        # time.sleep(0.2)


def run():
    try:
        left_camera_controller = CameraController()
        right_camera_controller = CameraController()

        left_camera_thread = threading.Thread(target=left_camera_controller.get_camera_pose)
        right_camera_thread = threading.Thread(target=right_camera_controller.get_camera_pose)
        hand_thread = threading.Thread(target=run_hand)

        left_camera_thread.start()
        right_camera_thread.start()
        hand_thread.start()

        # left_camera_thread.join()
        right_camera_thread.join()
        hand_thread.join()
    except KeyboardInterrupt:
        print("程序被手动中止")
    finally:
        left_camera_controller.pipe.stop()
        right_camera_controller.pipe.stop()

if __name__ == '__main__':
    run()

