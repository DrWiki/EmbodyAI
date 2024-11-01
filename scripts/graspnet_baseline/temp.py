import numpy as np

# Define the camera intrinsics matrix
# Example values: focal lengths (fx, fy) and optical center (cx, cy)
K = np.array([
    [500, 0, 320],
    [0, 500, 240],
    [0, 0, 1]
])

# Define the 3D point in camera coordinates
point_3D = np.array([1, 2, 5])

# Convert the 3D point to homogeneous coordinates (add the fourth dimension with 1)
point_3D_h = np.append(point_3D, 1)  # [X, Y, Z, 1]

# Note: We need to convert point_3D_h to a 3D vector in homogeneous coordinates without the extra 1 for multiplication with K
point_3D_3D = np.array([point_3D[0], point_3D[1], point_3D[2]])  # Only X, Y, Z

# Project the 3D point onto the 2D image plane
point_2D_h = K @ np.append(point_3D_3D, 1)  # Matrix multiplication including the homogeneous coordinate

# Normalize to get pixel coordinates (u, v)
u = point_2D_h[0] / point_2D_h[2]
v = point_2D_h[1] / point_2D_h[2]

print(f"Pixel coordinates: u = {u}, v = {v}")
