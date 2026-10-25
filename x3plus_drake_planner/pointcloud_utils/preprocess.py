"""Load, downsample, and fuse RGB-D point-cloud captures."""

import cv2
import numpy as np
import open3d as o3d
from pathlib import Path


# ---------------------------- LOAD NPZ AND CREATE WORLD-FRAME POINTCLOUD -----------------------------

def load_point_cloud(npz_path, max_depth=0.6, voxel_size=0.005):
    """
    Load NPZ file from rgbd_pose_dataset and generate pointcloud in world frame.
    
    Returns:
        pcd: downsampled pointcloud.
    """

    data = np.load(npz_path)

    depth = data["depth"].astype(np.float32) / 1000.0 # mm to m
    rgb = data["rgb"]
    pose = data["pose"]

    fx, fy, cx, cy, _, _ = data["depth_intrinsics"]

    H, W = depth.shape # depth cam resolution

    rgb_resized = cv2.resize(rgb, (W, H)) # resize rgb cam to depth cam resolution 
    rgb_flat = rgb_resized.reshape(-1, 3)

    # Generate pointcloud in depth cam frame
    u, v = np.meshgrid(np.arange(W), np.arange(H)) 
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    valid = Z.reshape(-1) > 0
    close = Z.reshape(-1) < max_depth # keep points within 0.6 m
    valid = valid & close

    points_cam = points_cam[valid]
    colors = rgb_flat[valid] / 255.0

    # Transform to world frame
    ones = np.ones((points_cam.shape[0], 1)) 
    points_h = np.hstack((points_cam, ones))
    points_world = (pose @ points_h.T).T[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    print(f"\nLoaded downsampled pointcloud with {len(pcd.points)} points")

    return pcd


def load_fused_point_cloud(captures_dir, start=7, end=7, max_depth=0.6, voxel_size=0.005):
    fused = o3d.geometry.PointCloud()
    
    # Determine step direction (1 if ascending, -1 if descending)
    step = 1 if start <= end else -1
    
    # Loop over the exact range (including the 'end' index)
    for i in range(start, end + step, step):
        # Format index to a 5-digit zero-padded string (e.g., 00019.npz)
        filename = f"{i:05d}.npz"
        npz_path = Path(captures_dir) / filename
        
        # Check if the file exists before attempting to load it
        if npz_path.exists():
            fused += load_point_cloud(npz_path, max_depth=max_depth, voxel_size=voxel_size)
        else:
            print(f"Warning: File {filename} not found in {captures_dir}")
            
    # Process the fused point cloud as before
    #fused = fused.voxel_down_sample(voxel_size)
    fused, _ = fused.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return fused