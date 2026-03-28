# adapted from https://github.com/rmurai0610/MASt3R-SLAM/tree/main
import sys
import os

import argparse
import pathlib
import copy
import cv2
from termcolor import colored

import numpy as np
import open3d as o3d
from natsort import natsorted
from scipy.spatial.transform import Rotation

import evo
from evo.core import sync
import evo.core.metrics as metrics
from evo.tools import file_interface
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（上级目录）
project_root = os.path.dirname(current_dir)

# 将项目根目录加入 Python 模块搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from evals import geometry_eval_utils as geom_utils


def vggt_resize(img, depth, new_size = (288, 512)):
    resized_img = np.array(img)
    resized_depth = np.array(depth)
    H, W = img.shape[:2]

    new_H, new_W = new_size
    resized_img = cv2.resize(
        resized_img, (new_W, new_H), interpolation=cv2.INTER_LANCZOS4
    )
    resized_depth = cv2.resize(
        resized_depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST
    )

    H1, W1 = resized_img.shape[:2]

    H2, W2 = resized_img.shape[:2]
    scale_w = W / W1
    scale_h = H / H1
    half_crop_w = (W1 - W2) / 2
    half_crop_h = (H1 - H2) / 2

    return np.ascontiguousarray(resized_img), np.ascontiguousarray(resized_depth), (scale_w, scale_h, half_crop_w, half_crop_h)

def load_7scenes(dataset, W, H, calib):
    """
    Returns the ground truth trajectory and point cloud in the world coordinate
    """
    subsample = 1  # TODO REMOVE THIS!
    rgb_files = natsorted(list((dataset / "seq-01").glob("*.color.png")))[::subsample]
    depth_files = natsorted(list((dataset / "seq-01").glob("*.depth.png")))[::subsample]
    pose_files = natsorted(list((dataset / "seq-01").glob("*.pose.txt")))[::subsample]
    fx, fy, cx, cy = calib  # kinect intrinsics
    # create rgbdimages
    rgbd_images = []
    gt_poses = []
    pcds = []
    gt_tum_traj_WC = []
    valid_masks = []

    for i, (rgb_file, depth_file, pose_file) in enumerate(zip(rgb_files, depth_files, pose_files)):
        color = cv2.imread(rgb_file.as_posix())
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_file.as_posix(), cv2.IMREAD_UNCHANGED)
        pose_WC = np.loadtxt(pose_file)
        color, depth, resize_params = vggt_resize(color, depth)
        H1, W1 = color.shape[:2]
        fx1 = fx / resize_params[0]
        fy1 = fy / resize_params[1]
        cx1 = cx / resize_params[0] - resize_params[2]
        cy1 = cy / resize_params[1] - resize_params[3]

        # depth range of kinect is 0.5 - 4.5m
        depth[depth == 65535] = 0
        depth[depth > 4.5 * 1000] = 0
        depth = np.nan_to_num(depth, nan=0)
        valid_mask = depth > 0

        color = o3d.geometry.Image(color)
        depth = o3d.geometry.Image(depth)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1000,  # mm -> m
            depth_trunc=4.5,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                width=W1, height=H1, fx=fx1, fy=fy1, cx=cx1, cy=cy1
            ),
            project_valid_depth_only=True,
        )
        pcd.transform(pose_WC)

        rgbd_images.append(rgbd_image)
        gt_poses.append(pose_WC)
        pcds.append(pcd)

        valid_masks.append(valid_mask)

        t_WC = pose_WC[:3, 3]
        R_WC = pose_WC[:3, :3]
        q_WC = Rotation.from_matrix(R_WC).as_quat()
        gt_tum_traj_WC.append(np.array([i * subsample, *t_WC, *q_WC]))

    return gt_tum_traj_WC, pcds, valid_masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/home/<user>/Documents/MASt3R-SLAM/datasets/7-scenes/office")
    parser.add_argument("--gt", default="/home/<user>/Documents/MASt3R-SLAM/groundtruths/7-scenes/office.txt")
    parser.add_argument("--est", default="/home/<user>/Documents/vggt/office.txt")
    parser.add_argument("--pcdest", default="/home/<user>/Documents/vggt/office.txt")

    parser.add_argument("--no-viz", action="store_true")

    args = parser.parse_args()

    dataset = pathlib.Path(args.dataset)
    calib = 585.0, 585.0, 320.0, 240.0  # kinect intrinsics
    W, H = 640, 480
    K = np.array(
        [[calib[0], 0.0, calib[2]], [0.0, calib[1], calib[3]], [0.0, 0.0, 1.0]]
    )
    gt_traj, gt_pcds, valid_masks = load_7scenes(dataset, W, H, calib)

    # intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, K)
    if not args.no_viz:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    traj_ref = file_interface.read_tum_trajectory_file(args.gt)
    traj_est = file_interface.read_tum_trajectory_file(args.est)
    matches = sync.matching_time_indices(traj_ref.timestamps, traj_est.timestamps)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    traj_est_aligned = copy.deepcopy(traj_est)
    r_a, t_a, s = traj_est_aligned.align(traj_ref, correct_scale=True, correct_only_scale=False)
    t_a = t_a.reshape(3, 1)

    # traj_est_aligned_poses = traj_est_aligned.poses_se3
    traj_est_poses = traj_est.poses_se3

    temp_path = args.est.replace(".txt", "_logs/")

    pcd_gt = []
    pcd_est_list = []
    for a, b in zip(matches[0], matches[1]):
        valid_mask = valid_masks[int(a)]
        gt_p = np.asarray(gt_pcds[int(a)].points)
        pcd_gt.append(gt_p)
        # est_points = np.load(temp_path+str(a)+".0.npz")['pointcloud']
        # est_mask = np.load(temp_path+str(a)+".0.npz")['mask']
        # pcd_est_list.append(est_points[valid_mask & est_mask])

    pcd_est = o3d.io.read_point_cloud(args.pcdest)
    # pcd_est = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_est))
    pcd_est.paint_uniform_color([0.0, 0.0, 1.0])

    center = np.asarray(pcd_est.get_center(), dtype=np.float64)
    # scale point cloud and given initial estimate of transformation.
    points = np.asarray(pcd_est.points)
    scaled_points = ((s*r_a) @ points.T + t_a).T
    pcd_est.points = o3d.utility.Vector3dVector(scaled_points)

    gt_pcd = np.concatenate(pcd_gt, axis=0)

    gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pcd))
    gt_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    # print(f"Number of points in estimated point cloud: {len(pcd_est.points)}")
    # print(f"Number of points in ground truth point cloud: {len(gt_pcd.points)}")
    
    # Run ICP alignment.
    # Downsample both point clouds for ICP only.
    voxel_size = 0.05  # Adjust this depending on your scale and desired speed/accuracy tradeoff
    pcd_est_down = pcd_est.voxel_down_sample(voxel_size)
    gt_pcd_down = gt_pcd.voxel_down_sample(voxel_size)

    # Run ICP on downsampled point clouds
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_est_down,
        gt_pcd_down,
        max_correspondence_distance=0.1,  # Adjust to match voxel size
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Apply the transformation to the full-resolution source point cloud
    transformation = reg_p2p.transformation
    pcd_est.transform(transformation)

    gt_mean = np.mean(np.asarray(gt_pcd.points), axis=0)
    est_mean = np.mean(np.asarray(pcd_est.points), axis=0)
    # print(f"Mean of the ground truth point cloud: {gt_mean}")
    # print(f"Mean of the estimated point cloud: {est_mean}")

    chamfer_dist, rmse_acc, rmse_comp, dists1, dists2 = (
        geom_utils.chamfer_distance_RMSE(gt_pcd, pcd_est, max_error=0.5)
    )

    # thresholds = np.arange(0, 1.0, 0.1)
    # valid1 = dists1.reshape(-1, 1) < thresholds.reshape(1, -1)
    # valid2 = dists2.reshape(-1, 1) < thresholds.reshape(1, -1)
    # print(valid1.sum(axis=0) / valid1.shape[0])
    # print(valid2.sum(axis=0) / valid2.shape[0])

    # print using colored
    print(colored("Dense eval results on", "green"), args.est)
    print(colored("RMSE acc:", "green"), rmse_acc)
    print(colored("RMSE comp:", "green"), rmse_comp)
    print(colored("Chamfer distance:", "green"), chamfer_dist)
    print()
    
    if not args.no_viz:
        vis.add_geometry(pcd_est, reset_bounding_box=False)
        vis.add_geometry(gt_pcd)
        vis.run()
