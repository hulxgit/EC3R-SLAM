# adapted from https://github.com/rmurai0610/MASt3R-SLAM/tree/main
import pathlib
#import lietorch
import torch
import numpy as np
import tqdm

# from mast3r_slam.lietorch_utils import as_SE3, lietorch_to_mat
from pykdtree.kdtree import KDTree as pyKDTree


def load_mast3r_slam(reconstruction_file, nanosec=False):
    reconstruction = torch.load(reconstruction_file)
    keyframes = {}
    est_traj = {}
    for keyframe_id, keyframe in reconstruction.items():
        timestamp = float(keyframe["timestamp"])
        if nanosec:
            timestamp /= 1e9
        T_WC, scale = lietorch_to_mat(keyframe["T_WC"], return_scale=True)
        # Apply scaling
        keyframes[timestamp] = {
            "T_WC": T_WC,
            "X": scale * np.array(keyframe["X_canon"], dtype=np.float64),
        }
        est_traj[timestamp] = as_SE3(keyframe["T_WC"]).data.numpy().reshape(-1).tolist()
    return keyframes, est_traj


def load_droid_slam(reconstruction_dir, nanosec=False):
    reconstruction_dir = pathlib.Path(reconstruction_dir)
    # load npy files
    disps = np.load(reconstruction_dir / "disps.npy")
    poses = np.load(reconstruction_dir / "poses.npy")
    timestamps = np.load(reconstruction_dir / "tstamps.npy")
    intrinsics = np.load(reconstruction_dir / "intrinsics.npy")
    keyframes = {}
    est_traj = {}
    for t, disp, pose, intrinsic in zip(timestamps, disps, poses, intrinsics):
        t = float(t)
        if nanosec:
            t /= 1e9

        pts = iproj(disp, intrinsic)
        T_WC = lietorch.SE3(torch.from_numpy(pose.reshape(1, -1)))
        keyframes[t] = {"T_WC": lietorch_to_mat(T_WC), "X": pts}
        est_traj[t] = T_WC.data.numpy().reshape(-1).tolist()
    return keyframes, est_traj


def find_visible_points(points, keyframes, W, H, calib):
    # check which points are visible from the keyframes
    fx, fy, cx, cy = calib
    points = torch.from_numpy(points).to(dtype=torch.float32, device="cuda")
    mask = torch.zeros(points.shape[0], dtype=bool, device="cuda")
    for keyframe in tqdm.tqdm(keyframes.values()):
        # if gt pose is not in the keyframe, it did not register with the ground truth
        if "gt_T_WC" not in keyframe:
            continue
        T_WC = keyframe["gt_T_WC"]
        R = T_WC[:3, :3].reshape(1, 3, 3)
        t = T_WC[:3, 3].reshape(1, 3, 1)
        Rinv = R.transpose(0, 2, 1)
        tinv = -Rinv @ t.reshape(1, 3, 1)
        Rinv = torch.from_numpy(Rinv).to(dtype=torch.float32, device="cuda")
        tinv = torch.from_numpy(tinv).to(dtype=torch.float32, device="cuda")
        point_C = (Rinv @ points.view(-1, 3, 1) + tinv).view(-1, 3)

        # apply projection
        z = point_C[:, 2]
        x = fx * point_C[:, 0] / z + cx
        y = fy * point_C[:, 1] / z + cy

        mask |= (y >= 0) & (y < H) & (x >= 0) & (x < W) & (z > 0)
    print(f"visible points: {mask.sum()} / {mask.shape[0]}")
    return points[mask].to(device="cpu", dtype=torch.float64).numpy()


def chamfer_distance(pcd_ref, pcd_est, max_error):
    ref, est = np.asarray(pcd_ref.points), np.asarray(pcd_est.points)
    kdtree_ref = pyKDTree(ref)
    kdtree_est = pyKDTree(est)
    dist1, _ = kdtree_ref.query(est)
    dist2, _ = kdtree_est.query(ref)
    dist1 = np.clip(dist1, 0, max_error)
    dist2 = np.clip(dist2, 0, max_error)
    chamfer_dist = 0.5 * np.mean(dist1) + 0.5 * np.mean(dist2)

    print("dist1", np.mean(dist1), np.median(dist1), np.max(dist1))
    print("dist2", np.mean(dist2), np.median(dist2), np.max(dist2))

    return chamfer_dist, dist1, dist2


def chamfer_distance_RMSE(pcd_ref, pcd_est, max_error):
    ref, est = np.asarray(pcd_ref.points), np.asarray(pcd_est.points)
    kdtree_ref = pyKDTree(ref)
    kdtree_est = pyKDTree(est)
    dist1, _ = kdtree_ref.query(est)
    dist2, _ = kdtree_est.query(ref)
    dist1 = np.clip(dist1, 0, max_error)
    dist2 = np.clip(dist2, 0, max_error)
    rmse_dist1 = np.sqrt(np.mean(dist1**2))
    rmse_dist2 = np.sqrt(np.mean(dist2**2))
    chamfer_dist = 0.5 * rmse_dist1 + 0.5 * rmse_dist2

    # print("dist1", rmse_dist1, np.mean(dist1), np.median(dist1), np.max(dist1))
    # print("dist2", rmse_dist2, np.mean(dist2), np.median(dist2), np.max(dist2))

    return chamfer_dist, rmse_dist1, rmse_dist2, dist1, dist2


def iproj(disps, intrinsics):
    """pinhole camera inverse projection"""
    ht, wd = disps.shape
    fx, fy, cx, cy = intrinsics
    x, y = np.meshgrid(np.arange(wd), np.arange(ht), indexing="xy")
    i = np.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = np.stack([X, Y, i], axis=-1)
    return pts * (1 / disps[..., None])