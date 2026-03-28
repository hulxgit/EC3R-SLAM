# adapted from https://github.com/rmurai0610/MASt3R-SLAM/tree/main
import sys
import os

import argparse
import pathlib
import copy
import cv2
from termcolor import colored
import trimesh
import random
from tqdm import tqdm
from scipy.spatial import cKDTree as KDTree

import numpy as np
import open3d as o3d
from natsort import natsorted
from scipy.spatial.transform import Rotation
from pykdtree.kdtree import KDTree as pyKDTree

import evo
from evo.core import sync
import evo.core.metrics as metrics
from evo.tools import file_interface
from os.path import join 

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（上级目录）
project_root = os.path.dirname(current_dir)

# 将项目根目录加入 Python 模块搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from evals import geometry_eval_utils as geom_utils

def voxelize_pcd(ori_pcd:trimesh.points.PointCloud, voxel_size=0.01):
    if voxel_size <= 0:
        return ori_pcd
    print(f"Downsample point cloud with voxel size {voxel_size}...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ori_pcd.vertices)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    return trimesh.points.PointCloud(downsampled_pcd.points)

    
def get_align_transformation(rec_pcd, gt_pcd, threshold=0.1):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    print("ICP alignment...")
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(rec_pcd))
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(gt_pcd))
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, o3d_gt_pc, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    transformation = reg_p2p.transformation
    return transformation

def calcu_pcd_fscore(pcd_rec, pcd_gt, align=True, scale=1, vis_dir=None, voxel_size=0.01):
    """
    3D reconstruction metric.
    """
    pcd_rec.vertices /= scale
    pcd_gt.vertices /= scale

    pcd_rec = voxelize_pcd(pcd_rec, voxel_size=voxel_size)
    pcd_gt = voxelize_pcd(pcd_gt, voxel_size=voxel_size)

    if align:
        transformation = get_align_transformation(pcd_rec, pcd_gt, threshold=voxel_size*2)
        pcd_rec = pcd_rec.apply_transform(transformation)

    rec_pointcloud = pcd_rec.vertices.astype(np.float32)
    gt_pointcloud = pcd_gt.vertices.astype(np.float32)

    out_dict = eval_pointcloud_rmse(rec_pointcloud, gt_pointcloud, vis_dir=vis_dir)
    
    return out_dict

def vggt_resize(img, depth, new_size = (224, 224)):
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

def homogeneous(coordinates):
    homogeneous_coordinates = np.hstack((coordinates, np.ones((coordinates.shape[0], 1))))
    return homogeneous_coordinates

def umeyama_alignment(X, Y):
    """
    Perform Umeyama alignment to align two point sets with potential size differences.

    Parameters:
    X (numpy.ndarray): Source point set with shape (N, D).
    Y (numpy.ndarray): Target point set with shape (N, D).

    Returns:
    T (numpy.ndarray): Transformation matrix (D+1, D+1) that aligns X to Y.
    """

    # Calculate centroids
    centroid_X = np.median(X, axis=0)
    centroid_Y = np.median(Y, axis=0)

    # Center the point sets
    X_centered = X - centroid_X
    Y_centered = Y - centroid_Y

    '''
    # Covariance matrix
    sigma = np.dot(X_centered.T, Y_centered) / X.shape[0]
    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(sigma)
    # Ensure a right-handed coordinate system
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Vt[-1] = -Vt[-1]
        U[:, -1] = -U[:, -1]
    # Rotation matrix
    R = np.dot(Vt.T, U.T)
    #'''

    # solve rotation using svd with rectification.
    S = np.dot(X_centered.T, Y_centered)
    U, _, VT = np.linalg.svd(S)
    rectification = np.eye(3)
    rectification[-1,-1] = np.linalg.det(VT.T @ U.T)
    R = VT.T @ rectification @ U.T 

    # Scale factor
    sx = np.median(np.linalg.norm(X_centered, axis=1))
    sy = np.median(np.linalg.norm(Y_centered, axis=1))
    c = sy / sx

    # Translation
    t = centroid_Y - c * np.dot(R, centroid_X)

    # Transformation matrix
    T = np.zeros((X.shape[1] + 1, X.shape[1] + 1))
    T[:X.shape[1], :X.shape[1]] = c * R
    T[:X.shape[1], -1] = t
    T[-1, -1] = 1

    return T

def SKU_RANSAC(src_pts, tar_pts):
    random.seed(args.seed)
    # generate and vote the best hypo.
    N_HYPO = 512
    ERR_MIN = 8888.
    Rt_init = np.identity(4)
    for hid in tqdm(range(N_HYPO), desc="Running umayama RANSAC"):
        ids = random.sample(range(len(src_pts)), 3)
        s_mini = src_pts[ids]
        t_mini = tar_pts[ids]
        hypo = umeyama_alignment(s_mini, t_mini)
        x = (hypo @ homogeneous(src_pts).transpose())[0:3] 
        y = homogeneous(tar_pts).transpose()[0:3]
        residuals = np.linalg.norm(x-y, axis=0)

        med_err = np.median(residuals)
        if ERR_MIN > med_err:
            ERR_MIN = med_err
            Rt_init = hypo
    # print("ERR_MIN", ERR_MIN)

    # todo: count inlier instead of median error.
    # todo: refine with inliers.

    return Rt_init


def align_pcd(source:np.array, target:np.array, icp=None, init_trans=None, mask=None, return_trans=True, voxel_size=0.1):
    """ Align the scale of source to target using umeyama,
    then refine the alignment using ICP.
    """
    if init_trans is not None:
        source = trimesh.transformations.transform_points(source, init_trans)
    #####################################
    # first step registration using umeyama.
    #####################################
    source_for_align = source if mask is None else source[mask]
    target = target if mask is None else target[mask]
    N = min(source_for_align.shape[1], target.shape[1])  # 取较小的点数

   
    Rt_step1 = SKU_RANSAC(source_for_align, target)
    source_step1 = (Rt_step1 @ homogeneous(source_for_align).transpose())[0:3].transpose()
    #####################################
    # second step registration using icp.
    #####################################
    print("point-to-plane ICP...")
    icp_thr = voxel_size * 2

    pcd_source_step1 = o3d.geometry.PointCloud()
    pcd_source_step1.points = o3d.utility.Vector3dVector(source_step1)
    pcd_source_step1 = pcd_source_step1.voxel_down_sample(voxel_size=voxel_size)
    
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target)
    pcd_target = pcd_target.voxel_down_sample(voxel_size=voxel_size)
    pcd_target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    if icp == "point":
        icp_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    elif icp == 'plain':
        icp_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        raise ValueError
    reg_p2l = o3d.pipelines.registration.registration_icp(
        pcd_source_step1, pcd_target, icp_thr, np.identity(4), icp_method)

    Rt_step2 = reg_p2l.transformation
    
    # apply RT on initial source without downsample
    transformation_s2t = Rt_step2 @ Rt_step1
    transformed_source = trimesh.transformations.transform_points(source, transformation_s2t)
    if return_trans:
        return transformed_source, transformation_s2t
    else:
        return transformed_source

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src, workers=8)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product
def calcu_pair_loss(align_gt_pcd, align_pred_pcd,
                    eval_gt_pcd=None, eval_pred_pcd=None, 
                    c2w=None, vis_dir=None,icp='plain',
                    voxel_size=0.01):
    """
    Keep the original scale of gt.
    First align the predicted pcd to the gt pcd with umeyama+icp,
    then calculating reconstruction metrics.
    """
    if eval_gt_pcd is None:
        eval_gt_pcd = align_gt_pcd
    if eval_pred_pcd is None:
        eval_pred_pcd = align_pred_pcd
    
    # align the predicted pcd to the gt pcd with umeyama+icp
    _, trans = align_pcd(align_pred_pcd, align_gt_pcd, 
                         init_trans=c2w, 
                         mask=None,
                         icp=icp, 
                         return_trans=True,
                         voxel_size=voxel_size*2)
    
    aligned_eval_pred_pcd = trimesh.transformations.transform_points(eval_pred_pcd, trans) 

    aligned_eval_pred_pcd = trimesh.points.PointCloud(aligned_eval_pred_pcd)
    gt_pcd = trimesh.points.PointCloud(eval_gt_pcd)

    # Calculate the reconstruction metrics
    res2 = calcu_pcd_fscore(aligned_eval_pred_pcd, gt_pcd, 
                            scale=1, align=True, vis_dir=vis_dir,
                            voxel_size=voxel_size)  
    align_flag = True
    # if res2["completeness"] > 10 or res2["accuracy"] > 10:
    #     align_flag = False

    return res2, align_flag

def get_threshold_percentage(dist, thresholds):
    """Evaluates a point cloud.

    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [(dist <= t).mean() for t in thresholds]
    return in_threshold




def eval_pointcloud_rmse(
    pointcloud,
    pointcloud_tgt,
    normals=None,
    normals_tgt=None,
    thresholds=np.linspace(1.0 / 1000, 1, 1000),
    vis_dir=None
):
    """
    Evaluate point cloud reconstruction quality with RMSE-style outputs.
    
    Returns:
        chamfer_rmse: float, symmetric RMSE-Chamfer distance
        accuracy_rmse: float, RMSE from predicted to target (est → ref)
        completeness_rmse: float, RMSE from target to predicted (ref → est)
        accuracy_mae: float, MAE version of accuracy
        completeness_mae: float, MAE version of completeness
        f_score_1: float, F-score @ ~1%
        f_score_15: float, F-score @ ~1.5%
        normals_acc: float, normal consistency (0.5*comp + 0.5*acc)
        comp_ratio_5: float, ratio of target points within 5cm
        accuracy_distances: np.array, per-point distance (est → ref)
        completeness_distances: np.array, per-point distance (ref → est)
    """
    assert len(pointcloud) > 0, "Empty pointcloud"

    pointcloud = np.asarray(pointcloud)
    pointcloud_tgt = np.asarray(pointcloud_tgt)

    # -------------------------------
    # 1. Completeness: ref → est
    # -------------------------------
    completeness_dist, completeness_normals = distance_p2p(
        pointcloud_tgt, normals_tgt, pointcloud, normals
    )
    completeness_mae = completeness_dist.mean()
    completeness_rmse = np.sqrt((completeness_dist ** 2).mean())
    completeness_normals_mean = completeness_normals.mean()

    comp_ratio_5 = (completeness_dist < 0.05).astype(float).mean()

    recall = get_threshold_percentage(completeness_dist, thresholds)

    if vis_dir is not None:
        save_vis(pointcloud_tgt, completeness_dist, join(vis_dir, "completeness.ply"))

    # -------------------------------
    # 2. Accuracy: est → ref
    # -------------------------------
    accuracy_dist, accuracy_normals = distance_p2p(
        pointcloud, normals, pointcloud_tgt, normals_tgt
    )
    accuracy_mae = accuracy_dist.mean()
    accuracy_rmse = np.sqrt((accuracy_dist ** 2).mean())
    accuracy_normals_mean = accuracy_normals.mean()

    precision = get_threshold_percentage(accuracy_dist, thresholds)

    if vis_dir is not None:
        save_vis(pointcloud, accuracy_dist, join(vis_dir, "accuracy.ply"))

    # -------------------------------
    # 3. Chamfer Distance (RMSE style)
    # -------------------------------
    chamfer_rmse = 0.5 * accuracy_rmse + 0.5 * completeness_rmse  # RMSE-Chamfer
    chamfer_l1 = 0.5 * accuracy_mae + 0.5 * completeness_mae      # MAE-Chamfer

    # -------------------------------
    # 4. F-Score
    # -------------------------------
    F = [
        2 * p * r / (p + r + 1e-8)
        for p, r in zip(precision, recall)
    ]
    f_score_1  = F[9]   # ~1.0%
    f_score_15 = F[14]  # ~1.5%
    f_score_20 = F[19]  # ~2.0%

    # -------------------------------
    # 5. Normals
    # -------------------------------
    normals_acc = 0.5 * completeness_normals_mean + 0.5 * accuracy_normals_mean

    # -------------------------------
    # 6. Return
    # -------------------------------
    print("\n" + "-"*40)
    print("RMSE Metrics")
    print("-"*40)
    print(f"accuracy_rmse: {accuracy_rmse:.6f}")
    print(f"completeness_rmse: {completeness_rmse:.6f}")
    print(f"chamfer_rmse: {chamfer_rmse:.6f}")

    print("\n" + "-"*40)
    print("MAE Metrics")
    print("-"*40)
    print(f"accuracy_mae: {accuracy_mae:.6f}")
    print(f"completeness_mae: {completeness_mae:.6f}")
    print(f"chamfer_l1: {chamfer_l1:.6f}")
    return (
        chamfer_rmse,
        accuracy_rmse,
        completeness_rmse,
        accuracy_mae,
        completeness_mae,
        chamfer_l1,
        f_score_1,
        f_score_15,
        f_score_20,
        normals_acc,
        comp_ratio_5,
        accuracy_dist,           # raw arrays for debug/vis
        completeness_dist
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_pcd", default="/home/lingxianghu/SLAM3R/results/gt/replica_288512/office0_pcds.npy")
    parser.add_argument("--res_dir", default="/home/lingxianghu/EC3R-SLAM/results/replica_recon/office0/")
    parser.add_argument("--gt", default="/home/lingxianghu/datasets/Replica/office0/traj_tum.txt")
    parser.add_argument("--est", default="/home/lingxianghu/EC3R-SLAM/results/replica_recon/office0/trajectory_kf.txt")

    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--seed", type=int, default=666, help="Random seed (default: 666)")
    args = parser.parse_args()


    eval_conf_thres = 3
    num_sample_points = 200000
    voxelize_size = 0.005
    gt_pcd = np.load(args.gt_pcd).astype(np.float32)
    valid_masks = np.load(args.gt_pcd.replace('_pcds', '_valid_masks')).astype(bool)
    pred_pcd = np.load(join(args.res_dir,  'kfpoint.npy')).astype(np.float32)
    pred_confs = np.load(join(args.res_dir,  'kfconf.npy')).astype(np.float32)
    print(pred_confs.shape)
    print(pred_pcd.shape)


    _,H,W = valid_masks.shape
    pred_confs = pred_confs.reshape(-1,H,W)


    traj_ref = file_interface.read_tum_trajectory_file(args.gt)
    traj_est = file_interface.read_tum_trajectory_file(args.est)
    matches = sync.matching_time_indices(traj_ref.timestamps, traj_est.timestamps)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    traj_est_aligned = copy.deepcopy(traj_est)
    r_a, t_a, s = traj_est_aligned.align(traj_ref, correct_scale=True, correct_only_scale=False)
    t_a = t_a.reshape(3, 1)

    # traj_est_aligned_poses = traj_est_aligned.poses_se3
    traj_est_poses = traj_est.poses_se3


    pcd_gt = []
    masks_valid = []
    pcd_est_list = []
    for a, b in zip(matches[0], matches[1]):
        masks_valid.append(valid_masks[int(a)])
        gt_p = gt_pcd[int(a)]
        pcd_gt.append(gt_p)

    # filter out points with conficence and valid masks
    pcd_gt = np.array(pcd_gt)
    masks_valid = np.array(masks_valid)
    print(pcd_gt.shape)
    valid_masks = masks_valid
    print(valid_masks.shape)
    pred_pcd = pred_pcd.reshape(-1,3)
        
    pred_pcd = ((s*r_a) @ pred_pcd.T + t_a).T
    
    pred_pcd = pred_pcd.reshape(-1,H,W,3)


    pred_confs[~valid_masks] = 0
    valid_ids = pred_confs > eval_conf_thres
    gt_pcd = pcd_gt[valid_ids]
    pred_pcd = pred_pcd[valid_ids]

    
    # prepare the pcds for alignment and evaluation
    assert gt_pcd.shape[0] > num_sample_points
    sample_ids = np.random.choice(gt_pcd.shape[0], num_sample_points, replace=False)
    gt_pcd = gt_pcd[sample_ids]
    pred_pcd = pred_pcd[sample_ids]

    metric_dict, align_succ = calcu_pair_loss(gt_pcd, 
                                            pred_pcd,
                                            c2w=None,
                                            vis_dir=None,
                                            voxel_size=voxelize_size,
                                            )



