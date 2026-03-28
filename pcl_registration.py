import open3d as o3d
import numpy as np
import os
import time
from scipy.linalg import lstsq
import torch
from torch import nn
from pypose import knn, is_SE3
from pypose.utils.stepper import ReduceToBason
#with confindence
import argparse, os
import pypose as pp
def filter_poionts(pcd,conf,N,min_conf_thr_percentile= 20):
    #for visualization 
    points = np.asarray(pcd.points)[N:]
    colors = np.asarray(pcd.colors)[N:]
    conf = conf[N:]
    conf_thr = np.percentile(conf, min_conf_thr_percentile)
    mask = conf > conf_thr
    points = points[mask]
    colors = colors[mask]
    return (points,colors)
def project_point3d_to_pixel(point3d, K):
    """
    point3d: (N, 3)
    K: (3, 3)
    返回 (N, 2) 像素坐标
    """
    X, Y, Z = point3d[:, 0], point3d[:, 1], point3d[:, 2]
    u = (K[0, 0] * X) / Z + K[0, 2]
    v = (K[1, 1] * Y) / Z + K[1, 2]
    return torch.stack([u, v], dim=1)
def point_regist(frame1, frame2,pointkeys,overlap):
    N = overlap*384 * 512
    num_samples = 50000
    point1 = np.array(frame1[pointkeys])

    color1 = np.array(frame1["rgb"])
    confidence1 = np.array(frame1["local_confidence_maps"][-overlap:])
    point2 = np.array(frame2[pointkeys])
    point2_original_shape = point2.shape
    color2 = np.array(frame2["rgb"])
    confidence2 = np.array(frame2["local_confidence_maps"][:overlap])
    allconf2 =  np.array(frame2["local_confidence_maps"]).reshape(-1)

    pts1 = point1.reshape(-1, 3)
    pts2 = point2.reshape(-1, 3)


    col1 = color1.reshape(-1, 3)
    col2 = color2.reshape(-1, 3)
    conf1 = confidence1.reshape(-1)
    conf2 = confidence2.reshape(-1)
    conf_avg =  np.sqrt(np.abs(conf1 * conf2))
    col1 = col1.astype(np.float32) 
    col2 = col2.astype(np.float32) 


    # 构建 Open3D 点云对象
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd1.colors = o3d.utility.Vector3dVector(col1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts2)
    pcd2.colors = o3d.utility.Vector3dVector(col2)

    sorted_indices = np.argsort(-conf_avg)  # 按置信度从高到低排序
    indices = sorted_indices[:num_samples]  # 选出 top-K 高置信度点
    # ➤ 2. 提取匹配的点和颜色（记得 pts2 要加 offset）

    matched_pts1 = pts1[-N:][indices]
    matched_col1 = col1[-N:][indices]

    matched_pts2 =pts2[:N][indices]
    matched_col2 = col2[:N][indices]
    print("🔹 前五个点的坐标1：\n", pts1[-N:][:5])
    print("🔹 前五个点的坐标2：\n", pts2[:N][:5])
    print("🔹 len\n", len(matched_pts2))
    print("🔹 indices\n", indices[:20])

    source_points = matched_pts2  # 要被配准（蓝色）

    target_points = matched_pts1  # 参考基准（黄色）


    # 转成 open3d 点云对象
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    # source_pcd.colors = o3d.utility.Vector3dVector(matched_col2)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    # target_pcd.colors = o3d.utility.Vector3dVector(matched_col1)

    # 构建对应关系：假设一一对应
    corres = o3d.utility.Vector2iVector([[i, i] for i in range(len(matched_pts1))])

    # # 估计变换 变换1
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)

    transformation = est.compute_transformation(source_pcd, target_pcd, corres)
    pcd2.transform(transformation)
    print("📝 Transformation Matrix:\n", transformation)




    pointcolor = filter_poionts(pcd2,allconf2,N)




    point2_restored =np.asarray(pcd2.points).reshape(point2_original_shape)
    point2_list = [point2_restored[i] for i in range(point2_restored.shape[0])]

    return pointcolor,point2_list
def svdtf_sim3(source, target, weights=None):
    if not source.dtype.is_floating_point or not target.dtype.is_floating_point:
        raise TypeError("Input tensors must be floating point.")

    assert source.size(-2) == target.size(-2), "Source and target must have the same number of points."
    assert source.size(-1) == 3 and target.size(-1) == 3, "Points must be 3D."

    if weights is None:
        weights = torch.ones(source.shape[:-2] + (source.shape[-2],), device=source.device, dtype=source.dtype)
    weights = weights / weights.sum()

    ctnsource = (weights[..., None] * source).sum(dim=-2, keepdim=True)
    ctntarget = (weights[..., None] * target).sum(dim=-2, keepdim=True)

    source_centered = source - ctnsource
    target_centered = target - ctntarget

    M = torch.einsum('...Ni,...Nj,...N->...ij', target_centered, source_centered, weights)

    U, _, Vh = torch.linalg.svd(M)
    R = U @ Vh
    det_R = torch.linalg.det(R)
    mask_reflection = det_R < 0
    R[mask_reflection] = -R[mask_reflection]

    numerator = (weights[..., None] * (target_centered * (R @ source_centered.transpose(-2, -1)).transpose(-2, -1))).sum(dim=(-2, -1))
    sum_sq_source = (weights[..., None] * (source_centered ** 2)).sum(dim=(-2, -1))
    s = numerator / (sum_sq_source + 1e-9)
    s = s.clamp(min=1e-9)

    t = ctntarget.squeeze(-2) - s.unsqueeze(-1) * (R @ ctnsource.squeeze(-2).unsqueeze(-1)).squeeze(-1)

    T_4x4 = torch.eye(4, device=source.device, dtype=source.dtype).unsqueeze(0).repeat(*R.shape[:-2], 1, 1)
    T_4x4[..., :3, :3] = s.unsqueeze(-1).unsqueeze(-1) * R
    T_4x4[..., :3, 3] = t

    return pp.mat2Sim3(T_4x4, check=False)

def scan_to_map_trans(old_idx, old_points, old_confs, new_points, new_confs, new_idx,K):
    points1 = []
    points2 = []
    confs1 = []
    confs2 = []
    match_idxs = []
    for i, (point1, conf1) in enumerate(zip(old_points, old_confs)):
        if old_idx[i] in new_idx:
            new_position = new_idx.index(old_idx[i])
            points1.append(point1)
            confs1.append(conf1)
            points2.append(new_points[new_position])
            confs2.append(new_confs[new_position])
            match_idxs.append(old_idx[i]) 
    num_samples = 50000 * len(points1)

    if len(points1) > 0:
        points1 = torch.stack(points1).view(-1, 3)
    else:
        points1 = torch.empty((0, 3))
    if len(points2) > 0:
        points2 = torch.stack(points2).view(-1, 3)
    else:
        points2 = torch.empty((0, 3))
    if len(confs1) > 0:
        confs1 = torch.stack(confs1).view(-1)
    else:
        confs1 = torch.empty((0,))
    if len(confs2) > 0:
        confs2 = torch.stack(confs2).view(-1)
    else:
        confs2 = torch.empty((0,))

    device = points1.device

    # 置信度加权，取几何平均
    conf_avg = torch.sqrt(torch.abs(confs1 * confs2))

    # 排序，取前 num_samples 个高置信度对应点
    sorted_indices = torch.argsort(-conf_avg)
    indices = sorted_indices[:num_samples]

    matched_pts1 = points1[indices]
    matched_pts2 = points2[indices]
    matched_conf = conf_avg[indices]
    weights = torch.exp(matched_conf)
    weights = weights / weights.sum()
    # ICP with weights!
    source_points = matched_pts2  # 要被配准
    target_points = matched_pts1  # 参考基准
    weights = torch.exp(matched_conf)
    weights = weights / weights.sum()

    # 初始位姿平移部分：按加权均值
    centroid_src = (weights[:, None] * source_points).sum(dim=0, keepdim=True)
    centroid_tgt = (weights[:, None] * target_points).sum(dim=0, keepdim=True)
    centroid_shift = (centroid_tgt - centroid_src).squeeze(0)

    init_trans = pp.mat2Sim3(torch.eye(4).unsqueeze(0).to(device))
    transformation_matrix = pp.matrix(init_trans)
    transformation_matrix[0, :3, 3] = centroid_shift
    init_trans = pp.mat2Sim3(transformation_matrix)

    stepper = pp.utils.ReduceToBason(steps=500, patience=10, decreasing=1e-4, verbose=False)

    try:
        # 全量数据执行（不抽样）
        icp = ICP(init_trans, stepper=stepper).to(device)
        est = icp(
            source_points.to(device),
            target_points.to(device),
            weights=weights.to(device) if weights is not None else None
        )
        print("全量数据运行成功，继续执行后续步骤")
        transformation = pp.matrix(est)
        # 正常执行后续逻辑...

    except Exception as e:
        print(f"全量数据运行出错: {e}，尝试多次抽样验证...")
        # 转成 open3d 点云对象
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)

        corres = o3d.utility.Vector2iVector([[i, i] for i in range(len(matched_pts1))])
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
        est_matrix = est.compute_transformation(source_pcd, target_pcd, corres)
        transformation = torch.from_numpy(est_matrix)
        transformation = transformation.to(dtype=torch.float32)    # 转换变换矩阵

    
    # ✅ 
    N = points2.shape[0]
    source_points_h = torch.cat([points2, torch.ones(N, 1, device=device)], dim=1)  # (N, 4)

    # ⚠️ 不再用 .T，直接左乘 4x4 变换矩阵
    transformed_points_h = source_points_h @ transformation  # (N, 4) 或 (B, N, 4)

    # ✅ 只取前 3 个坐标，兼容多维度
    transformed_points = transformed_points_h[..., :3]

    # ✅ 确保 target_points 也在同一维度
    diffs = transformed_points - points1


    # source_points, target_points 都 shape (N, 3)
    source_pixels = project_point3d_to_pixel(source_points, K)
    target_pixels = project_point3d_to_pixel(target_points, K)

    # 2D residual
    pixel_errors = torch.norm(source_pixels - target_pixels, dim=1)
    # 逐点欧氏距离
    errors = np.linalg.norm(diffs, axis=1)  # (N,)

    # 平均 RMSE
    rmse = np.sqrt(np.mean(errors ** 2))


    print("整体 RMSE:", rmse)

    # print(f"✅  第 {match_idx} 个匹配点的 RMSE: {pixel_errors.item():.6f}")
    print(f"✅  第 {match_idxs[-1]} 个匹配点的 RMSE: { torch.sqrt(torch.mean(pixel_errors ** 2))}")

    return transformation,match_idxs

class ICP(nn.Module):
    def __init__(self, init=None, stepper=None):
        super().__init__()
        self.stepper = ReduceToBason(steps=200) if stepper is None else stepper
        self.init = init

    def forward(self, source, target, ord=2, dim=-1, init=None, weights=None):
        temporal = source
        init = init if init is not None else self.init

        if init is not None:
            temporal = init.unsqueeze(-2) @ temporal

        batch = torch.broadcast_shapes(source.shape[:-2], target.shape[:-2])
        self.stepper.reset()

        target = target.expand(batch + target.shape[-2:])

        while self.stepper.continual():
            diff = temporal - target
            # 加权均值误差
            error = torch.sqrt(torch.sum(diff ** 2, dim=dim))
            if weights is not None:
                error = (error * weights).sum()
            else:
                error = error.mean()

            # 使用带权 svdtf_sim3
            T = svdtf_sim3(temporal, target, weights=weights)
            temporal = T.unsqueeze(-2) @ temporal
            self.stepper.step(error)

        return svdtf_sim3(source, temporal, weights=weights)



def scan_to_map_loop(points1,confs1,points2,confs2 ):

    num_samples = 50000

    # 📝 合并到单个大 Tensor
    if len(points1) > 0:
        points1 = points1.view(-1, 3)
    else:
        points1 = torch.empty((0, 3))

    if len(points2) > 0:
        points2 = points2.view(-1, 3)
    else:
        points2 = torch.empty((0, 3))

    if len(confs1) > 0:
        confs1 = confs1.view(-1)
    else:
        confs1 = torch.empty((0,))

    if len(confs2) > 0:
        confs2 = confs2.view(-1)
    else:
        confs2 = torch.empty((0,))
    device = points1.device

    conf_avg = torch.sqrt(torch.abs(confs1 * confs2))

    # 使用 `torch.argsort` 进行排序，降序排列
    sorted_indices = torch.argsort(-conf_avg)  # 负号表示降序

    # 选出 top-K 的高置信度点
    indices = sorted_indices[:num_samples]

    # 从原始点集中选择这些高置信度对应的点
    matched_pts1 = points1[indices]
    matched_pts2 = points2[indices]
    matched_conf = conf_avg[indices]

    # ICP with weights!
    source_points = matched_pts2  # 要被配准
    target_points = matched_pts1  # 参考基准
    weights = torch.exp(matched_conf)
    weights = weights / weights.sum()
    # 初始位姿平移部分：按加权均值
    centroid_src = (weights[:, None] * source_points).sum(dim=0, keepdim=True)
    centroid_tgt = (weights[:, None] * target_points).sum(dim=0, keepdim=True)
    centroid_shift = (centroid_tgt - centroid_src).squeeze(0)  # (3,)
    init_trans = pp.mat2Sim3(torch.eye(4).unsqueeze(0).to(device))

    # 🔍 先将 LieTensor 转换为普通矩阵
    transformation_matrix = pp.matrix(init_trans)

    # 🔄 更新平移部分
    transformation_matrix[0, :3, 3] = centroid_shift
    stepper = pp.utils.ReduceToBason(steps=500, patience=10,
                                    decreasing=1e-4, verbose=False)
    try:
        # 全量数据执行（不抽样）
        icp = ICP(init_trans, stepper=stepper).to(device)
        est = icp(
            source_points.to(device),
            target_points.to(device),
            weights=weights.to(device) if weights is not None else None
        )
        print("全量数据运行成功，继续执行后续步骤")
        transformation = pp.matrix(est)
        # 正常执行后续逻辑...

    except Exception as e:
        print(f"全量数据运行出错: {e}，尝试多次抽样验证...")
        # 转成 open3d 点云对象
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)

        corres = o3d.utility.Vector2iVector([[i, i] for i in range(len(matched_pts1))])
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
        est_matrix = est.compute_transformation(source_pcd, target_pcd, corres)
        transformation = torch.from_numpy(est_matrix)
        transformation = transformation.to(dtype=torch.float32)    # 转换变换矩阵
    return transformation