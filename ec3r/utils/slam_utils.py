import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F
import math
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
import cv2
import random
import numpy as np

@torch.inference_mode()
def match(feats1, feats2, min_cossim = 0.82):

    cossim = feats1 @ feats2.t()
    cossim_t = feats2 @ feats1.t()
    
    _, match12 = cossim.max(dim=1)
    _, match21 = cossim_t.max(dim=1)

    idx0 = torch.arange(len(match12), device=match12.device)
    mutual = match21[match12] == idx0

    if min_cossim > 0:
        cossim, _ = cossim.max(dim=1)
        good = cossim > min_cossim
        idx0 = idx0[mutual & good]
        idx1 = match12[mutual & good]
    else:
        idx0 = idx0[mutual]
        idx1 = match12[mutual]

    return idx0, idx1


def pad_tensor_to(tensor, max_len=4096, device=None):
    """
    将任意形状为 (N, M) 的张量填充为 (max_len, M)，其余部分填充为 0。

    参数:
        tensor (torch.Tensor): 输入张量，形状为 (N, M)
        max_len (int): 填充后的目标长度
        device (str or torch.device, optional): 输出张量的目标设备，默认为 None 表示保持一致

    返回:
        torch.Tensor: 形状为 (max_len, M) 的填充张量
    """
    N, M = tensor.shape
    if device is None:
        device = tensor.device
    padded = torch.zeros((max_len, M), dtype=tensor.dtype, device=device)
    copy_len = min(N, max_len)
    padded[:copy_len] = tensor[:copy_len]
    return padded
def triangulate_and_rescale_depth(K, T0_w2c, T1_w2c, pts0, pts1, average_depth, smooth_scale=0.1, filter_percent=0.05):
    """
    Triangulate 3D points and rescale their depth after filtering extreme values and negatives.

    返回：
    - normalized_points3d: (M, 3) ndarray, 经过归一化后、去掉异常值的 3D 点
    - valid_mask: (N,) bool ndarray, True 表示该点是有效点（未被过滤）
    """
    # 1️⃣ 三角化
    P0 = K @ T0_w2c[:3, :]
    P1 = K @ T1_w2c[:3, :]
    pts0 = np.asarray(pts0, dtype=np.float32).T  # (2, N)
    pts1 = np.asarray(pts1, dtype=np.float32).T
    pts4d = cv2.triangulatePoints(P0, P1, pts0, pts1)
    pts4d /= pts4d[3]
    recovered_points3d = pts4d[:3].T  # (N, 3)

    # 2️⃣ 深度
    depths = recovered_points3d[:, 2]

    # 3️⃣ 过滤负深度 + 过滤 top & bottom filter_percent 的极值
    lower_bound = np.percentile(depths, filter_percent * 100)
    upper_bound = np.percentile(depths, (1 - filter_percent) * 100)
    valid_mask = (depths > 0) & (depths >= lower_bound) & (depths <= upper_bound)

    # 4️⃣ 提取有效 3D 点
    valid_points3d = recovered_points3d
    valid_depths = valid_points3d[:, 2]

    # 5️⃣ sigmoid 平滑归一化
    depths_centered = valid_depths - np.median(valid_depths)
    depths_sigmoid = 1 / (1 + np.exp(-smooth_scale * depths_centered))

    # 6️⃣ 缩放系数，让平均深度等于 average_depth
    rescale_factor = average_depth / np.mean(depths_sigmoid)
    depths_normalized = depths_sigmoid * rescale_factor

    # 7️⃣ 替换回 Z 坐标
    valid_points3d[:, 2] = depths_normalized

    return valid_points3d, valid_mask
def compute_average_depth(selected_points3dtrue, keyframe_pose):
    """
    计算选定的 3D 点在相机坐标系下的平均深度。

    参数：
    - selected_points3dtrue: (N, 3) ndarray，世界坐标系下的 3D 点。
    - keyframe_pose: (4, 4) ndarray，相机的 4x4 位姿矩阵 (世界到相机的变换)。

    返回：
    - 平均深度（相机坐标系下 Z 坐标的均值）。
    """
    # 1️⃣ 将 3D 点转换为齐次坐标 (N, 4)
    points_hom = np.hstack((selected_points3dtrue, np.ones((selected_points3dtrue.shape[0], 1))))

    # 2️⃣ 用 keyframe pose 变换到相机坐标系
    # 如果 keyframe_pose 是世界到相机的变换矩阵
    # 则：camera_points = keyframe_pose @ points_hom.T
    # 这里 .T 变成 (4, N)，然后再 .T 回来
    camera_points = (keyframe_pose @ points_hom.T).T

    # 3️⃣ 取 Z 坐标
    depths = camera_points[:, 2]

    # 4️⃣ 计算平均深度
    avg_depth = np.mean(depths)

    return avg_depth

def select_keyframes(initposes, rotation_threshold=5, translation_threshold=0.05):
    """
    根据旋转和平移的变化，选择关键帧索引 (基于 numpy 实现)

    参数:
    - initposes: List[np.ndarray] 4x4 的变换矩阵 (c2w)
    - rotation_threshold: 旋转角度的阈值 (degree)
    - translation_threshold: 平移距离的阈值

    返回:
    - keyframe_indices: List[int] 关键帧的索引
    """
    def rotation_angle(R1, R2):
        """ 计算两个旋转矩阵之间的角度差 """
        R = R1 @ R2.T
        trace = np.trace(R)
        trace = np.clip(trace, -1.0, 3.0)  # 避免浮点数误差
        angle = np.arccos((trace - 1) / 2)
        return np.degrees(angle)

    def translation_distance(t1, t2):
        """ 计算两个平移向量之间的欧式距离 """
        return np.linalg.norm(t1 - t2)

    # 初始化关键帧列表，默认第一个是关键帧
    keyframe_indices = [0]
    last_pose = initposes[0]

    # 遍历所有帧
    for i in range(1, len(initposes)):
        current_pose = initposes[i]
        
        # 提取旋转和平移
        R_last, t_last = last_pose[:3, :3], last_pose[:3, 3]
        R_curr, t_curr = current_pose[:3, :3], current_pose[:3, 3]
        
        # 计算旋转角度和位移距离
        angle_diff = rotation_angle(R_last, R_curr)
        trans_diff = translation_distance(t_last, t_curr)

        # 判断是否超过阈值
        if angle_diff > rotation_threshold or trans_diff > translation_threshold:
            keyframe_indices.append(i)
            last_pose = current_pose

    # # 确保最后一帧是关键帧
    if keyframe_indices[-1] != len(initposes) - 1:
        keyframe_indices.append(len(initposes) - 1)

    return keyframe_indices
def filter_points_in_front_of_cameras(points3d_w, T0_w2c, T1_w2c):
    """
    保留那些在两个相机坐标系下 Z > 0 的三角点
    """
    N = points3d_w.shape[0]
    points4d_h = np.hstack([points3d_w, np.ones((N, 1))])  # (N, 4)

    pts_cam0 = (T0_w2c @ points4d_h.T).T[:, :3]
    pts_cam1 = (T1_w2c @ points4d_h.T).T[:, :3]

    mask = (pts_cam0[:, 2] > 0) & (pts_cam1[:, 2] > 0)
    return mask
def compute_mean_std_filter(matched_pts1, matched_pts2, k=2.0):
    """
    用 mean+std 剔除离群点，并打印平均位移
    """
    displacements = np.linalg.norm(matched_pts2 - matched_pts1, axis=1)
    mean_disp = np.mean(displacements)
    std_disp = np.std(displacements)

    # 合理范围：[mean - k*std, mean + k*std]
    lower = mean_disp - k * std_disp
    upper = mean_disp + k * std_disp
    mask = (displacements >= lower) & (displacements <= upper)

    filtered_pts1 = matched_pts1[mask]
    filtered_pts2 = matched_pts2[mask]

    # print(f"Mean displacement (after filtering): {mean_disp:.2f} pixels")
    # print(f"Kept {len(filtered_pts1)} / {len(matched_pts1)} matches")

    return filtered_pts1, filtered_pts2, mean_disp,std_disp

def print_pose_difference(T1: np.ndarray, T2: np.ndarray, idx1: int, idx2: int):
    """
    打印两个 SE(3) pose 之间的旋转角度和平移距离，并显示它们的帧编号。

    参数：
        T1: np.ndarray, shape (4, 4) 第一个位姿
        T2: np.ndarray, shape (4, 4) 第二个位姿
        idx1: int，第一个帧的编号
        idx2: int，第二个帧的编号
    """
    assert T1.shape == (4, 4) and T2.shape == (4, 4), "输入必须是两个 4x4 的变换矩阵"

    delta_T = np.linalg.inv(T1) @ T2
    R = delta_T[:3, :3]
    t = delta_T[:3, 3]

    # 平移距离
    translation = np.linalg.norm(t)

    # 旋转角度
    trace_R = np.trace(R)
    cos_theta = (trace_R - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    rotation_deg = np.degrees(theta_rad)

    print(f"🔁 帧 {idx1} → 帧 {idx2}")
    print(f"   🌀 旋转角度: {rotation_deg:.4f}°")
    print(f"   📏 平移距离: {translation:.4f} m")

    return rotation_deg, translation


def np2torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Converts a NumPy ndarray to a PyTorch tensor.
    Args:
        array: The NumPy ndarray to convert.
        device: The device to which the tensor is sent. Defaults to 'cpu'.

    Returns:
        A PyTorch tensor with the same data as the input array.
    """
    return torch.from_numpy(array).float().to(device)
def torch2np(tensor: torch.Tensor) -> np.ndarray:
    """ Converts a PyTorch tensor to a NumPy ndarray.
    Args:
        tensor: The PyTorch tensor to convert.
    Returns:
        A NumPy ndarray with the same data and dtype as the input tensor.
    """
    return tensor.detach().cpu().numpy()



def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """
    Creates a 1D Gaussian kernel.

    Args:
        window_size: The size of the window for the Gaussian kernel.
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        The 1D Gaussian kernel.
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> Variable:
    """
    Creates a 2D Gaussian window/kernel for SSIM computation.

    Args:
        window_size: The size of the window to be created.
        channel: The number of channels in the image.

    Returns:
        A 2D Gaussian window expanded to match the number of channels.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: Variable, window_size: int,
          channel: int, size_average: bool = True) -> torch.Tensor:
    """
    Internal function to compute the Structural Similarity Index (SSIM) between two images.

    Args:
        img1: The first image.
        img2: The second image.
        window: The Gaussian window/kernel for SSIM computation.
        window_size: The size of the window to be used in SSIM computation.
        channel: The number of channels in the image.
        size_average: If True, averages the SSIM over all pixels.

    Returns:
        The computed SSIM value.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    Args:
        img1: The first image.
        img2: The second image.
        window_size: The size of the window to be used in SSIM computation. Defaults to 11.
        size_average: If True, averages the SSIM over all pixels. Defaults to True.

    Returns:
        The computed SSIM value.
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)

def get_loss_tracking_es(config, image, depth, opacity, 
                         viewpoint,usemask = True, rgb=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if rgb:
        return get_sparse_tracking_loss(config, image_ab, depth, opacity, viewpoint,usemask = usemask)
    return get_loss_tracking_rgbd_es(config, image_ab, depth, opacity, viewpoint,usemask = usemask)


def get_loss_tracking_rgb(config, image, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    
    
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()
def get_loss_normal(depth_mean, viewpoint):
    prior_normal = np2torch(viewpoint.normal,device="cuda")
    prior_normal = prior_normal.reshape(3, *depth_mean.shape[-2:]).permute(1,2,0)
    prior_normal_normalized = torch.nn.functional.normalize(prior_normal, dim=-1)

    normal_mean, _ = depth_to_normal(viewpoint, depth_mean, world_frame=False)
    tv_loss_fn = TVLoss()
    tv_loss = tv_loss_fn(normal_mean.unsqueeze(0)) 
    tv_weight = 0.1  # TVLoss 权重
    normal_weight = 1.0  # Normal Error 权重
    normal_error = 1 - (prior_normal_normalized * normal_mean).sum(dim=-1)
    normal_error[prior_normal.norm(dim=-1) < 0.2] = 0
    combined_loss = normal_weight * normal_error + tv_weight * tv_loss

    return normal_error.mean()

def get_sparse_tracking_loss(config, image, depth, opacity, viewpoint,usemask = True):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    gt_depth = torch.from_numpy(viewpoint.es_depth).to(
        dtype=torch.float32, device=gt_image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    max_depth_value = gt_depth.max()  # 获取 gt_depth 的最大值
    sky_mask = (gt_depth < max_depth_value).view(*depth.shape)  # sky_mask 基于最大值

    if usemask:
        rgb_region = torch.from_numpy(viewpoint.mask).cuda()
        rgb_pixel_mask = rgb_pixel_mask*rgb_region
        l1 = opacity * torch.abs(sky_mask*image * rgb_pixel_mask -sky_mask* gt_image * rgb_pixel_mask)
        return l1.mean()
    else:
        l1 = opacity * torch.abs(sky_mask*image * rgb_pixel_mask*depth_pixel_mask -sky_mask*depth_pixel_mask* gt_image * rgb_pixel_mask)
        return l1.mean()          


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    #focus on near loss
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    l1_depth = depth_normalized.detech()*l1_depth
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_tracking_rgbd_es(
    config, image, depth, opacity, viewpoint, usemask = True,initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.es_depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_sparse_tracking_loss(config, image, depth, opacity, viewpoint,usemask=usemask)
    depth_mask = depth_pixel_mask * opacity_mask
    max_depth_value = gt_depth.max()  # 获取 gt_depth 的最大值
    if max_depth_value>100:
        sky_mask = (gt_depth < max_depth_value).view(*depth.shape)  # sky_mask 基于最大值
    else:
        sky_mask = torch.ones_like(gt_depth, dtype=bool).cuda()    

    if usemask:
        rgb_region = torch.from_numpy(viewpoint.mask).cuda()
        l1_depth = torch.abs(depth * rgb_region*sky_mask - sky_mask*gt_depth * rgb_region)
        return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()

    
    l1_depth = torch.abs(depth * depth_mask*sky_mask - sky_mask*gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)

def get_loss_mapping_es(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    return get_loss_mapping_rgbd_es(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()




def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def get_loss_mapping_rgbd_es(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.es_depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()

def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()


def calculate_quaternion_difference(rotation_matrix,rotation_matrix_gt):
    if not isinstance(rotation_matrix, np.ndarray):  
        rotation_matrix = rotation_matrix.detach().cpu().numpy() 

    if not isinstance(rotation_matrix_gt, np.ndarray):  
        rotation_matrix_gt = rotation_matrix_gt.detach().cpu().numpy() 


    quat = R.from_matrix(rotation_matrix).as_quat()    
    quat_gt = R.from_matrix(rotation_matrix_gt).as_quat()
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    quat_gt = quat_gt / np.linalg.norm(quat_gt, axis=-1, keepdims=True)
    # Calculate the dot product between quaternions
    dot_product = np.einsum("...i,...i->...", quat, quat_gt)  # (...,)

    # Clamp the dot product to the valid range for acos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    # Calculate the angular difference (in radians)
    angular_difference_rad = 2 * np.arccos(np.abs(dot_product))  # Use abs to account for quaternion symmetry

    # Convert the angular difference to degrees
    angular_difference_deg = np.degrees(angular_difference_rad)

    # Convert back to PyTorch tensor
    return angular_difference_deg

def calculate_translation_difference(translation, translation_gt):
    """
    Calculates the Euclidean distance between two translation vectors.

    Args:
        translation (torch.Tensor): The estimated translation vector, shape (..., 3).
        translation_gt (torch.Tensor): The ground-truth translation vector, shape (..., 3).

    Returns:
        torch.Tensor: The Euclidean distance between the translations, shape (...).
    """
    # Ensure input tensors have the correct shape
    if len(translation.shape) == 2:
        translation = translation[0]



    if not isinstance(translation, np.ndarray):  
        translation = translation.detach().cpu().numpy() 

    if not isinstance(translation_gt, np.ndarray):  
        translation_gt = translation_gt.detach().cpu().numpy() 


    # Compute the difference between translations
    diff = translation - translation_gt

    # Compute the Euclidean norm (L2 distance)
    distance = np.linalg.norm(diff, axis=-1)

    return distance


def depths_to_points(view, depthmap, world_frame):
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor([[fx, 0., W/2.], [0., fy, H/2.], [0., 0., 1.0]]).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5, torch.arange(H, device='cuda').float() + 0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    if world_frame:
        c2w = (view.world_view_transform.T).inverse()
        rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
        rays_o = c2w[:3,3]
        points = depthmap.reshape(-1, 1) * rays_d + rays_o
    else:
        rays_d = points @ intrins.inverse().T
        points = depthmap.reshape(-1, 1) * rays_d
    return points


def depth_to_normal(view, depth, world_frame=False):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth, world_frame).reshape(*depth.shape[1:], 3)
    normal_map = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map[1:-1, 1:-1, :] = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    return normal_map, points



class EdgeAwareTV(nn.Module):
    """Edge Aware Smooth Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, depth: Tensor, rgb: Tensor):
        """
        Args:
            depth: [batch, H, W, 1]
            rgb: [batch, H, W, 3]
        """
        grad_depth_x = torch.abs(depth[..., :, :-1, :] - depth[..., :, 1:, :])
        grad_depth_y = torch.abs(depth[..., :-1, :, :] - depth[..., 1:, :, :])

        grad_img_x = torch.mean(
            torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
        )

        grad_depth_x *= torch.exp(-grad_img_x)
        grad_depth_y *= torch.exp(-grad_img_y)

        return grad_depth_x.mean() + grad_depth_y.mean()
# def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2, scale=0.25, num_matches=100):
#     import torch
#     import random
#     import cv2
#     import numpy as np

#     # 处理 img1
#     if isinstance(img1, torch.Tensor):
#         img1 = img1[0].permute(1, 2, 0).cpu().numpy()
#     else:
#         img1 = img1[0].transpose(1, 2, 0)
#     img1 = ((img1 * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

#     # 处理 img2
#     if isinstance(img2, torch.Tensor):
#         img2 = img2[0].permute(1, 2, 0).cpu().numpy()
#     else:
#         img2 = img2[0].transpose(1, 2, 0)
#     img2 = ((img2 * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

#     # 确保3通道
#     if img1.ndim == 2:
#         img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#     if img2.ndim == 2:
#         img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

#     # 创建黑底文字条幅
#     banner_height = 60
#     banner1 = np.zeros((banner_height, img1.shape[1], 3), dtype=np.uint8)
#     banner2 = np.zeros((banner_height, img2.shape[1], 3), dtype=np.uint8)

#     # 在条幅上写字
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1.2
#     thickness = 2
#     text_color = (255, 255, 255)  # 白色字
#     cv2.putText(banner1, "Keyframe", (20, 45), font, font_scale, text_color, thickness, cv2.LINE_AA)
#     cv2.putText(banner2, "Current Frame", (20, 45), font, font_scale, text_color, thickness, cv2.LINE_AA)

#     # 拼接：上面是banner1+img1，下面是img2+banner2
#     top_part = np.vstack((banner1, img1))
#     bottom_part = np.vstack((img2, banner2))
#     stacked_img = np.vstack((top_part, bottom_part))
#     stacked_img = np.vstack((img1, img2))

#     stacked_img = np.ascontiguousarray(stacked_img)

#     # 拿到尺寸
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]
#     h_total, w_total = stacked_img.shape[:2]

#     # 画蓝色框：框住 top_part（Keyframe部分）
#     color_blue = (255, 0, 0)
#     start_top = (0, 0)
#     end_top = (w1 - 1, banner_height + h1 - 1)
#     cv2.rectangle(stacked_img, start_top, end_top, color_blue, thickness=4)

#     # 画红色框：框住 bottom_part（Current Frame部分）
#     color_red = (0, 0, 255)
#     start_bottom = (0, banner_height + h1)
#     end_bottom = (w2 - 1, banner_height + h1 + h2 + banner_height - 1)
#     cv2.rectangle(stacked_img, start_bottom, end_bottom, color_red, thickness=4)

#     # 开始画连线
#     matches_to_draw = min(len(ref_points), num_matches)
#     selected_indices = random.sample(range(len(ref_points)), matches_to_draw)

#     offset_y_img1 = banner_height
#     offset_y_img2 = banner_height + h1

#     for idx in selected_indices:
#         p1 = ref_points[idx]
#         p2 = dst_points[idx]

#         pt1 = (int(p1[0]), int(p1[1]) + offset_y_img1)
#         pt2 = (int(p2[0]), int(p2[1]) + offset_y_img2)

#         cv2.circle(stacked_img, pt1, radius=4, color=(0, 255, 0), thickness=-1)
#         cv2.circle(stacked_img, pt2, radius=4, color=(0, 0, 255), thickness=-1)
#         cv2.line(stacked_img, pt1, pt2, (0, 255, 0), 2, lineType=cv2.LINE_AA)

#     # 🔥最后统一整体缩小
#     target_size = (int(w_total * scale), int(h_total * scale))
#     stacked_img = cv2.resize(stacked_img, target_size)

#     return stacked_img


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2, scale=0.25, num_matches=100):
    import torch
    import random
    import cv2
    import numpy as np
    # 处理 img1
    if isinstance(img1, torch.Tensor):
        img1 = img1.permute(1, 2, 0).cpu().numpy()
    else:
        img1 = img1.transpose(1, 2, 0)
    img1 = ((img1 ) * 255).clip(0, 255).astype(np.uint8)

    # 处理 img2
    if isinstance(img2, torch.Tensor):
        img2 = img2.permute(1, 2, 0).cpu().numpy()
    else:
        img2 = img2.transpose(1, 2, 0)
    img2 = ((img2 ) * 255).clip(0, 255).astype(np.uint8)

    # # 确保3通道
    # if img1.ndim == 2:
    #     img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    # if img2.ndim == 2:
    #     img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    # 去掉拼接部分，只保留上下的连接
    stacked_img = np.vstack((img1, img2))
    stacked_img = np.ascontiguousarray(stacked_img)

    # 拿到尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 开始画连线
    matches_to_draw = min(len(ref_points), num_matches)
    selected_indices = random.sample(range(len(ref_points)), matches_to_draw)
    h1 = img1.shape[0]
    cv2.line(stacked_img, (0, h1), (stacked_img.shape[1], h1), (255, 255, 255), 2)
    # 计算 offset
    offset_y_img1 = 0
    offset_y_img2 = h1

    for idx in selected_indices:
        p1 = ref_points[idx]
        p2 = dst_points[idx]

        pt1 = (int(p1[0]), int(p1[1]) + offset_y_img1)
        pt2 = (int(p2[0]), int(p2[1]) + offset_y_img2)

        cv2.circle(stacked_img, pt1, radius=5, color=(0, 0, 255), thickness=-1)  # 蓝色大圆
        cv2.circle(stacked_img, pt2, radius=5, color=(255, 0, 0), thickness=-1)  # 红色大圆
        cv2.line(stacked_img, pt1, pt2, (0, 255, 0, 128), 2, lineType=cv2.LINE_AA)

    # 🔥最后统一整体缩小
    target_size = (int(w1 * scale), int((h1 + h2) * scale))
    stacked_img = cv2.resize(stacked_img, target_size)

    return stacked_img
def unpad_tensor(padded_tensor, device=None):
    """
    从填充后的张量中剔除掉全为 0 的行，还原出原始的 (N, M) 张量。

    参数:
        padded_tensor (torch.Tensor): 填充后的张量，形状为 (max_len, M)
        device (str or torch.device, optional): 输出张量的目标设备，默认为 None 表示与输入一致

    返回:
        torch.Tensor: 去除填充后的张量，形状为 (N, M)，其中 N <= max_len
    """
    # 按行判断是否全为 0
    mask = ~(padded_tensor == 0).all(dim=1)
    result = padded_tensor[mask]

    if device is not None:
        result = result.to(device)

    return result
def get_top_k_frame_ids(
    data: List[Tuple[int, float]],
    topk: int = 2
) -> List[int]:
    """
    从输入的 (frame_id, score) 数据中，去重保留每个 frame_id 的最高得分，
    然后按得分排序并返回前 topk 个 frame_id。

    参数:
        data (List[Tuple[int, float]]): 输入数据，每项为 (frame_id, score)
        topk (int): 返回得分最高的前 topk 项

    返回:
        List[int]: 得分最高的前 topk 个 frame_id
    """
    best_scores = {}
    for frame_id, score in data:
        if frame_id not in best_scores or score > best_scores[frame_id]:
            best_scores[frame_id] = score

    sorted_best = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    return [frame_id for frame_id, _ in sorted_best[:topk]]

def dedup_and_difference_clustering(data, threshold=5, max_clusters=2, max_per_cluster=3):
    """
    根据 index 差值聚类，每组最多保留3个高分 index，返回最多2组。
    """
    """
    去重后基于 index 差值聚类的函数：
    1. 对每个 index 保留最高 score
    2. 根据 threshold 聚类
    3. 每类最多保留 max_per_cluster 个高分 index
    4. 最多保留 max_clusters 类

    参数:
    - data: List of (index, score)
    - threshold: index 差值最大阈值
    - max_clusters: 最多保留的聚类数量
    - max_per_cluster: 每类最多保留的 index 数

    返回:
    - List[List[int]]
    """
    if not data:
        return []

    # Step 1: 去重 - 保留每个 index 的最大 score
    index_to_best_score = {}
    for idx, score in data:
        if idx not in index_to_best_score or score > index_to_best_score[idx]:
            index_to_best_score[idx] = score
    dedup_data = list(index_to_best_score.items())

    # Step 2: 按 index 升序排序
    sorted_data = sorted(dedup_data, key=lambda x: x[0])

    # Step 3: 聚类
    clusters = []
    current_cluster = [sorted_data[0]]
    for i in range(1, len(sorted_data)):
        if sorted_data[i][0] - sorted_data[i - 1][0] <= threshold:
            current_cluster.append(sorted_data[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_data[i]]
    clusters.append(current_cluster)

    # Step 4: 保留最多 max_clusters 个（按长度排序）
    if len(clusters) > max_clusters:
        clusters = sorted(clusters, key=lambda c: len(c), reverse=True)[:max_clusters]

    # Step 5: 每类保留最多 max_per_cluster 个高分 index
    final_clusters = []
    for group in clusters:
        top_items = sorted(group, key=lambda x: x[1], reverse=True)[:max_per_cluster]
        indices = [x[0] for x in top_items]
        final_clusters.append(indices)

    return final_clusters