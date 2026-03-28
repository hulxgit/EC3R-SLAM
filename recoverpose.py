import cv2
import numpy as np
import torch
import pypose as pp
from ec3r.utils.slam_utils  import torch2np
def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')



def recover_pose(matcheses,c2w_idx,kf_idx,pointmaps,K,poinkey,c2ws,overlap):
    #revocer kf pose
    points3d = pointmaps[poinkey][overlap-1:].copy()
    confs = pointmaps["local_confidence_maps"][overlap-1:]
    H,W,_ = points3d[0].shape
    kfpose = []

    for conf,pts3d in zip(confs,points3d):
        
        pixels = np.mgrid[:W, :H].T.astype(np.float32)
        conf = conf.reshape(-1,1)
        pixels = pixels.reshape(-1,2)
        pts3d = pts3d.reshape(-1,3)
        N = H*W
        num_select = N // 3

        top_indices = np.argsort(-conf.ravel())[:num_select]
        mask = np.zeros(N, dtype=bool)
        mask[top_indices] = True

        success, R, T, inliers = cv2.solvePnPRansac(pts3d[mask], pixels[mask], K, None,
                             iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
        assert success

        R = cv2.Rodrigues(R)[0]  # world to cam
        kfpose.append(inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])) # cam to world


    #  keyframe
    for i, idx in enumerate(kf_idx[overlap-1:]):
        if i==0:
            continue
        c2ws[idx] = kfpose[i]
        print(idx)
    for i, matches in enumerate(matcheses):
        pcl3d = points3d[i]
        for match in matches:
            
            kframeidx = match["index"][0]
            curidx = match["index"][1]
            if curidx not in kf_idx:
                kf_point = match["points"][0]
                cur_point = match["points"][1]
                xs = kf_point[:, 0].astype(int)
                ys = kf_point[:, 1].astype(int)
                pointmap = pcl3d[ys, xs] 
                success, R, T, inliers = cv2.solvePnPRansac(pointmap, cur_point, K, None,
                             iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
                assert success

                R = cv2.Rodrigues(R)[0]  # world to cam
                T_curr = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world
                c2ws[curidx] = T_curr
   
    return c2ws


def recover_pose_init(initlist,pointmaps,K,poinkey):
    c2ws = len(initlist)*[np.ones(4)]
    points3d = pointmaps[poinkey].copy()
    confs = pointmaps["local_confidence_maps"]
    H,W,_ = points3d[0].shape
    kfpose = []

    for conf,pts3d in zip(confs,points3d):
        
        pixels = np.mgrid[:W, :H].T.astype(np.float32)
        conf = conf.reshape(-1,1)
        pixels = pixels.reshape(-1,2)
        pts3d = pts3d.reshape(-1,3)
        N = H*W
        num_select = N // 3

        top_indices = np.argsort(-conf.ravel())[:num_select]
        mask = np.zeros(N, dtype=bool)
        mask[top_indices] = True

        success, R, T, inliers = cv2.solvePnPRansac(pts3d[mask], pixels[mask], K, None,
                             iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
        assert success

        R = cv2.Rodrigues(R)[0]  # world to cam
        kfpose.append(inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])) # cam to world
    for i, idx in enumerate(initlist):
        c2ws[idx] = kfpose[i]


    point1 = np.array(points3d)
    point1 = point1.reshape(-1,3)
    color1 = np.array(pointmaps["rgb"])
    color1 = color1.reshape(-1, 3)
    confs = np.array(confs).reshape(-1)
    conf_thr = np.percentile(confs, 20)
    mask = confs > conf_thr
    points = point1[mask]
    colors = color1[mask]
    return c2ws,(points,colors)



def reestimate_pose(pointmaps,K,poinkey,overlap):
    #revocer kf pose
    points3d = pointmaps[poinkey][overlap-1:].copy()
    confs = pointmaps["local_confidence_maps"][overlap-1:]
    H,W,_ = points3d[0].shape
    kfpose = []

    for conf,pts3d in zip(confs,points3d):
        
        pixels = np.mgrid[:W, :H].T.astype(np.float32)
        conf = conf.reshape(-1,1)
        pixels = pixels.reshape(-1,2)
        pts3d = pts3d.reshape(-1,3)
        N = H*W
        num_select = N // 3

        top_indices = np.argsort(-conf.ravel())[:num_select]
        mask = np.zeros(N, dtype=bool)
        mask[top_indices] = True

        success, R, T, inliers = cv2.solvePnPRansac(pts3d[mask], pixels[mask], K, None,
                             iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
        assert success

        R = cv2.Rodrigues(R)[0]  # world to cam
        kfpose.append(inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])) # cam to world
    

    # #  keyframe
    # for i, idx in enumerate(kf_idx[overlap-1:]):
    #     if i==0:
    #         continue
    #     c2ws[idx] = kfpose[i]
    # for i, matches in enumerate(matcheses):
    #     pcl3d = points3d[i]
    #     for match in matches:
            
    #         kframeidx = match["index"][0]
    #         curidx = match["index"][1]
    #         if curidx not in kf_idx:
    #             kf_point = match["points"][0]
    #             cur_point = match["points"][1]
    #             xs = kf_point[:, 0].astype(int)
    #             ys = kf_point[:, 1].astype(int)
    #             pointmap = pcl3d[ys, xs] 
    #             success, R, T, inliers = cv2.solvePnPRansac(pointmap, cur_point, K, None,
    #                          iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
    #             assert success

    #             R = cv2.Rodrigues(R)[0]  # world to cam
    #             T_curr = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world
    #             c2ws[curidx] = T_curr
   
    return kfpose


def estimation_pose_from_point(new_point,new_conf,K):
    #后面统统改成在torch上面弄！！！
    new_point = [torch2np(p) for p in new_point]
    H,W,_= new_point[0].shape
    kfpose = []

    for conf,pts3d in zip(new_conf,new_point):
        
        pixels = np.mgrid[:W, :H].T.astype(np.float32)
        conf = np.array(conf).reshape(-1,1)
        pixels = pixels.reshape(-1,2)
        pts3d = pts3d.reshape(-1,3)
        N = H*W
        num_select = N // 3

        top_indices = np.argsort(-conf.ravel())[:num_select]
        mask = np.zeros(N, dtype=bool)
        mask[top_indices] = True

        success, R, T, inliers = cv2.solvePnPRansac(pts3d[mask], pixels[mask], K, None,
                             iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)

        R = cv2.Rodrigues(R)[0]  # world to cam
        kfpose.append(inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])) # cam to world
    return kfpose



def estimation_pose_from_point_torch(new_point, new_conf, K_np):
    """
    使用 PyPose 的 EPnP 从稠密点云和置信度图估计相机姿态。

    参数:
        new_point: list[torch.Tensor]，每张图的 (H, W, 3) 三维点坐标
        new_conf:  list[torch.Tensor]，每张图的 (H, W) confidence map
        K_np: numpy.ndarray，相机内参矩阵 (3, 3)

    返回:
        kfpose: list[torch.Tensor]，每张图估计得到的 SE(3) 4x4 pose 矩阵
    """
    K = torch.tensor(K_np, dtype=torch.float32)
    kfpose = []

    for conf_map, pts3d in zip(new_conf, new_point):
        H, W, _ = pts3d.shape
        N = H * W
        num_select = N // 3

        conf = conf_map.reshape(-1)
        pts3d = pts3d.reshape(-1, 3)
        top_indices = torch.topk(conf, num_select).indices
        pts3d_selected = pts3d[top_indices]

        # 生成 pixel 坐标 (x, y)
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        pixels = torch.stack((xx, yy), dim=-1).reshape(-1, 2).float()
        pixels_selected = pixels[top_indices]

        # 使用 PyPose EPnP 模块估计 pose（返回的是 cam->world）
        epnp = pp.module.EPnP(K)
        pose = epnp(pts3d_selected, pixels_selected)

        # 转换为 4x4 矩阵
        kfpose.append(pose.SE3().matrix())

    return kfpose