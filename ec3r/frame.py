import numpy as np
from .config import config
from ec3r.utils.fast3r_utils import resize_img

from collections import defaultdict
import torch
from ec3r.utils.slam_utils import torch2np,np2torch,unpad_tensor
from typing import List


def expand_and_include_extras(values: List[int], tol: int = 2) -> List[int]:
    """
    1. 找到 values 的最小值 vmin；
    2. 生成闭区间 [vmin - tol, vmin + tol] 中的整数；
    3. 同时包含原列表中所有超出这个区间的值；
    4. 最后去重并排序输出。
    """
    if not values:
        return []

    vmin = min(values)
    low, high = vmin - tol, vmin + tol

    # 区间中的所有整数
    interval = list(range(low, high + 1))

    # 原始值中超出区间的部分
    extras = [v for v in values if v < low or v > high]

    # 合并，去重，然后排序
    result = sorted(set(interval + extras))
    return result


def filter_kf_clusters_exclusive(
    kf_cluster: List[List[int]],
    kframeidx: List[int],
    tol: int = 20
) -> List[List[int]]:
    """
    过滤 kf_cluster 中的子列表，仅保留那些子列表中所有索引都 *不* 落在
    kframeidx 中任一值的 ±tol 范围内。

    参数:
        kf_cluster: List[List[int]] -- 原始的关键帧聚类列表
        kframeidx: List[int]       -- 基准的关键帧索引列表
        tol: int                   -- 容差范围，默认为 10

    返回:
        List[List[int]] -- 满足“元素均不靠近 kframeidx ±tol”条件的子列表
    """
    valid_clusters = []
    for cluster in kf_cluster:
        # 如果 cluster 中所有元素都满足“不靠近任何 kframeidx ±tol”
        if all(
            not any(abs(elem - idx) <= tol for idx in kframeidx)
            for elem in cluster
        ):
            valid_clusters.append(cluster)
    return valid_clusters
def transform_points_to_global(point, kf_pose, pointpose):
    """
    将相机局部坐标下的点云转换到全局坐标系，并计算全局相机位姿。

    参数:
        point (Tensor): [B, ..., 3] 或 [N, 3]，相机坐标系下的 3D 点。
        kf_pose (Tensor): [4, 4]，关键帧在当前窗口下的位姿。
        pointpose (Tensor): [4, 4]，当前窗口相对于全局坐标系的变换矩阵。

    返回:
        point_global (Tensor): [B, ..., 3]，变换后的点云（全局坐标系）。
        kf_pose_global (Tensor): [4, 4]，关键帧的全局位姿。
    """
    # Step 1: 相机 pose 从局部 → 全局
    kf_pose_global = pointpose @ kf_pose  # (4, 4)

    # Step 2: 点坐标从相机 → 全局
    original_shape = point.shape
    point3d = point.reshape(-1, 3)
    N = point3d.shape[0]

    # 齐次坐标扩展
    point_h = torch.cat([point3d, torch.ones(N, 1, device=point.device)], dim=1)  # (N, 4)
    point_global_h = (pointpose @ point_h.T).T  # (N, 4)
    point_global = point_global_h[:, :3]  # (N, 3)
    point_global = point_global.reshape(original_shape)

    return point_global, kf_pose_global
class SharedKeyframes:
    def __init__(self, manager, h,w, buffer=94, dtype=torch.float32, device="cpu"):
        self.lock = manager.RLock()
        self.n_size = manager.Value("i", 0)
        self.K = torch.zeros(3, 3, device=device, dtype=dtype).share_memory_()

        self.h, self.w = h, w
        self.buffer = buffer
        self.dtype = dtype
        self.device = device

        self.feat_dim = 1024
        self.num_patches = h * w // (14 * 14)

        ### state attributes ###
        self.dataset_idx = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.color = torch.zeros(buffer, h, w, 3,device=device, dtype=dtype).share_memory_()
        self.point = torch.zeros(buffer, h, w, 3,device=device, dtype=dtype).share_memory_()
        self.conf = torch.zeros(buffer, 1, h, w, device=device, dtype=dtype).share_memory_()
        self.c2ws = torch.zeros(buffer, 4,4, device=device, dtype=dtype).share_memory_()
        self.intrinsics = torch.zeros(buffer, 3,3, device=device, dtype=dtype).share_memory_()

        self.pointpose = torch.zeros(buffer, 4,4, device=device, dtype=dtype).share_memory_()
        self.xfeat_kpts = torch.zeros(buffer, 4096,2, device=device, dtype=dtype).share_memory_()
        self.xfeat_desc = torch.zeros(buffer, 4096,64, device=device, dtype=dtype).share_memory_()


        ### feature attributes ###
        self.feat = torch.zeros(buffer, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.dirty = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()

        ### other attributes ###
        self.ismapframe = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.graph_idx = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.is_empty = torch.ones(buffer, dtype=torch.bool).share_memory_()

        # fmt: on

    def update_K(self, K):
        # 先转换成 Torch Tensor

        self.K.copy_(K)
    def __setitem__(self, idx, items):
        with self.lock:
            if isinstance(items, (list, tuple)):
                # 批量写入
                batch_size = len(items[0])

                # 查找空位
                empty_slots = torch.where(self.is_empty)[0]

                if len(empty_slots) >= batch_size:
                    real_idxs = empty_slots[:batch_size].tolist()
                else:
                    additional_needed = batch_size - len(empty_slots)
                    real_idxs = empty_slots.tolist() + [(idx + i) % self.buffer for i in range(additional_needed)]

                # 批量插入数据
                if len(items)>11:
                    for i, real_idx in enumerate(real_idxs):
                        self.dataset_idx[real_idx] = items[0][i]
                        self.color[real_idx]       = items[1][i]
                        self.point[real_idx]       = items[2][i]
                        self.conf[real_idx]        = items[3][i]
                        self.c2ws[real_idx]        = items[4][i]
                        self.intrinsics[real_idx]  = items[5][i]

                        self.pointpose[real_idx]   = items[6][i]       # 新增的字段
                        self.feat[real_idx]        = items[7][i]
                        self.dirty[real_idx]       = items[8][i]
                        self.ismapframe[real_idx]  = items[9][i]
                        self.graph_idx[real_idx]   = items[10][i]
                        self.xfeat_kpts[real_idx]   = items[11][i]
                        self.xfeat_desc[real_idx]   = items[12][i]

                        self.is_empty[real_idx]    = False
                else:
                    for i, real_idx in enumerate(real_idxs):
                        self.dataset_idx[real_idx] = items[0][i]
                        self.color[real_idx]       = items[1][i]
                        self.point[real_idx]       = items[2][i]
                        self.conf[real_idx]        = items[3][i]
                        self.c2ws[real_idx]        = items[4][i]
                        self.intrinsics[real_idx]  = items[5][i]

                        self.pointpose[real_idx]   = items[6][i]       # 新增的字段
                        self.feat[real_idx]        = items[7][i]
                        self.dirty[real_idx]       = items[8][i]
                        self.ismapframe[real_idx]  = items[9][i]
                        self.graph_idx[real_idx]   = items[10][i]

                        self.is_empty[real_idx]    = False                    
                return real_idxs
    def __len__(self):
        with self.lock:
            return self.n_size.value

    def get_unique_dataset_idx(self):
        with self.lock:
            valid_mask = ~self.is_empty
            valid_dataset_idx = self.dataset_idx[valid_mask]
            unique_indices = torch.unique(valid_dataset_idx).tolist()
        return unique_indices
    def append(self, value):
        with self.lock:
            # 📝 如果是单个元素（不是 list 或 tuple），先包装成 list 处理
            if not isinstance(value, (list, tuple)):
                value = [value]

            # 📝 判断是否是批量 keyframe_idx
            if isinstance(value[0], (list, tuple)):
                # 如果是批量 keyframe_idx 的情况
                real_idxs = self[self.n_size.value] = value
                # 维护 n_size 增长
                self.n_size.value = (self.n_size.value + len(value[0])) % self.buffer
            else:
                # 如果是单个的
                real_idx = self[self.n_size.value] = value
                self.n_size.value = (self.n_size.value + 1) % self.buffer
                real_idxs = [real_idx]

            return real_idxs

    def __delitem__(self, idx):
        with self.lock:
            real_idx = idx % self.buffer
            self.color[real_idx].zero_()
            self.point[real_idx].zero_()
            self.conf[real_idx].zero_()
            self.c2ws[real_idx].zero_()
            self.pointpose[real_idx].zero_()
            self.feat[real_idx].zero_()
            self.intrinsics[real_idx].zero_()
            self.dirty[real_idx] = False
            self.ismapframe[real_idx] = False
            self.graph_idx[real_idx].zero_()
            self.dataset_idx[real_idx].zero_()
            
            # ✅ 设置为未占用状态
            self.is_empty[real_idx] = True

    def find_overlap_idx(self,graph_idx):
        with self.lock:
            n = len(self)
            mask  = (
                (self.graph_idx[:n] == graph_idx) &
                (~self.is_empty[:n]) &
                (self.ismapframe[:n])
            )
            idxs1 = torch.where(mask)[0]
            ds_idx = self.dataset_idx[idxs1].cpu().numpy()
            return ds_idx

    def find_inthisgraph(self,graph_idx):
        with self.lock:
            n = len(self)
            mask  = (
                (self.graph_idx[:n] == graph_idx) &
                (~self.is_empty[:n])             )
            matches = torch.where(mask)[0]
            points = [self.point[idx].clone() for idx in matches]
            confs = [self.conf[idx].clone() for idx in matches]
            dataset_indices = [int(self.dataset_idx[idx].item()) for idx in matches]            
            return points,confs,dataset_indices

    def last_pointpose(self,graph_idx):
        with self.lock:
            n = len(self)
            mask  = (
                (self.graph_idx[:n] == graph_idx) &
                (~self.is_empty[:n])             )
            matches = torch.where(mask)[0]
            points = [self.point[idx].clone() for idx in matches]
            return 
        

    def init_add_cfeat(self, cur_frame_idx,feat_kpt,feat_desc):
        with self.lock:
            # 查找所有满足 dataset_idx == cur_frame_idx 的索引 n
            matches = torch.where(self.dataset_idx == cur_frame_idx)[0]

            if len(matches) == 0:
                print(f"❌ 未找到 dataset_idx == {cur_frame_idx} 的 KeyFrame")
                return None  # 或 raise Exception(...) 视具体情况

            # 如果有多个匹配，可以选择第一个，或全部处理
            n = matches[0].item()
            print(f"✅ 找到匹配的索引 n = {n} 对应 cur_frame_idx = {cur_frame_idx}")
            self.xfeat_kpts[n] = feat_kpt
            self.xfeat_desc[n] = feat_desc

            return n

    def find_by_dataset_idx(self, target_idx):
        """
        查找指定的 dataset_idx，且 ismapframe 为 True 的 KeyFrames。
        
        参数:
            target_idx (int or List[int]): 需要查找的 dataset_idx 索引
        
        返回:
            Tuple[...] + matched_flags: 满足条件的数据 + 告知 target_idx 中哪些被匹配
        """
        with self.lock:
            if isinstance(target_idx, int):
                target_idx = [target_idx]
            
            mask = torch.zeros_like(self.dataset_idx, dtype=torch.bool)
            
            for idx in target_idx:
                mask |= (self.dataset_idx == idx)

            mask &= self.ismapframe
            matches = torch.where(mask)[0]

            feats = [self.feat[idx].clone() for idx in matches]
            points = [self.point[idx].clone() for idx in matches]
            confs = [self.conf[idx].clone() for idx in matches]
            dataset_indices = [int(self.dataset_idx[idx].item()) for idx in matches]
            graph_idx = [int(self.graph_idx[idx].item()) for idx in matches]
            pointspose = [self.pointpose[idx].clone() for idx in matches]

            # 新增：指示 target_idx 中哪些被匹配
            matched_flags = [any(self.dataset_idx[m] == idx for m in matches) for idx in target_idx]

        print(f"🔍 找到 {len(feats)} 个符合条件的 KeyFrames，dataset_idx={target_idx}")
        print(f"✅ 匹配情况: {[f'✓' if flag else '✗' for flag in matched_flags]}")
        
        return feats, confs, points, graph_idx, dataset_indices, pointspose
    

    def find_matched_idx(self, target_idx):
        """
        查找指定的 dataset_idx，且 ismapframe 为 True 的 KeyFrames。
        
        参数:
            target_idx (int or List[int]): 需要查找的 dataset_idx 索引
        
        返回:
            Tuple[...] + matched_flags: 满足条件的数据 + 告知 target_idx 中哪些被匹配
        """
        with self.lock:

            
            mask = torch.zeros_like(self.dataset_idx, dtype=torch.bool)
            
            mask |= (self.dataset_idx == target_idx)

            mask &= self.ismapframe
            matches = torch.where(mask)[0]
            if len(matches) == 0:
                if isinstance(target_idx, int):
                    target_idx = [target_idx]
                # print(f"❌ 没有匹配到任何关键帧，target_idx: {target_idx}")
                return None,None,None,None,None,None,None
            idx = matches[0]  # 取第一个匹配结果

            poses = self.c2ws[idx].clone()
            points = self.point[idx].clone()
            confs = self.conf[idx].clone()
            graph_index = int(self.graph_idx[idx].item())
            pointspose = self.pointpose[idx].clone()
            xfeat_point = unpad_tensor(self.xfeat_kpts[idx].clone())
            intrinsic = self.intrinsics[idx].clone()
        return poses, confs, points, graph_index, pointspose,xfeat_point,intrinsic 
    


    def find_oldkf_idx(self, target_idx):
        """
        查找指定的 dataset_idx，且 ismapframe 为 True 的 KeyFrames。
        
        参数:
            target_idx (int or List[int]): 需要查找的 dataset_idx 索引
        
        返回:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]: 满足条件的 (feat, pos, dataset_idx) 的列表
        """
        with self.lock:

            # 📝 构建返回结果，改成你想要的格式
            feats = [self.feat[idx].clone() for idx in target_idx]
            points = [self.point[idx].clone() for idx in target_idx]
            pointpose = [self.pointpose[idx].clone() for idx in target_idx]
            color =  [self.color[idx].clone() for idx in target_idx]
            confs = [self.conf[idx].clone() for idx in target_idx]


            dataset_indices = [int(self.dataset_idx[idx].item()) for idx in target_idx]
            graph_idx = [int(self.graph_idx[idx].item()) for idx in target_idx]
            xfeat_desc =  [self.xfeat_desc[idx].clone() for idx in target_idx]
            xfeat_kpt = [self.xfeat_kpts[idx].clone() for idx in target_idx]
        print(f"🔍 找到 {len(feats)} 个符合条件的 KeyFrames，dataset_idx={target_idx}")
        return feats,confs, points,color,graph_idx,dataset_indices,pointpose,xfeat_desc,xfeat_kpt

    def find_features(self, target_idx):
        """
        查找指定的 dataset_idx，且 ismapframe 为 True 的 KeyFrames。
        
        参数:
            target_idx (int or List[int]): 需要查找的 dataset_idx 索引
        
        返回:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]: 满足条件的 (feat, pos, dataset_idx) 的列表
        """
        with self.lock:

            # 📝 构建返回结果，改成你想要的格式
            xfeat_desc = unpad_tensor(self.xfeat_desc[target_idx])
            xfeat_kpt = unpad_tensor(self.xfeat_kpts[target_idx])


        return xfeat_desc,xfeat_kpt


    def __getitem__(self, idx):
        """
        支持单个索引或批量索引来获取 KeyFrame 的所有信息。
        
        参数:
            idx (int or List[int]): 要查询的索引，支持单个索引和批量索引
        
        返回:
            如果是单个索引，返回一个字典:
            {
                "dataset_idx": dataset_idx,
                "color": color,
                "point": point,
                "conf": conf,
                "c2ws": c2ws,
                "feat": feat,
                "pos": pos,
                "dirty": dirty,
                "ismapframe": ismapframe,
                "graph_idx": graph_idx
            }

            如果是批量索引，返回一个列表，每个元素是上面格式的字典
        """
        with self.lock:
            if isinstance(idx, int):
                # ✅ 单个索引查询
                real_idx = idx % self.buffer
                return {
                    "dataset_idx": self.dataset_idx[real_idx].clone(),
                    "color": self.color[real_idx].clone(),
                    "point": self.point[real_idx].clone(),
                    "pointpose": self.pointpose[real_idx].clone(),

                    "conf": self.conf[real_idx].clone(),
                    "c2ws": self.c2ws[real_idx].clone(),
                    "feat": self.feat[real_idx].clone(),
                    "intrinsics": self.intrinsics[real_idx].clone(),
                    "dirty": self.dirty[real_idx].clone(),
                    "ismapframe": self.ismapframe[real_idx].clone(),
                    "graph_idx": self.graph_idx[real_idx].clone()
                }
            elif isinstance(idx, (list, tuple)):
                # ✅ 批量索引查询
                result = []
                for i in idx:
                    real_idx = i % self.buffer
                    result.append({
                        "dataset_idx": self.dataset_idx[real_idx].clone(),
                        "color": self.color[real_idx].clone(),
                        "point": self.point[real_idx].clone(),
                        "pointpose": self.pointpose[real_idx].clone(),
                        "conf": self.conf[real_idx].clone(),
                        "c2ws": self.c2ws[real_idx].clone(),
                        "feat": self.feat[real_idx].clone(),
                        "intrinsics": self.intrinsics[real_idx].clone(),
                        "dirty": self.dirty[real_idx].clone(),
                        "ismapframe": self.ismapframe[real_idx].clone(),
                        "graph_idx": self.graph_idx[real_idx].clone()
                    })
                return result
            else:
                raise TypeError(f"索引必须是 `int` 或 `list`，但你给的是 `{type(idx)}`")
    
    
    
    def get_latest_keyframe(self):
        """
        获取最新添加的一个 KeyFrame 的 `point`, `c2ws`, 和 `dataset_idx`。
        
        返回:
            - point (torch.Tensor): 最新添加的 KeyFrame 的 3D 点云数据
            - kf_pose (torch.Tensor): 最新添加的 KeyFrame 的位姿矩阵
            - frame_idx (int): 最新添加的 KeyFrame 对应的 `dataset_idx`
        """
        with self.lock:
            # 获取最新的索引
            if self.n_size.value == 0:
                real_idx = self.buffer - 1
            else:
                real_idx = (self.n_size.value - 1) % self.buffer
            
            # 提取对应的数据
            point = self.point[real_idx].clone()
            kf_pose = self.c2ws[real_idx].clone()
            pointpose = self.pointpose[real_idx].clone()
            frame_idx = int(self.dataset_idx[real_idx].item())

     

            point_global, kf_pose_global = transform_points_to_global(point, kf_pose, pointpose)

        return point_global, kf_pose_global, frame_idx

    def get_feat(self,n):
        """
        获取最新添加的一个 KeyFrame 的 `point`, `c2ws`, 和 `dataset_idx`。
        
        返回:
            - point (torch.Tensor): 最新添加的 KeyFrame 的 3D 点云数据
            - kf_pose (torch.Tensor): 最新添加的 KeyFrame 的位姿矩阵
            - frame_idx (int): 最新添加的 KeyFrame 对应的 `dataset_idx`
        """
        with self.lock:
            # 获取最新的索引
            if self.n_size.value == 0:
                real_idx = self.buffer - 1
            else:
                real_idx = (self.n_size.value - 1) % self.buffer
            
            # 提取对应的数据
            feat = self.feat[real_idx-n].clone()
        return feat
    
    def get_posepoint(self, real_idx=None):
        """ismapframe
        获取指定 index（或默认最后一个）的点云 + 相机位姿
        """
        with self.lock:
            # ✅ 如果没有传 index，则默认取最后一个
            if real_idx is None:
                if self.n_size.value == 0:
                    real_idx = self.buffer - 1
                else:
                    real_idx = (self.n_size.value - 1) % self.buffer

            # 提取数据
            kf_pose = self.c2ws[real_idx].clone()
            pointpose = self.pointpose[real_idx].clone()
            point = self.point[real_idx].clone()

            point_global, kf_pose_global = transform_points_to_global(point, kf_pose, pointpose)

        return point_global, kf_pose_global

    def search_graph_idx(self,real_idx):
        """
        获取最新添加的一个 KeyFrame 的 `point`, `c2ws`, 和 `dataset_idx`。
        
        返回:
            - point (torch.Tensor): 最新添加的 KeyFrame 的 3D 点云数据
            - kf_pose (torch.Tensor): 最新添加的 KeyFrame 的位姿矩阵
            - frame_idx (int): 最新添加的 KeyFrame 对应的 `dataset_idx`
        """
        with self.lock:
            # 获取最新的索引
            graphidx = self.graph_idx[real_idx]

        return graphidx
    
    def get_common_pointconf_between_graphs(self, gidx1=1, gidx2=2):
        with self.lock:
            n = len(self)

            # 加入 ismapframe 条件
            mask1 = (
                (self.graph_idx[:n] == gidx1) &
                (~self.is_empty[:n]) &
                (self.ismapframe[:n])
            )
            mask2 = (
                (self.graph_idx[:n] == gidx2) &
                (~self.is_empty[:n]) 
            )

            idxs1 = torch.where(mask1)[0]
            idxs2 = torch.where(mask2)[0]

            ds_idx1 = self.dataset_idx[idxs1].cpu().numpy()
            ds_idx2 = self.dataset_idx[idxs2].cpu().numpy()



            # 找交集
            common_vals, i1, i2 = np.intersect1d(ds_idx1, ds_idx2, return_indices=True)
            print("common_vals,i1,i2",common_vals,i1,i2)
            if len(common_vals) == 0:
                print(f"⚠️ No common dataset_idx between graph {gidx1} and graph {gidx2}")
                return None, None

            real_idxs1 = idxs1[i1]
            real_idxs2 = idxs2[i2]

            point1 = self.point[real_idxs1]
            conf1  = self.conf[real_idxs1]
            point2 = self.point[real_idxs2]
            conf2  = self.conf[real_idxs2]
            common_vals = list(common_vals)
            return (point1, conf1), (point2, conf2),common_vals
    def graphimgnum(self):
        with self.lock:
            n = len(self)
            target_graph_idx = self.graph_idx[n-1]
            mask = self.graph_idx == target_graph_idx  
            num_true = mask.sum().item()
            return num_true    
    def set_pointpose_for_graph_idx(self, target_graph_idx, new_pose):
        """
        将 graph_idx == target_graph_idx 的所有 pointpose 赋值为 new_pose。

        参数:
            target_graph_idx (int): 要筛选的 graph_idx 的值。
            new_pose (Tensor): 4x4 的新变换矩阵。
        """
        with self.lock:  # 保证多进程安全
            # 找到 graph_idx == target_graph_idx 的索引
            mask = self.graph_idx == target_graph_idx
            # 如果 new_pose 不是 Tensor，先转成 Tensor
            if not isinstance(new_pose, torch.Tensor):
                new_pose = torch.tensor(new_pose, dtype=self.dtype, device=self.device)

            # 赋值到对应的 pointpose
            self.pointpose[mask] = new_pose
    def get_max_graph_idx(self):
        """
        搜索并返回当前所有 graph_idx 中的最大值。
        仅考虑 is_empty=False 的帧。
        """
        with self.lock:
            # 找到非空帧
            occupied_indices = torch.where(~self.is_empty)[0]
            if len(occupied_indices) == 0:
                print("⚠️ 当前没有任何占用帧，返回 -1 作为默认最大 graph_idx。")
                return -1

            # 提取所有非空帧的 graph_idx
            graph_indices = self.graph_idx[occupied_indices]
            max_idx = torch.max(graph_indices).item()
            print(f"✅ 当前所有 graph_idx 中的最大值: {max_idx}")
            return max_idx
        
    def count_prev_mapframes(self):
        """
        搜索并返回当前所有 graph_idx 中的最大值。
        仅考虑 is_empty=False 的帧。
        """
        with self.lock:
            # 找到非空帧
            occupied_indices = torch.where(~self.is_empty)[0]
            if len(occupied_indices) == 0:
                print("⚠️ 当前没有任何占用帧，返回 -1 作为默认最大 graph_idx。")
                return -1

            # 提取所有非空帧的 graph_idx
            graph_indices = self.graph_idx[occupied_indices]
            max_idx = torch.max(graph_indices).item()
            condition = (self.graph_idx == max_idx) & self.ismapframe
            count = condition.sum().item()
            return count

    def search_loop(self, offset=35):
        """
        获取早于当前帧 offset 帧的、非空且 ismapframe=True 的 keyframe 索引。
        以及 offset 以内的（不早于 offset 的）索引。

        返回:
            early_indices (List[int]): 早于 offset 的 keyframe 索引
            late_indices  (List[int]): offset 以内的 keyframe 索引
        """
        with self.lock:
            # 找出非空的 keyframe 索引
            valid_mask = ~self.is_empty & self.ismapframe
            valid_indices = torch.where(valid_mask)[0]

            if len(valid_indices) <= offset:
                # 没有足够帧，全部算 late
                early_indices = []
                late_indices = valid_indices.tolist()
            else:
                early_indices = valid_indices[:-offset].tolist()
                late_indices = valid_indices[-offset:].tolist()

            return early_indices, late_indices

    def print_occupied_frames_info(self):
        """
        遍历所有 is_empty 为 False 的帧，输出对应的 ismapframe, graph_idx, dataset_idx, 以及 pointpose 的第一列。
        """
        with self.lock:
            # 找到所有 is_empty 为 False 的索引
            occupied_indices = torch.where(~self.is_empty)[0]

            if len(occupied_indices) == 0:
                print("✅ 没有已占用的帧，所有帧都是空的。")
                return

            print(f"📝 找到 {len(occupied_indices)} 个 is_empty = False (已占用) 的帧：")

            for idx in occupied_indices:
                idx_int = idx.item()
                ismap = self.ismapframe[idx_int].item()
                graph_id = self.graph_idx[idx_int].item()
                dataset_id = self.dataset_idx[idx_int].item()
                pointpose_first_col = self.pointpose[idx_int][:, 0].cpu().numpy()  # (4,)

                # 格式化输出第一列
                first_col_str = ", ".join([f"{v:.4f}" for v in pointpose_first_col])
                print(f" - Index: {idx_int} | ismapframe: {ismap} | "
                    f"graph_idx: {graph_id} | dataset_idx: {dataset_id} | "
                    f"pointpose[:,0]: [{first_col_str}]")

            print("-" * 50)

        return occupied_indices

    def update_pointpose_all_graphs(self, pose_list):
        """
        给定每个 graph 的 pose，更新当前 keyframes 中所有属于该 graph 的帧的 pointpose。
        pose_list 的顺序应与当前图中所有唯一 graph_idx 的顺序一致。
        
        如果 pose_list 较短（说明期间 keyframes 被新图更新），只更新前 len(pose_list) 个 graph。
        """
        with self.lock:        
            unique_graph_ids = torch.unique(self.graph_idx[self.is_empty == False]).tolist()

            if len(pose_list) != len(unique_graph_ids):
                print(f"⚠️ pose_list 长度 {len(pose_list)} 与当前图数量 {len(unique_graph_ids)} 不一致，"
                    f"仅更新前 {len(pose_list)} 个图。")

            # ✅ 只使用前 len(pose_list) 个 graph
            update_graph_ids = unique_graph_ids[:len(pose_list)]

            for pose, gid in zip(pose_list, update_graph_ids):
                if not torch.is_tensor(pose):
                    pose = torch.tensor(pose, dtype=self.dtype, device=self.device)
                else:
                    pose = pose.to(dtype=self.dtype, device=self.device)

                mask = (self.graph_idx == gid) & (~self.is_empty)
                indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)

                for idx in indices:
                    self.pointpose[idx] = pose
                    self.dirty[idx] = True


    def get_imgparis(self,kfs_idxs,current_idx):
        with self.lock: 
            kfimgs = []
            for i in kfs_idxs:
                kfimgs.append(self.color[i]) 
                
            img2 = self.color[current_idx]
            return kfimgs, img2     

    def check_projection_coverage(self, current_idx,points3d, min_coverage_ratio=0.75, top_k=20):
        """
        检查3D点在最近关键帧中的投影覆盖率
        
        参数:
            points3d: (N, 3) 当前帧的3D点云坐标
            min_coverage_ratio: 最小有效投影比例阈值(默认0.6即60%)
            top_k: 考虑的最邻近关键帧数量(默认20)
        
        返回:
            bool: 是否满足覆盖率要求
            int: 实际满足的帧数
            float: 平均覆盖率
        """
        
        with self.lock:  # 确保线程安全
            # 1. 获取有效的关键帧索引
            points3d = np2torch(points3d).to(self.device)
            valid_mask = ~self.is_empty & self.ismapframe 
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) <10:
                return None,None
            selected_kf_indices = torch.where(valid_mask)[0][:-6]
            if len(selected_kf_indices)>30:
                selected_kf_indices = selected_kf_indices[-30:]
            # 3. 投影检查
            points3d_hom = torch.cat([points3d, torch.ones(len(points3d), 1, device=self.device)], dim=1)
            total_covered = 0
            coverkfs = []
            coverages = []
            for kf_idx in selected_kf_indices:
                # 获取关键帧的相机参数
                c2w = self.c2ws[kf_idx]
                pointpose = self.pointpose[kf_idx]
                c2w_global = pointpose @ c2w 

                w2c = torch.inverse(c2w_global)
                K = self.K
                # 投影计算
                points_cam = (w2c @ points3d_hom.T).T[:, :3]  # 转相机坐标系
                points_img = (K @ points_cam.T).T              # 投影到图像平面
                pixels = points_img[:, :2] / points_img[:, 2:] # 归一化像素坐标
                
                # 检查是否在图像范围内
                in_x = (pixels[:, 0] >= 0) & (pixels[:, 0] < self.w)
                in_y = (pixels[:, 1] >= 0) & (pixels[:, 1] < self.h)
                in_front = points_cam[:, 2] > 0  # z>0表示在相机前方
                visible = in_x & in_y & in_front
                
                coverage = visible.float().mean()
                # coverage_ratios.append(coverage.item())
                if coverage >= min_coverage_ratio:
                    total_covered += 1
                    coverkfs.append(kf_idx.item())
                    coverages.append(coverage.item())
            return coverkfs,coverages

    def check_projection_coverage_loop(self, selected_kf_indices, points3d, min_coverage_ratio=0.6, top_k=20):
        """
        检查3D点在最近关键帧中的投影覆盖率
        返回:
            coverkfs: 满足 min_coverage_ratio 的关键帧 index 列表
        """
        with self.lock:  # 确保线程安全
            points3d = np2torch(points3d).to(self.device)
            points3d_hom = torch.cat([points3d, torch.ones(len(points3d), 1, device=self.device)], dim=1)
            
            total_covered = 0
            coverkfs = []
            coverages = []
            for kf_idx in selected_kf_indices:
                if self.ismapframe[kf_idx] == True:
                    c2w = self.c2ws[kf_idx]
                    pointpose = self.pointpose[kf_idx]
                    c2w_global = pointpose @ c2w

                    w2c = torch.inverse(c2w_global)
                    K = self.K
                    points_cam = (w2c @ points3d_hom.T).T[:, :3]
                    points_img = (K @ points_cam.T).T
                    pixels = points_img[:, :2] / points_img[:, 2:]

                    in_x = (pixels[:, 0] >= 0) & (pixels[:, 0] < self.w)
                    in_y = (pixels[:, 1] >= 0) & (pixels[:, 1] < self.h)
                    in_front = points_cam[:, 2] > 0
                    visible = in_x & in_y & in_front

                    coverage = visible.float().mean()
                    realidx = self.dataset_idx[kf_idx]
                    # ✅ 打印该关键帧的 index 和可见内点数
                    # print(f"关键帧 {realidx}: 内点数量 = {visible.sum()} / {len(points3d)}, 覆盖率 = {coverage:.2f}")

                    if coverage >= min_coverage_ratio:
                        total_covered += 1
                        coverkfs.append(kf_idx)
                        coverages.append(coverage.item())
            
            return coverkfs,coverages


def analyze_unique_point_observations_per_frame(frame_history_map, accuratemap):
    """
    统计每一帧被观测到的“唯一点”的数量，并细分 accurate 状态（True/False/None）

    参数：
        frame_history_map: List[List[(frame_id, (u, v))]]
        accuratemap: List[bool or None]

    返回：
        frame_stats: dict[frame_id] = {'True': x, 'False': y, 'None': z}
    """
    assert len(frame_history_map) == len(accuratemap), "长度不一致"

    frame_to_point_ids = defaultdict(set)  # 每帧观测到哪些唯一点
    for point_idx, history in enumerate(frame_history_map):
        for frame_id, _ in history:
            frame_to_point_ids[frame_id].add(point_idx)

    frame_stats = {}
    for frame_id, point_ids in frame_to_point_ids.items():
        stats = {"True": 0, "False": 0, "None": 0}
        for pid in point_ids:
            acc = accuratemap[pid]
            if acc is True:
                stats["True"] += 1
            elif acc is False:
                stats["False"] += 1
            else:
                stats["None"] += 1
        frame_stats[frame_id] = stats

    # 打印结果
    print("📊 每帧观测的【唯一点数】统计：")
    for frame_id in sorted(frame_stats):
        s = frame_stats[frame_id]
        total = s["True"] + s["False"] + s["None"]
        print(f" - 第 {frame_id} 帧: 共 {total} 个点（✅True: {s['True']}, ❌False: {s['False']}, ❓None: {s['None']}）")

    return frame_stats

def analyze_observation_per_frame(frame_history_map, accuratemap):
    """
    分析：每一帧中被观测到的点数，以及它们的 accurate 状态。

    参数：
        frame_history_map: List[List[Tuple[frame_id, (u, v)]]]
        accuratemap: List[bool or None]

    返回：
        frame_stats: dict[frame_id] = {'True': x, 'False': y, 'None': z}
    """
    assert len(frame_history_map) == len(accuratemap), "长度不一致"

    frame_stats = defaultdict(lambda: {"True": 0, "False": 0, "None": 0})

    for history, acc in zip(frame_history_map, accuratemap):
        for frame_id, _ in history:
            if acc is True:
                frame_stats[frame_id]["True"] += 1
            elif acc is False:
                frame_stats[frame_id]["False"] += 1
            else:
                frame_stats[frame_id]["None"] += 1

    # 打印结果
    print("✅ 帧级别观测统计 (每帧被观测的点数 + Accurate 状态)：")
    for frame_id in sorted(frame_stats):
        s = frame_stats[frame_id]
        total = s["True"] + s["False"] + s["None"]
        print(f" - 第 {frame_id} 帧: 共 {total} 点（True: {s['True']}, False: {s['False']}, None: {s['None']}）")

    return dict(frame_stats)

def build_current_frame_maps_from_last(
    keyframe_last,
    cur_frame_idx: int,
    points3d: np.ndarray,
    points2d: np.ndarray,
    kf2dp_kf: np.ndarray
):
    """
    根据上一帧的 MapPoints，构建当前帧的 frame_history_map 和 accurate map。

    返回：
        frame_history_map: (H, W) object array
        accuratemap: (H, W) object array
    """
    H, W = keyframe_last["frame_history_map"].shape
    frame_history_map = np.full((H, W), None, dtype=object)
    accuratemap = np.full((H, W), None, dtype=object)

    for i, (xyz, uv_curr, uv_last) in enumerate(zip(points3d, points2d, kf2dp_kf)):
        u_last, v_last = np.round(uv_last).astype(int)
        u_curr, v_curr = np.round(uv_curr).astype(int)

        if not (0 <= v_last < H and 0 <= u_last < W):
            continue
        if not (0 <= v_curr < H and 0 <= u_curr < W):
            continue

        mp = keyframe_last["frame_history_map"][v_last, u_last]
        if mp is not None:
            mp.position = xyz  # 更新位置
            mp.add_observation(cur_frame_idx, uv=uv_curr, keypoint_index=i)

            frame_history_map[v_curr, u_curr] = mp
            accuratemap[v_curr, u_curr] = (
                keyframe_last["accuratemap"][v_last, u_last] is True
            )

    return frame_history_map, accuratemap


def apply_dense_correction_from_backend(
    keyframe: dict,
    map3d: np.ndarray,
    corrected_kf_id: int,
    pose
):
    """
    用后端 map3d 更新前端 keyframe 的 3D 点，并更新相机位姿。

    参数：
        keyframe: dict，包含 points3d, frame_history_map, accuratemap, pose
        map3d: (H, W, 3) ndarray，稠密点云（优化后）
        corrected_kf_id: int，后端优化的是哪一帧
        pose: np.ndarray or torch.Tensor，新的优化相机位姿（SE3或4x4矩阵）

    返回：
        updated_indices: list[int]，表示哪些点被更新
    """
    H, W, _ = map3d.shape
    updated_indices = []

    for i, history in enumerate(keyframe["frame_history_map"]):
        for frame_id, (u, v) in history:
            if frame_id == corrected_kf_id:
                u, v = int(round(u)), int(round(v))
                if 0 <= v < H and 0 <= u < W:
                    keyframe["points3d"][i] = map3d[v, u]
                    keyframe["accuratemap"][i] = True
                    updated_indices.append(i)
                else:
                    print(f"⚠️ 点 {i} 的像素 ({u},{v}) 超出 map3d 范围")
                break

    # 更新位姿（假设字段是 "kfpose" 或 "pose"）
    keyframe["kfpose"] = pose
    print(f"✅ 已根据后端第 {corrected_kf_id} 帧更新 {len(updated_indices)} 个点并同步位姿")
    return updated_indices


