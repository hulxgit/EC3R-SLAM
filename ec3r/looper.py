
import time
import torch.multiprocessing as mp
from fastrender import mapinfer,init_infer,image_similarity
import open3d as o3d
import torch
from fast3r.models.fast3r_custom import Fast3R
from recoverpose import reestimate_pose,recover_pose_init,estimation_pose_from_point
from ec3r.utils.slam_utils import match,select_keyframes,unpad_tensor,compute_mean_std_filter
import numpy as np
from termcolor import colored
import queue
from ec3r.utils.slam_utils import torch2np,np2torch
import cv2
from verify_loop import detect_loop_closure
from save_file import save_keyframes_pointcloud
import pypose as pp 
from tabulate import tabulate
def process_pending_loops(pending_loops):
    """
    展平、筛选 sim > 0.7 的回环项，去除重复 current_gid，只保留 sim 最大者，并返回最多3个回环。
    参数:
        pending_loops: List[Tuple(current_idx, idx, current_gid, idx_gid, sim)] 或 嵌套 List
    返回:
        final_loops: List[Tuple(...)]，最多3个
    """
    from collections import defaultdict

    # 1. 展平
    if isinstance(pending_loops[0], list):
        flat_list = [item for group in pending_loops for item in group]
    else:
        flat_list = pending_loops

    # 2. 打印所有原始项
    print("\n📦 [原始 Pending Loop 项目] 总数:", len(flat_list))
    for i, item in enumerate(flat_list):
        print(f"    🔄 {i+1}: 当前帧 {item[0]} (g{item[2]}) ↔ 匹配帧 {item[1]} (g{item[3]}) | sim={item[4]:.4f}")
    print("--------------------------------------------------")

    # 3. 过滤 sim ≤ 0.7 的项
    filtered_raw = [item for item in flat_list if item[4] > 0.7]

    # # 4. 每个 idx_gid 保留最高 sim 的项
    # gid_to_best = defaultdict(lambda: (-1, -1, -1, -1, -float('inf')))
    # for item in filtered_raw:
    #     current_idx, idx, current_gid, idx_gid, sim = item
    #     if sim > gid_to_best[idx_gid][-1]:
    #         gid_to_best[idx_gid] = item

    # # 5. 排序并截断
    # filtered_items = list(gid_to_best.values())
    # filtered_items.sort(key=lambda x: -x[-1])
    # final_loops = filtered_items[:3]
    filtered_items = sorted(filtered_raw, key=lambda x: -x[-1])
    final_loops = filtered_items[:3]
    # 6. 打印最终保留项
    print(f"🎯 [回环筛选结果] 过滤后 sim>0.7 项数: {len(filtered_raw)}")
    print(f"✅ 保留 idx_gid 去重后最大 sim 项数: {len(filtered_items)}")
    print(f"🏁 最终使用的回环数: {len(final_loops)}")

    print("📝 保留回环列表（按相似度排序）：")
    for i, item in enumerate(final_loops):
        print(f"    #{i+1}: 当前帧 {item[0]} (g{item[2]}) ↔ 匹配帧 {item[1]} (g{item[3]}) | sim={item[4]:.4f}")
    print("==================================================✨")

    return final_loops

class FrameLooper(mp.Process):
    def __init__(self, config,keyframes,pgo,loop_indices):
        super().__init__()
        self.config = config
        self.device = "cpu"
        self.keyframes = keyframes
        self.looper_queue = None 
        self.loop_indices = loop_indices
        self.K =None
        self.pgo = pgo
        self.mapper_queue = None
        self.tracker_queue = None
        self.loop_punish =0
        self.loop_cfg = self.config["looping"]
        self.search_begin = self.loop_cfg["search_begin"]
        self.search_end = self.loop_cfg["search_end"]
        self.begin_idx = self.loop_cfg["search_gap"]
        self.search_gap = self.loop_cfg["search_gap"]
        self.sim_thre = self.loop_cfg["sim_thre"]
        self.loop_score = self.loop_cfg["loop_score"]
        self.search_offset = self.loop_cfg["search_offset"]

    def run(self):
        keyframelen = 0
        record_loop_idx =0
        record_loop_idx2 = []
        loop_match_records = []
        pending_loops = []
        while True:
            try:
                # 非阻塞或带超时地尝试获取指令消息
                data = self.looper_queue.get(timeout=0.1)
                if data[0] == "stop":
                    print("[Looper] 🛑 Stopping...")
                    self.keyframes.print_occupied_frames_info()
                    break
                elif data[0] == "add_odometry_factor":
                    self.pgo.add_frame_node(data[4])

                    self.pgo.add_odometry_factor(data[1],data[2],data[3])
                    # self.keyframes.print_occupied_frames_info()
                    # self.pgo.print_graph()
                elif data[0] == "add_global_loop_factor":
                    tfs =  data[3]
                    new_pointpose = data[2]
                    cur_gf = data[1]
                    idx_loop = data[4]
                    global_scale = 5 # 全局放大 10 倍
                    info_matrix = global_scale * torch.eye(7, device=self.device)


                    for gid, tf in tfs:
                        self.pgo.add_odometry_factor(gid,cur_gf,tf,info_matrix)
                    # self.pgo.print_graph()
                    pose_after_pgo = self.pgo.optimize_pose_graph()
                    
                    if pose_after_pgo:
                        pose_after_pgo1 = pose_after_pgo[1:]
                        #这一步会出现问题
                        self.keyframes.update_pointpose_all_graphs(pose_after_pgo1)
                        self.loop_indices.add_batch(idx_loop, len(self.keyframes)-1,source="loop")
                    else:#delete!!!!<empty
                        self.pgo.delete_node_edge(cur_gf)
                    self.pgo.print_graph()

                elif data[0] == "add_global_loop_factor_new":
                    tfs =  data[3]
                    new_pointpose = data[2]
                    cur_gf = data[1]
                    idx_loop = data[4]
                    global_scale = 5 # 全局放大 10 倍
                    info_matrix = global_scale * torch.eye(7, device=self.device)
                    self.pgo.add_frame_node(new_pointpose)
                    self.pgo.nodes[cur_gf] = pp.mat2Sim3(new_pointpose).squeeze(0)

                    for gid, tf in tfs:
                        self.pgo.add_odometry_factor(gid,cur_gf,tf,info_matrix)
                    # self.pgo.print_graph()
                    pose_after_pgo = self.pgo.optimize_pose_graph()
                    
                    if pose_after_pgo:
                        pose_after_pgo1 = pose_after_pgo[1:]
                        #这一步会出现问题
                        self.keyframes.update_pointpose_all_graphs(pose_after_pgo1)
                        self.loop_indices.add_batch(idx_loop, len(self.keyframes)-1,source="loop")
                        self.keyframes.print_occupied_frames_info()
                        self.loop_punish+=10
                    else:#delete!!!!<empty
                        self.pgo.delete_node_edge(cur_gf)
                    self.pgo.print_graph()
                    


                elif data[0] == "add_prior":
                    self.pgo.add_pose_prior(data[1],data[2],data[3])
                elif data[0] == "add_local_loop_factor":
                    cur_gid = data[1]
                    tf_new_to_history_list = data[2]
                    info = data[3]
                    for history_idx, tf_new_to_hist in tf_new_to_history_list:
                        self.pgo.add_loop_factor(
                            idx1=history_idx,       # 历史帧 index
                            idx2=cur_gid,      # 当前帧 index
                            relative_pose=tf_new_to_hist,
                            info=info  # loop info 可以根据 tf 质量动态调整
                        )
                    self.pgo.print_graph() 
                    pose_after_pgo = self.pgo.optimize_pose_graph()


                    if pose_after_pgo:
                        pose_after_pgo = pose_after_pgo[1:]
                        self.keyframes.update_pointpose_all_graphs(pose_after_pgo)
                        print("pose_after_pgo",len(pose_after_pgo))
                        #这一步会出现问题
                        
                    else:
                        for history_idx, tf_new_to_hist in tf_new_to_history_list:
                            self.pgo.delete_factor(
                            idx1=history_idx,       # 历史帧 index
                            idx2=cur_gid,      # 当前帧 index

                        )
                     
                elif data[0] == "SHUTDOWN":

                    pose_after_pgo = self.pgo.optimize_pose_graph()
                    if pose_after_pgo:
                        pose_after_pgo = pose_after_pgo[1:]
                        #这一步会出现问题
                        self.keyframes.update_pointpose_all_graphs(pose_after_pgo)
                    else:
                        for history_idx, tf_new_to_hist in tf_new_to_history_list:
                            self.pgo.delete_factor(
                            idx1=history_idx,       # 历史帧 index
                            idx2=cur_gid,      # 当前帧 index
                            )
                    print("[looper] Received shutdown signal, cleaning up...")
                    self.pgo.print_graph()  
                    self.pgo.print_edge_losses()

                    self.keyframes.print_occupied_frames_info()

                    # ✅ 向 mapper 发送确认
                    self.looper_queue.put(["SHUTDOWN_ACK"])
                    print("[looper] Sent SHUTDOWN_ACK to mapper. Looper shutdown done.")

                    # ✅ 如果 looper 线程/循环是 while True，需要 break 或 return
                    # 例如：
                    break

            except queue.Empty:
                pass  # 没有消息就跳过，不阻塞

            # ✅ 每一轮都可以执行回环检测逻辑
            time.sleep(0.1)
            newlen = len(self.keyframes)
            
            if newlen != keyframelen and self.keyframes.ismapframe[newlen-1]==True and self.keyframes.graphimgnum()>2:
                
                keyframelen = newlen
                if self.loop_punish>0:
                    self.loop_punish = self.loop_punish-1
                if keyframelen > self.begin_idx and self.loop_punish==0:
                    returnidx, late_indices = self.keyframes.search_loop(offset=self.search_offset)



                    
                    current_feat = self.keyframes.get_feat(0).squeeze().to(self.device)
                    current_point,current_c2w = self.keyframes.get_posepoint(len(self.keyframes)-1)

                    def get_similarity(index,feat,mode="cross_topk"):
                        if 0 <= index < len(self.keyframes.feat):
                            pastfeat = self.keyframes.feat[index].squeeze().to(self.device)
                            return image_similarity(feat, pastfeat,mode=mode)
                        return 0

                      # 🔹 存放所有合法回环匹配 (current_idx, loop_idx, similarity)
                    found_loop = False  # 添加标志变量
                    send_loop = []         

                    for i in range(self.search_begin, len(returnidx)-self.search_end, self.search_gap):  
                        idx = returnidx[i]
                        if record_loop_idx in late_indices or idx in record_loop_idx2 or idx in self.loop_indices.loop_indices:                            
                            continue   #过滤掉附近的
                        
                        sim = get_similarity(idx,current_feat)
                        # 用 emoji 和分隔符突出显示 sim 的值


                        if sim > self.sim_thre:
                            current_idx = newlen - 1
                            current_gid = int(self.keyframes.search_graph_idx(current_idx))
                            idx_gid = int(self.keyframes.search_graph_idx(idx))

                            kf_info = [(current_idx, idx, current_gid, idx_gid, sim)]
                            neighbors = [s for s in returnidx[i-2:i+3]]
                            
                            # [i - 2, i - 1,i, i + 1, i + 2]
                            count_high = 0
                            neighbor_info = []
                            feat_matrix = np.ones((5, 5))*0.001  # Use a tuple (5, 5) for the shape
                            for j,n in enumerate(neighbors):
                                for k in range(0,5):
                                    neighbor_idx =neighbors[j]
                                    # kf_info.append((current_idx, neighbor_idx, current_gid, neighbor_gid, neighbor_sim))
                                    # neighbor_info.append((neighbor_idx, neighbor_sim))
                                    feat_current = self.keyframes.get_feat(k).squeeze().to(self.device)
                                    neighbor_sim = get_similarity(neighbor_idx,feat_current)
                                    neighbor_gid = int(self.keyframes.search_graph_idx(neighbor_idx))
                                    neighbor_info.append((neighbor_idx, neighbor_sim))
                                    feat_matrix[j,k] = neighbor_sim



                            # 横坐标 ： [self.keyframes[current_idx-j] for j in range(5)]
                            # 纵坐标: [self.keyframes[x] for x in neighbors]
                            # X-coordinates: [self.keyframes[current_idx-j] for j in range(5)]
                            x_coords = [self.keyframes.dataset_idx[current_idx - j].item() for j in range(5)]

                            # Y-coordinates: [self.keyframes[x] for x in neighbors]
                            y_coords = [self.keyframes.dataset_idx[x].item()  for x in neighbors]
                            table = []
                            for i, y in enumerate(y_coords):
                                row = [y] + [f"{val:.2f}" for val in feat_matrix[i]]  # Format matrix values
                                table.append(row)

                            # Print the matrix with coordinates in a table
                            headers = ["Y/X"] + x_coords  # Header with x-coordinates
                            print("Feature Matrix with Coordinates:")
                            print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
                            top_k = 5
                            loop_score = 0
                            loop_pairs = []
                            flat_indices = np.argsort(feat_matrix.ravel())[-top_k:][::-1]  # 从大到小排序
                            row_indices, col_indices = np.unravel_index(flat_indices, feat_matrix.shape)
                            for i,j in zip(row_indices, col_indices ):#行，列
                                map_idx1 = neighbors[i]
                                map_idx2 = current_idx-j
                                xfeat_desc1 = unpad_tensor(self.keyframes.xfeat_desc[map_idx1])
                                xfeat_desc2 =  unpad_tensor(self.keyframes.xfeat_desc[map_idx2])
                                xfeat_kpt1 = unpad_tensor(self.keyframes.xfeat_kpts[map_idx1])
                                xfeat_kpt2 = unpad_tensor(self.keyframes.xfeat_kpts[map_idx2])
                                assert xfeat_desc1.shape[0] == xfeat_kpt1.shape[0]

                                idxs0, idxs1 = match(xfeat_desc1,xfeat_desc2, min_cossim=-1)
                                points1 = xfeat_kpt1[idxs0].cpu().numpy()
                                points2 = xfeat_kpt2[idxs1].cpu().numpy()
                                _, _, mean_disp ,std_disp= compute_mean_std_filter(points1, points2)

                                F, mask = cv2.findFundamentalMat(points1, points2, method=cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)

                                inliers = np.where(mask.ravel() == 1)[0]  # 内点索引
                                outliers = np.where(mask.ravel() == 0)[0]  # 外点索引

                                # 计算内点比例
                                total_matches = len(points1)  # 总匹配点数量
                                inlier_count = len(inliers)  # 内点数量
                                inlier_ratio = inlier_count / total_matches if total_matches > 0 else 0.0
                                print(f"🔍 Pair (i={i}, j={j})")
                                print(f"   🔸 Similarity Score     : {feat_matrix[i][j]:.4f}")
                                print(f"   🔸 Mean Displacement    : {mean_disp:.2f}")
                                print(f"   🔸 Std Displacement     : {std_disp:.2f}")
                                print(f"   🔸 Inlier Ratio (RANSAC): {inlier_ratio:.2%}  ({inlier_count}/{total_matches})")
                                print("-" * 60)
                                index_Pair = (map_idx1,map_idx2,inlier_ratio)
                                loop_pairs.append(index_Pair)

                                if feat_matrix[i][j]>0.8:
                                    loop_score +=  5
                                if inlier_ratio>0.2:
                                    loop_score += 10
                                elif 0.2>inlier_ratio >0.15:
                                    loop_score += 5
                                elif  0.15>inlier_ratio >0.1:
                                    loop_score +=2
                                if std_disp < 30:
                                    loop_score += 5
                                elif 30<std_disp<60:
                                    loop_score += 1

                            if loop_score > self.loop_score:
                                print("🔥 回环成立 🎉")
                                print(f"🚀 loop_score is {loop_score} 💥")
                                print()
                                loop_pairs_sorted = sorted(loop_pairs, key=lambda x: x[2], reverse=True)
                                def insert_list_every_n(a, b, n):
                                    result = []
                                    b_idx = 0  # 用来记录 b 中元素的位置

                                    for i, val in enumerate(a):
                                        result.append(val)
                                        # 每隔 n 个就插入 b 中的一个元素（如果还有）
                                        if (i + 1) % n == 0 and b_idx < len(b):
                                            result.append(b[b_idx])
                                            b_idx += 1

                                    return result
                                if send_loop:
                                    n = int(len(send_loop)/5)
                                    print("n",n)

                                    send_loop = insert_list_every_n(send_loop,loop_pairs_sorted, n)
                                else:
                                    send_loop+= loop_pairs_sorted

                            
                            
                    if send_loop:       
                        send_loop_data_idx =[]#回环不能有重复！'
                        final_loop = []
                        for (id1, id2, _) in send_loop:
                            if self.keyframes.dataset_idx[id1] not in send_loop_data_idx:
                                final_loop.append(id1)
                                send_loop_data_idx.append(self.keyframes.dataset_idx[id1])
                            
                            if self.keyframes.dataset_idx[id2] not in send_loop_data_idx:
                                final_loop.append(id2)
                                send_loop_data_idx.append(self.keyframes.dataset_idx[id2])
                            
                            if len(final_loop) > 4:
                                break  # 正确地退出 for 循环
                                
                        print("send",final_loop)
                        self.mapper_queue.put(["loop_new",final_loop])
                                
    def compute_projection_overlap(self, current, index, intrinsics):
        """
        利用图像投影后的像素坐标交集判断两个帧是否有 overlap。

        参数:
            current, index: 帧索引
            intrinsics: 相机内参 (Tensor [3, 3])

        返回:
            overlap_ratio: 投影像素坐标交集比例
        """
        current_point, current_c2w = self.keyframes.get_posepoint(current)  # [H, W, 3]
        index_point, index_c2w = self.keyframes.get_posepoint(index)
        H, W, _ = current_point.shape

        def project_world_points(points_world, target_c2w, intrinsics):
            w2c = torch.inverse(target_c2w)
            N = points_world.shape[0] * points_world.shape[1]
            points_world = points_world.reshape(-1, 3)
            points_h = torch.cat([points_world, torch.ones((N, 1), device=points_world.device)], dim=1)  # [N, 4]

            cam_points = (w2c @ points_h.T).T[:, :3]  # [N, 3] in camera frame
            in_front = cam_points[:, 2] > 0

            proj = (intrinsics @ cam_points.T).T  # [N, 3]
            uv = proj[:, :2] / proj[:, 2:3]

            in_img = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            valid = in_front & in_img
            uv_int = uv[valid].long()

            pixel_coords = set(map(tuple, uv_int.cpu().numpy()))  # set of (u, v)
            return pixel_coords

        # current 点投影到 index 图像
        set1 = project_world_points(current_point, index_c2w, intrinsics)
        # index 点投影到 current 图像
        set2 = project_world_points(index_point, current_c2w, intrinsics)

        intersection = set1.intersection(set2)
        denom = max(len(set1), len(set2))
        ratio = len(intersection) / denom if denom > 0 else 0.0

        print(f"🎯 投影像素交集: {len(intersection)} / {denom} → ratio = {ratio:.3f}")
        return ratio
