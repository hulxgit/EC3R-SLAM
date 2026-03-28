import torch
from ec3r.frame import (
    filter_kf_clusters_exclusive,
    apply_dense_correction_from_backend,
    expand_and_include_extras,
)
import traceback
from accelerated_features.modules.xfeat import XFeat
import torch.multiprocessing as mp
from ec3r.utils.fast3r_utils import resize_img
import cv2
import numpy as np
from ec3r.utils.slam_utils import (
    dedup_and_difference_clustering,
    unpad_tensor,
    get_top_k_frame_ids,
    pad_tensor_to,
    filter_points_in_front_of_cameras,
    compute_mean_std_filter,
    compute_average_depth,
    torch2np,
    np2torch,
    warp_corners_and_draw_matches,
)
import time
import queue
from pyposeba import run_full_local_ba, run_structure_only_ba, estimate_pose_epnp
import copy


def to_cpu_recursive(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    elif isinstance(x, dict):
        return {k: to_cpu_recursive(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_cpu_recursive(v) for v in x]
    elif isinstance(x, tuple):
        return tuple(to_cpu_recursive(v) for v in x)
    else:
        return x


class FrameTracker(mp.Process):
    def __init__(self, config, shared_imgmatch, shared_points3d, currentpose, mapper_keyframes, loopindices):
        self.cfg_tracker = config["tracking"]
        self.inlier = config["inlier_threshold"]
        ransac_method = self.inlier["RANSAC"]
        method_mapping = {
            "RANSAC": cv2.RANSAC,
            "USAC_FAST": cv2.USAC_FAST,
            "USAC_MAGSAC": cv2.USAC_MAGSAC,
            "USAC_ACCURATE": cv2.USAC_ACCURATE,
            "USAC_PARALLEL": cv2.USAC_PARALLEL,
            "LMEDS": cv2.LMEDS,
        }

        if ransac_method not in method_mapping:
            raise ValueError(
                f"Unsupported RANSAC method: {ransac_method}. "
                f"Supported methods: {list(method_mapping.keys())}"
            )

        self.ransac_method = method_mapping[ransac_method]
        self.mid_threshold = self.inlier["mid"]
        self.extractmatch = XFeat()
        self.pointkeys = "global_pointmaps"
        self.matchtopk = self.cfg_tracker["matchtopk"]
        self.use_gui = True
        self.n_map = self.cfg_tracker["n_map"]
        self.record_match = True
        self.initialized = False
        self.device = "cuda:0"
        self.point3d = shared_points3d
        self.keyframes = []
        self.tracker_queue = None
        self.mapper_queue = None
        self.img_size = 518
        self.K = None
        self.initnum = 6
        self.overlap = 1
        self.initgap = 1
        self.shared_imgmatch = shared_imgmatch
        self.match_record = []
        self.currentpose = currentpose
        self.mapper_keyframes = mapper_keyframes
        self.loopindices = loopindices
        self.update_stats = False

    def track(self, curframe, keyframe):
        import time

        t_prepare_start = time.time()
        iskf = False
        match = dict()
        timestamp, img = self.dataset[curframe]
        img = resize_img(img, self.img_size)
        cframe = img["backend_img"].to(device=self.device)
        _, H, W = cframe.shape
        disp_threshold = ((H / self.cfg_tracker["init_kf_dis"]) * (W / self.cfg_tracker["init_kf_dis"])) ** 0.5
        t_prepare_end = time.time()
        t_prepare = t_prepare_end - t_prepare_start

        t_feat_start = time.time()
        cfeats = self.extractmatch.detectAndCompute(img["unnormalized_img"], top_k=self.matchtopk)[0]
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        t_feat_end = time.time()
        t_feat = t_feat_end - t_feat_start

        t_match_start = time.time()
        img = dict(img=cframe)

        idxs0, idxs1 = self.extractmatch.match(
            keyframe["features"]["descriptors"],
            cfeats["descriptors"],
            min_cossim=-1,
        )

        points1 = keyframe["features"]["keypoints"][idxs0].cpu().numpy()
        points2 = cfeats["keypoints"][idxs1].cpu().numpy()

        _, _, mean_disp, mean_std = compute_mean_std_filter(points1, points2)
        t_match_end = time.time()
        t_match = t_match_end - t_match_start

        t_pose_start = time.time()
        match["points"] = (points1, points2)
        match["timestamp"] = timestamp
        match["index"] = (keyframe["index"], curframe)
        keyframe["matches"].append(match)

        a = points1.copy().view([("", points1.dtype)] * 2).squeeze()
        b = keyframe["points2d"].copy().view([("", points1.dtype)] * 2).squeeze()

        inter, idx_a, idx_b = np.intersect1d(a, b, return_indices=True)

        selected_points3d = keyframe["points3d"][idx_b]
        selected_points2d_kf = keyframe["points2d"][idx_b]
        selected_points2d = points2[idx_a]
        selected_accurate = [keyframe["accuratemap"][i] for i in idx_b]
        selected_historyframe = [keyframe["frame_history_map"][i] for i in idx_b]

        selected_points3dtrue = selected_points3d[selected_accurate]
        selected_points2dtrue = selected_points2d[selected_accurate]

        success3d = False
        rvec, tvec, inliers = None, None, None
        ratio = None

        try:
            if len(selected_points3dtrue) >= 4:
                success3d, rvec, tvec, inliers = cv2.solvePnPRansac(
                    selected_points3dtrue,
                    selected_points2dtrue,
                    self.K,
                    None,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    reprojectionError=3.0,
                    confidence=0.99,
                    iterationsCount=100,
                )
                ratio = len(inliers) / len(selected_points3dtrue)
        except Exception as e:
            success3d = False
            print(f"[WARN] solvePnPRansac with accurate points failed: {e}")

        if not success3d:
            print("[INFO] Falling back to all selected points.")
            try:
                if len(selected_points3d) >= 4:
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        selected_points3d,
                        selected_points2d,
                        self.K,
                        None,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                        reprojectionError=3.0,
                        confidence=0.99,
                        iterationsCount=100,
                    )
                    ratio = len(inliers) / len(selected_points3d)

            except Exception as e:
                print(f"[ERROR] solvePnPRansac fallback failed: {e}")

        optimized = False
        t_pose_end = time.time()
        t_pose = t_pose_end - t_pose_start

        t_triangulation = 0
        kf = False
        if ratio is None:
            ratio = 0
        ratio_threshold = self.mid_threshold
        if ("mapped_idx" in keyframe and inliers is not None and ratio < ratio_threshold) or len(selected_points3d) < 50:
            kf = True
        elif keyframe["mapped"] is False:
            if mean_std > 30 or ratio < ratio_threshold or len(selected_points3d) < 50:
                kf = True
        else:
            if mean_std > 30 or ratio < ratio_threshold or len(selected_points3d) < 50:
                kf = True

        if kf:
            if success3d and mean_disp < disp_threshold and len(selected_points3d) > 50:
                kf = False

        if kf:
            print("meanstd", mean_std, "ratio", ratio, "len(selected_points3d)", len(selected_points3d))

        T_curr = np.eye(4)

        if rvec is not None and tvec is not None:
            try:
                R, _ = cv2.Rodrigues(rvec)
                T_curr[:3, :3] = R
                T_curr[:3, 3] = tvec.reshape(-1)
            except cv2.error as e:
                print(f"[WARN] Rodrigues failed, using identity transform: {e}")
        else:
            print("[WARN] PnP failed, using identity transform as default T_curr")
            kf = True

        if kf:
            if len(selected_points3dtrue) > 10:
                average_depth = compute_average_depth(selected_points3dtrue, keyframe["keypose"])
                print("Average Depth (accurate points):", average_depth)
            else:
                average_depth = compute_average_depth(selected_points3d, keyframe["keypose"])
                print("Average Depth (all points):", average_depth)

            iskf = True
            print("[INFO] Creating new keyframe")
            t_update_start = time.time()

            for i, uv in enumerate(selected_points2d):
                u, v = np.round(uv).astype(int)
                selected_historyframe[i].append((curframe, (u, v)))

            all_indices = np.arange(len(points1))
            no_match_indices = np.setdiff1d(all_indices, idx_a)
            recover_points1 = points1[no_match_indices]
            recover_points2 = points2[no_match_indices]
            t_update_end = time.time()
            t_update = t_update_end - t_update_start

            if len(recover_points1) > 100:

                def triangulate_points(K, T0_w2c, T1_w2c, pts0, pts1):
                    P0 = K @ T0_w2c[:3, :]
                    P1 = K @ T1_w2c[:3, :]
                    pts0 = np.asarray(pts0, dtype=np.float32).T
                    pts1 = np.asarray(pts1, dtype=np.float32).T
                    pts4d = cv2.triangulatePoints(P0, P1, pts0, pts1)
                    pts4d /= pts4d[3]
                    return pts4d[:3].T

                t_triangulation_start = time.time()

                recovered_points3d = triangulate_points(
                    K=self.K,
                    T0_w2c=keyframe["keypose"],
                    T1_w2c=T_curr,
                    pts0=recover_points1,
                    pts1=recover_points2,
                )

                z_valid = filter_points_in_front_of_cameras(
                    recovered_points3d,
                    T0_w2c=keyframe["keypose"],
                    T1_w2c=T_curr,
                )

                recovered_points3d = recovered_points3d[z_valid]
                recover_points1 = recover_points1[z_valid]
                recover_points2 = recover_points2[z_valid]

                t_triangulation_end = time.time()
                t_triangulation = t_triangulation_end - t_triangulation_start

                recover_accurate = [False] * len(recover_points1)
                recover_framehistory = [
                    [(curframe, tuple(np.round(uv).astype(int)))]
                    for uv in recover_points2
                ]

                structba = False
                if structba:
                    t_ba_start = time.time()
                    try:
                        pts3d_optimized = run_structure_only_ba(
                            K=np2torch(self.K),
                            points2d=np2torch(recover_points2),
                            points3d=np2torch(recovered_points3d),
                            fixed_pose=np2torch(T_curr),
                            device=self.device,
                        )
                        recovered_points3d = pts3d_optimized
                        print(f"[INFO] Structure-only BA succeeded on {len(recovered_points3d)} points.")
                    except Exception as e:
                        print(f"[WARN] Structure-only BA failed: {e}")
                        print("[WARN] Proceeding with unoptimized triangulated points.")

                    t_ba_end = time.time()
                    t_ba = t_ba_end - t_ba_start
                    print(f"[TIME] Structure-only BA: {t_ba:.4f}s")

                t_merge_start = time.time()

                if len(recovered_points3d) > 0:
                    kf3dp = np.vstack([selected_points3d, recovered_points3d])
                    kf2dp = np.vstack([selected_points2d, recover_points2])
                    kf2dp_kf = np.vstack([selected_points2d_kf, recover_points1])
                    kf_accurate = selected_accurate + recover_accurate
                    kf_framehistory = selected_historyframe + recover_framehistory

                    print(f"[INFO] Added {len(recovered_points3d)} new triangulated points.")
                else:
                    print("[WARN] Triangulation produced no valid 3D points.")
                    kf3dp = selected_points3d
                    kf2dp = selected_points2d
                    kf2dp_kf = selected_points2d_kf
                    kf_accurate = selected_accurate
                    kf_framehistory = selected_historyframe

                t_merge_end = time.time()
                t_merge = t_merge_end - t_merge_start

            else:
                print("[INFO] Not enough unmatched points for recovery. Skipping triangulation.")
                kf3dp = selected_points3d
                kf2dp = selected_points2d
                kf2dp_kf = selected_points2d_kf
                kf_accurate = selected_accurate
                kf_framehistory = selected_historyframe

        else:
            kf3dp = selected_points3d
            kf2dp = selected_points2d
            kf2dp_kf = selected_points2d_kf
            kf_accurate = selected_accurate
            kf_framehistory = selected_historyframe

        t_vismatch_start = time.time()
        if self.use_gui:
            imgmatch = warp_corners_and_draw_matches(
                points1[idx_a], points2[idx_a], keyframe["image"]["img"], cframe
            )
            if len(self.shared_imgmatch) == 0:
                self.shared_imgmatch.append(imgmatch.tolist())
            else:
                self.shared_imgmatch[0] = imgmatch.tolist()

        t_vismatch_end = time.time()
        t_vismatch = t_vismatch_end - t_vismatch_start

        t_full_ba = 0
        if iskf:
            if optimized:
                t_full_ba_start = time.time()
                optimized_pose, optimized_pts3d = run_full_local_ba(
                    K=np2torch(self.K),
                    points2d=np2torch(kf2dp),
                    points3d=np2torch(kf3dp),
                    init_pose=np2torch(T_curr),
                    device=self.device,
                )
                t_full_ba_end = time.time()
                t_full_ba = t_full_ba_end - t_full_ba_start

                keyframe["pose"].append(optimized_pose)
                kf3dp = optimized_pts3d
            else:
                keyframe["pose"].append(T_curr)

        return keyframe, img, cfeats, T_curr, kf3dp, kf2dp, kf2dp_kf, kf_framehistory, kf_accurate, iskf

    def track_init(self, curframe, keyframe):
        import time

        t_prepare_start = time.time()
        kf = False
        match = dict()
        timestamp, img = self.dataset[curframe]
        img = resize_img(img, self.img_size)
        cframe = img["backend_img"].to(device=self.device)

        _, H, W = cframe.shape
        disp_threshold = ((H / self.cfg_tracker["init_kf_dis"]) * (W / self.cfg_tracker["init_kf_dis"])) ** 0.5
        t_prepare_end = time.time()
        t_prepare = t_prepare_end - t_prepare_start

        t_feat_start = time.time()
        cfeats = self.extractmatch.detectAndCompute(img["unnormalized_img"], top_k=self.matchtopk)[0]
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        t_feat_end = time.time()
        t_feat = t_feat_end - t_feat_start

        t_match_start = time.time()
        img = dict(img=cframe)

        idxs0, idxs1 = self.extractmatch.match(
            keyframe["features"]["descriptors"],
            cfeats["descriptors"],
            min_cossim=-1,
        )

        points1 = keyframe["features"]["keypoints"][idxs0].cpu().numpy()
        points2 = cfeats["keypoints"][idxs1].cpu().numpy()
        match["points"] = (points1, points2)
        match["timestamp"] = timestamp
        match["index"] = (keyframe["index"], curframe)
        keyframe["matches"].append(match)
        _, _, mean_disp, mean_std = compute_mean_std_filter(points1, points2)

        inlier_ratio = None

        F, mask = cv2.findFundamentalMat(
            points1,
            points2,
            method=self.ransac_method,
            ransacReprojThreshold=1.0,
            confidence=0.99,
        )
        if mask is not None:
            inlier_count = np.sum(mask)
            inlier_ratio = inlier_count / len(points1)

        kf = False
        if not inlier_ratio:
            kf = True
        if mean_disp > disp_threshold:
            kf = True
        if kf:
            if mean_disp < disp_threshold and len(points2) > 50:
                kf = False

        T_curr = np.eye(4)
        return keyframe, img, cfeats, T_curr, kf

    def run(self):
        while True:
            if not self.tracker_queue.empty():
                msg = self.tracker_queue.get()
                if msg[0] == "loadfinish":
                    print("[tracker] Load finished, continue.")
                    break
            time.sleep(0.01)

        initkeyframes = []
        cur_frame_idx = 0
        _, img = self.dataset[cur_frame_idx]

        img = resize_img(img, self.img_size)
        cframe = img["backend_img"]
        cfeats = self.extractmatch.detectAndCompute(img["unnormalized_img"], top_k=self.matchtopk)[0]
        keyframe = {
            "image": dict(img=cframe),
            "features": cfeats,
            "index": cur_frame_idx,
            "keypose": np.eye(4),
            "matches": [],
        }
        initkeyframes.append(keyframe)

        while len(initkeyframes) < self.cfg_tracker["init_frame_num"]:
            start_time = time.time()

            cur_frame_idx += 1
            keyframe, img, cfeats, T_curr, iskf = self.track_init(cur_frame_idx, keyframe)

            end_time = time.time()
            elapsed = end_time - start_time

            self.match_record.append((keyframe["matches"][-1], elapsed))

            if iskf:
                keyframe = {
                    "image": img,
                    "features": cfeats,
                    "index": cur_frame_idx,
                    "keypose": np.eye(4),
                    "matches": [],
                }
                initkeyframes.append(keyframe)

        init_indices = [i["index"] for i in initkeyframes]
        safe_initkeyframes = to_cpu_recursive(initkeyframes)
        safe_init_indices = to_cpu_recursive(init_indices)
        self.mapper_queue.put(["init", safe_initkeyframes, safe_init_indices])

        while True:
            if not self.tracker_queue.empty():
                msg = self.tracker_queue.get()
                if msg[0] == "initfinish":
                    select_kf_idx = msg[1]
                    points3d_init = msg[2]
                    inintpose = msg[3]
                    self.K = msg[4]
                    self.K = self.K.numpy()

                    print("[tracker] Received initfinish, continue.")
                    break
            time.sleep(0.01)

        points2d = cfeats["keypoints"].cpu().numpy()
        points3d = points3d_init[np.round(points2d[:, 1]).astype(int), np.round(points2d[:, 0]).astype(int)]
        keyframe = {
            "image": dict(img=cframe),
            "features": cfeats,
            "index": cur_frame_idx,
            "keypose": np.linalg.inv(inintpose),
            "points3d": points3d,
            "points2d": points2d,
            "matches": [],
            "pose": [],
            "frame_history_map": [
                [(cur_frame_idx, tuple(np.round(uv).astype(int)))]
                for uv in points2d
            ],
            "accuratemap": [True] * len(points2d),
            "mapped": True,
        }
        assert len(keyframe["image"]["img"].shape) == 3

        self.keyframes = []
        self.keyframes.append(keyframe)
        cur_frame_idx += 1
        sendpointmap = 0
        starttracking = 1
        total_coverkf = []
        kf_cluster = []

        while True:
            queue_size = self.mapper_queue.qsize()
            if queue_size > 0:
                time.sleep(0.01)
                continue

            if self.tracker_queue.empty():
                if starttracking:
                    start_track = time.time()

                    keyframe, img, cfeats, T_curr, points3d, points2d, kf2dp_kf, kf_framehistory, kf_accurate, iskf = self.track(cur_frame_idx, keyframe)
                    print(f"🖼️ frame[{cur_frame_idx}] ⏱️ time: {(time.time() - start_track) * 1000:.1f} ms\n")
                    self.match_record.append(keyframe["matches"][-1])

                    self.point3d[:] = points3d
                    self.currentpose[:] = np.linalg.inv(T_curr)

                    if iskf:
                        current_kfscore = []
                        if len(self.mapper_keyframes) > 5:
                            coverkfs, coverkfs_score = self.mapper_keyframes.check_projection_coverage(cur_frame_idx, points3d)
                            if coverkfs:
                                for kf, score in zip(coverkfs, coverkfs_score):
                                    current_kfscore.append((kf, score))

                        if self.loopindices.begin.value:
                            coverkfs_loop, coverkfs_loop_score = self.mapper_keyframes.check_projection_coverage_loop(
                                self.loopindices.loop_indices,
                                points3d,
                                min_coverage_ratio=0.1,
                            )
                            if coverkfs_loop:
                                for kf, score in zip(coverkfs_loop, coverkfs_loop_score):
                                    current_kfscore.append((kf, score))

                        total_coverkf += current_kfscore
                        kf_cluster = dedup_and_difference_clustering(total_coverkf)

                        keyframe = {
                            "image": img,
                            "features": cfeats,
                            "index": cur_frame_idx,
                            "keypose": T_curr,
                            "matches": [],
                            "pose": [],
                            "points3d": points3d,
                            "points2d": points2d,
                            "frame_history_map": kf_framehistory,
                            "accuratemap": kf_accurate,
                            "mapped": False,
                        }
                        self.keyframes.append(keyframe)

                        if current_kfscore:
                            top5_kf = get_top_k_frame_ids(current_kfscore)
                            top_extend = expand_and_include_extras(top5_kf)
                            top_extend = [i for i in top_extend if not self.mapper_keyframes.is_empty[i].item()]

                            min_pixel_std = 100
                            min_flow_index = -1
                            min_kpt = None
                            min_desc = None
                            min_points1 = None
                            min_points2 = None

                            for i in top_extend:
                                xfeats_kpt = unpad_tensor(self.mapper_keyframes.xfeat_kpts[i], device=self.device)
                                xfeat_desc = unpad_tensor(self.mapper_keyframes.xfeat_desc[i], device=self.device)
                                assert xfeat_desc.shape[0] == xfeat_desc.shape[0]
                                idxs0, idxs1 = self.extractmatch.match(keyframe["features"]["descriptors"], xfeat_desc, min_cossim=-1)
                                points1 = keyframe["features"]["keypoints"][idxs0].cpu().numpy()
                                points2 = xfeats_kpt[idxs1].cpu().numpy()
                                _, _, mean_disp, std_disp = compute_mean_std_filter(points1, points2)
                                if std_disp < min_pixel_std:
                                    min_pixel_std = std_disp
                                    min_flow_index = i
                                    min_kpt = xfeats_kpt
                                    min_desc = xfeat_desc
                                    min_points1 = points1
                                    min_points2 = points2

                            print(
                                f"🚩 Dataset Index: {self.mapper_keyframes.dataset_idx[min_flow_index].item()} | "
                                f"💨 Min Pixel Flow: {min_pixel_std:.4f}"
                            )

                            inlier_ratio = 0.0

                            if min_points1 is not None and min_points2 is not None:
                                if isinstance(min_points1, np.ndarray) and isinstance(min_points2, np.ndarray):
                                    if min_points1.shape[0] >= 8 and min_points2.shape[0] >= 8:
                                        try:
                                            F, mask = cv2.findFundamentalMat(
                                                min_points1,
                                                min_points2,
                                                method=self.ransac_method,
                                                ransacReprojThreshold=1.0,
                                                confidence=0.99,
                                            )
                                            if mask is not None:
                                                inlier_count = np.sum(mask)
                                                inlier_ratio = inlier_count / len(min_points1)
                                            else:
                                                print("[WARN] Fundamental matrix mask is None. Returning 0.")
                                        except cv2.error as e:
                                            print(f"[ERROR] OpenCV error: {e}. Returning 0.")
                                    else:
                                        print(f"[WARN] Not enough points: {min_points1.shape[0]}, {min_points2.shape[0]}")
                                else:
                                    print("[WARN] One of the inputs is not a valid numpy array.")
                            else:
                                print("[WARN] min_points1 or min_points2 is None.")

                            if inlier_ratio > self.mid_threshold:
                                self.loopindices.add_batch([min_flow_index], len(self.keyframes) - 1, source="map")

                                pose = self.mapper_keyframes.c2ws[min_flow_index].cpu().numpy()
                                pointpose = self.mapper_keyframes.pointpose[min_flow_index].cpu().numpy()
                                keypose = pose @ pointpose
                                feature = {
                                    "keypoints": min_kpt,
                                    "descriptors": min_desc,
                                }
                                index_now = self.mapper_keyframes.dataset_idx[min_flow_index].item()

                                points3d = self.mapper_keyframes.point[min_flow_index].cpu().numpy()
                                pts_shape = points3d.shape
                                points = points3d.reshape(-1, 3)
                                N = points.shape[0]
                                points_h = np.hstack([points, np.ones((N, 1))])
                                points_world_h = (pointpose @ points_h.T).T
                                points = points_world_h[:, :3]
                                points = points.reshape(pts_shape)
                                points2d = min_kpt.cpu().numpy()
                                tensor = self.mapper_keyframes.color[min_flow_index]

                                def color_tensor_to_img(tensor: torch.Tensor) -> "np.ndarray":
                                    img = tensor.detach().cpu().permute(2, 0, 1)
                                    return img

                                img = color_tensor_to_img(tensor)
                                keyframe = {
                                    "image": dict(img=img),
                                    "features": feature,
                                    "index": index_now,
                                    "keypose": np.linalg.inv(keypose),
                                    "matches": [],
                                    "pose": [],
                                    "points3d": points[np.round(points2d[:, 1]).astype(int), np.round(points2d[:, 0]).astype(int)],
                                    "points2d": points2d,
                                    "frame_history_map": [
                                        [(index_now, tuple(np.round(uv).astype(int)))]
                                        for uv in points2d
                                    ],
                                    "accuratemap": [True] * len(points2d),
                                    "mapped": True,
                                    "mapped_idx": min_flow_index,
                                }
                                self.keyframes[-1] = keyframe
                                print(
                                    f"\n🚨 🔁 ⚡ [change frame] cur_frame_idx {cur_frame_idx} => "
                                    f"replaced with index_now: {index_now} ⚡ 🔁 🚨\n"
                                )
                                kf_cluster = []
                            else:
                                kframeidx = [
                                    k["mapped_idx"]
                                    for k in self.keyframes
                                    if "mapped_idx" in k and k["mapped_idx"] is not None
                                ]

                                kf_cluster = filter_kf_clusters_exclusive(kf_cluster, kframeidx)

                map_stats = [k["mapped"] for k in self.keyframes]
                count_false = map_stats.count(False)

                if len(self.keyframes) > self.n_map or count_false > self.n_map - 1 or len(self.keyframes) + len(kf_cluster) > self.n_map:
                    kfl = [k["index"] for k in self.keyframes]

                    if False not in map_stats:
                        self.keyframes = self.keyframes[-self.overlap:]

                    kfidx = []
                    newkflist = []
                    for kf in self.keyframes:
                        if kf["index"] not in kfidx:
                            newkflist.append(kf)
                            kfidx.append(kf["index"])
                    self.keyframes = newkflist
                    kfl = [k["index"] for k in self.keyframes]

                    kf_cluster_filtered = []
                    for cluster in kf_cluster:
                        new_cluster = [x for x in cluster if self.mapper_keyframes.dataset_idx[x] not in kfl]
                        if new_cluster:
                            kf_cluster_filtered.append(new_cluster)

                    kf_cluster = kf_cluster_filtered

                    if len(self.keyframes) > 4 or count_false > 3 or len(self.keyframes) + len(kf_cluster) > 4:
                        self.update_stats = False

                        last_kf = [copy.deepcopy(kf) for kf in self.keyframes[-self.overlap:]]
                        for k in last_kf:
                            k["mapped"] = True

                        assert self.keyframes[0]["mapped"] is True

                        kf_send = [self.keyframes[0]]
                        for kf in self.keyframes[1:]:
                            if kf["mapped"] is False:
                                kf_send.append(kf)
                            else:
                                flat_set = [self.mapper_keyframes.dataset_idx[i] for cluster in kf_cluster for i in cluster]
                                flatgraph_set = [self.mapper_keyframes.graph_idx[i] for cluster in kf_cluster for i in cluster]

                                if kf["index"] not in flat_set and self.mapper_keyframes.graph_idx[kf["mapped_idx"]] not in flatgraph_set:
                                    kf_cluster.append([kf["mapped_idx"]])

                        kf_send = to_cpu_recursive(kf_send)
                        self.mapper_queue.put(["map", kf_send, kf_cluster])
                        kf_cluster = []
                        total_coverkf = []
                        self.keyframes = last_kf
                        sendpointmap += 1
                        for k in self.keyframes:
                            k["mapped"] = True

                elif map_stats == [True, False] and self.update_stats is False:
                    sendkeyframes = to_cpu_recursive(self.keyframes)
                    self.mapper_queue.put(["update_tracking", sendkeyframes])
                    self.update_stats = True

                cur_frame_idx += 1
                end_track = time.time()
                elapsed = end_track - start_track
                a = self.match_record[-1]
                b = elapsed
                self.match_record[-1] = (a, b)

                if cur_frame_idx >= len(self.dataset):
                    if keyframe["index"] != self.keyframes[-1]["index"]:
                        self.keyframes.append(keyframe)

                    if cur_frame_idx != keyframe["index"]:
                        img = img if isinstance(img, dict) else dict(img=img)

                        keyframe = {
                            "image": img,
                            "features": cfeats,
                            "index": cur_frame_idx,
                            "keypose": T_curr,
                            "matches": [],
                            "pose": [],
                            "points3d": points3d,
                            "points2d": points2d,
                            "frame_history_map": kf_framehistory,
                            "accuratemap": kf_accurate,
                            "mapped": False,
                        }
                        self.keyframes.append(keyframe)
                        map_stats = [k["mapped"] for k in self.keyframes]
                        if False in map_stats:
                            kf_send = [self.keyframes[0]]
                            for kf in self.keyframes[1:]:
                                if kf["mapped"] is False:
                                    kf_send.append(kf)
                                else:
                                    flat_set = [self.mapper_keyframes.dataset_idx[i] for cluster in kf_cluster for i in cluster]
                                    flatgraph_set = [self.mapper_keyframes.graph_idx[i] for cluster in kf_cluster for i in cluster]

                                    if kf["index"] not in flat_set and self.mapper_keyframes.graph_idx[kf["mapped_idx"]] not in flatgraph_set:
                                        kf_cluster.append([kf["mapped_idx"]])
                            kf_send = to_cpu_recursive(kf_send)
                            self.mapper_queue.put(["map", kf_send, kf_cluster])

                    time.sleep(10)
                    print("[tracker] Reached max frames, initiating shutdown.")
                    self.mapper_queue.put(["SHUTDOWN"])
                    start_time = time.time()
                    while time.time() - start_time < 10:
                        try:
                            data = self.tracker_queue.get(timeout=1)
                            if data[0] == "SHUTDOWN_ACK":
                                print("[tracker] Mapper confirmed shutdown.")
                                break
                        except queue.Empty:
                            print("[tracker] Waiting for mapper shutdown signal...")
                            continue
                    else:
                        print("[tracker] Timeout waiting for mapper shutdown.")
                        self.keyframes = to_cpu_recursive(self.keyframes)
                        self.mapper_queue.put(["map", self.keyframes, kf_cluster])
                    break
            else:
                data = self.tracker_queue.get()
                if data[0] == "map finished":
                    point, kf_pose, frame_idx = self.mapper_keyframes.get_latest_keyframe()

                    self.K = self.mapper_keyframes.K.numpy()

                    updated_ids = apply_dense_correction_from_backend(
                        keyframe=keyframe,
                        map3d=torch2np(point),
                        corrected_kf_id=frame_idx,
                        pose=np.linalg.inv(torch2np(kf_pose)),
                    )
                    print(f"\n📥 [Received from tracker]")
                    print(f" - Frame index: {frame_idx}")

                if data[0] == "update point":
                    point, kf_pose, frame_idx = data[1], data[2], data[3]
                    updated_ids = apply_dense_correction_from_backend(
                        keyframe=keyframe,
                        map3d=torch2np(point),
                        corrected_kf_id=frame_idx,
                        pose=np.linalg.inv(torch2np(kf_pose)),
                    )
                    print(f"\n📥 [Update from tracker]")
                    print(f" - Frame index: {frame_idx}")

                if data[0] == "pause":
                    print("[tracker] Pausing...")
                    time.sleep(10)

        return self.match_record