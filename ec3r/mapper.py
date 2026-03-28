import time
import torch.multiprocessing as mp
import open3d as o3d
import torch
import sys

sys.path.append("vggt/")
from vggtinfer import init_infer, mapinfer, encoder_infer
from vggt.models.vggt import VGGT

from pcl_registration import point_regist, scan_to_map_trans, scan_to_map_loop
from recoverpose import reestimate_pose, recover_pose_init, estimation_pose_from_point
from ec3r.utils.slam_utils import select_keyframes, pad_tensor_to, match
import numpy as np
from termcolor import colored
from ec3r.frame import transform_points_to_global
import queue
import cv2


def get_exclusive_indices(new_idx, match_indexs):
    match_set = set(match_indexs)
    return [i for i, val in enumerate(new_idx) if val not in match_set]


def list_get(lst, indices):
    return [lst[i] for i in indices]


def transform_point_tf(point, tf):
    """
    Transform 3D points from a local frame to a target frame using a 4x4 transform.

    Input:
        point: Tensor [..., 3] or list of Tensor
        tf: (4, 4) transform matrix

    Return:
        list of Tensor
    """
    if isinstance(point, list):
        return [transform_point_tf(p, tf)[0] for p in point]

    if not isinstance(point, torch.Tensor):
        point = torch.tensor(point, dtype=torch.float32, device=tf.device)

    original_shape = point.shape
    point3d = point.reshape(-1, 3)
    N = point3d.shape[0]

    point_h = torch.cat(
        [point3d, torch.ones(N, 1, device=point.device, dtype=point.dtype)], dim=1
    )
    point_transformed_h = (tf @ point_h.T).T
    point_transformed = point_transformed_h[:, :3].reshape(original_shape)

    return [point_transformed]


class FrameMapper(mp.Process):
    def __init__(self, config, keyframes, pgo, loopindices):
        super().__init__()
        self.config = config["mapping"]
        self.tracker_queue = None
        self.mapper_queue = None
        self.looper_queue = None
        self.K = None
        self.loopindices = loopindices

        self.device = "cuda:0"
        self.mainpcd = []
        self.c2ws = []
        self.pointmaps = []
        self.keyframes = keyframes
        self.pgo = pgo
        self.graphinfo = []
        self.submap_num = 1
        self.current_graph = 1

    def run(self):
        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

        model.eval()
        self.model = model.to(self.device)
        self.overlap = self.config["overlap"]
        self.model.eval()
        self.tracker_queue.put(["loadfinish"])
        self.feature_cache = {}

        while True:
            try:
                data = self.mapper_queue.get(timeout=1)

                if data[0] == "pause":
                    print("[mapper] Pausing...")
                    time.sleep(10)

                elif data[0] == "stop":
                    print("[mapper] Stopping...")
                    break

                elif data[0] == "init":
                    print("[mapper] Begin initialization")
                    imgdicts = []
                    keyframes = data[1]

                    cfeats_kpt = []
                    cfeats_desc = []
                    new_idx = []
                    for keyframe in keyframes:
                        print("[DEBUG] keyframe index:", keyframe["index"])
                        imgdicts.append(keyframe["image"])
                        new_idx.append(keyframe["index"])
                        cfeats_kpt.append(
                            pad_tensor_to(keyframe["features"]["keypoints"], device="cpu")
                        )
                        cfeats_desc.append(
                            pad_tensor_to(keyframe["features"]["descriptors"], device="cpu")
                        )

                    kfpose, selected_poins, selected_rgb, selected_confs, encoded_feats, K_pred = init_infer(
                        imgdicts,
                        self.model,
                        torch.device("cuda"),
                        dtype=torch.float32,
                    )

                    self.K = K_pred[-1].to("cpu")
                    self.keyframes.update_K(self.K)
                    dirty = [True] * len(new_idx)
                    ismapframe = [True] * len(new_idx)
                    graph_idx = [self.current_graph] * len(new_idx)

                    pointpose = [torch.eye(4)] * len(new_idx)
                    self.keyframes.append(
                        (
                            new_idx,
                            selected_rgb,
                            selected_poins,
                            selected_confs,
                            kfpose,
                            K_pred,
                            pointpose,
                            encoded_feats,
                            dirty,
                            ismapframe,
                            graph_idx,
                            cfeats_kpt,
                            cfeats_desc,
                        )
                    )
                    prior_info = torch.eye(7) * 1e6
                    self.current_graph += 1
                    self.tracker_queue.put(
                        [
                            "initfinish",
                            new_idx,
                            selected_poins[-1].cpu().numpy(),
                            kfpose[-1].cpu().numpy(),
                            self.K,
                        ]
                    )
                    self.looper_queue.put(["add_prior", 0, pointpose[-1], prior_info])

                elif data[0] == "update_tracking":
                    keyframes = data[1]
                    imgdicts = []
                    new_idx = []
                    cfeats_kpt = []
                    cfeats_desc = []

                    for keyframe in keyframes:
                        print("[DEBUG] keyframe index:", keyframe["index"])
                        imgdicts.append(keyframe["image"])
                        new_idx.append(keyframe["index"])
                        cfeats_kpt.append(
                            pad_tensor_to(keyframe["features"]["keypoints"], device="cpu")
                        )
                        cfeats_desc.append(
                            pad_tensor_to(keyframe["features"]["descriptors"], device="cpu")
                        )

                    old_feats, old_confs, old_points, old_graph_idx, old_idx, old_pointposes = \
                        self.keyframes.find_by_dataset_idx(new_idx)

                    old_pointpose = old_pointposes[-1]

                    new_points, new_rgb, new_confs, encoded_feats, kfpose, intrinsics = mapinfer(
                        imgdicts,
                        self.model,
                        torch.device("cuda"),
                        dtype=torch.float32,
                        feat_embeddings=old_feats,
                    )
                    torch.cuda.empty_cache()

                    self.K = K_pred[-1].to("cpu")
                    self.keyframes.update_K(self.K)
                    self.feature_cache["feats"] = encoded_feats[1]
                    self.feature_cache["new_idx"] = new_idx[1]

                    tf_new_to_old, match_idxs = scan_to_map_trans(
                        old_idx, old_points, old_confs, new_points, new_confs, new_idx, K=self.K
                    )
                    new_pointpose = old_pointpose @ tf_new_to_old

                    point_global, kf_pose_global = transform_points_to_global(
                        new_points[-1],
                        kfpose[-1].to(dtype=torch.float32),
                        new_pointpose,
                    )
                    frame_idx = new_idx[-1]
                    self.tracker_queue.put(["map finished", point_global, kf_pose_global, frame_idx])

                elif data[0] == "map":
                    new_encoded_feats = None
                    self.loopindices.add_batch([], len(self.keyframes) - 1, source="map")
                    keyframes = data[1]
                    kf_cluster = data[2]
                    print(keyframes[-1])
                    lastindex = keyframes[-1]["index"]

                    imgdicts = []
                    new_idx = []
                    cfeats_kpt = []
                    cfeats_desc = []

                    for keyframe in keyframes:
                        print("[DEBUG] keyframe index:", keyframe["index"])
                        assert isinstance(
                            keyframe["image"], dict
                        ), f"Expected dict, but got {type(keyframe['image'])}"
                        imgdicts.append(keyframe["image"])
                        new_idx.append(keyframe["index"])
                        cfeats_kpt.append(
                            pad_tensor_to(keyframe["features"]["keypoints"], device="cpu")
                        )
                        cfeats_desc.append(
                            pad_tensor_to(keyframe["features"]["descriptors"], device="cpu")
                        )

                    old_feats, old_confs, old_points, old_graph_idx, old_idx, old_pointposes = \
                        self.keyframes.find_by_dataset_idx(new_idx)

                    old_pointpose = old_pointposes[-1]

                    if new_idx[1] == self.feature_cache["new_idx"]:
                        old_feats.append(self.feature_cache["feats"])

                    if kf_cluster:
                        if new_idx[1] == self.feature_cache["new_idx"]:
                            if len(imgdicts) > 2:
                                new_encoded_feats = encoder_infer(
                                    imgdicts,
                                    self.model,
                                    torch.device("cuda"),
                                    dtype=torch.float32,
                                    feat_embeddings=old_feats,
                                )
                            else:
                                new_encoded_feats = None

                        start_time = time.time()
                        selected_history_keyframes = []

                        for cluster in kf_cluster:
                            best_kf = None
                            best_score = 0.0

                            for kf_id in cluster:
                                if self.keyframes.dataset_idx[kf_id] not in new_idx:
                                    xfeat_desc, xfeat_kpt = self.keyframes.find_features(kf_id)
                                    highest_ratio = 0.0

                                    for ckpt, cdesc in zip(cfeats_kpt, cfeats_desc):
                                        idxs0, idxs1 = match(xfeat_desc, cdesc, min_cossim=-1)

                                        if len(idxs0) < 8 or len(idxs1) < 8:
                                            continue

                                        points1 = xfeat_kpt[idxs0].cpu().numpy()
                                        points2 = ckpt[idxs1].cpu().numpy()

                                        F, mask = cv2.findFundamentalMat(
                                            points1,
                                            points2,
                                            method=cv2.USAC_MAGSAC,
                                            ransacReprojThreshold=1.0,
                                            confidence=0.99,
                                        )

                                        if mask is not None:
                                            inlier_count = np.sum(mask)
                                            total_matches = len(mask)
                                            inlier_ratio = inlier_count / total_matches if total_matches > 0 else 0.0

                                            if inlier_ratio > highest_ratio:
                                                highest_ratio = inlier_ratio

                                    status = "✅ valid" if highest_ratio > 0.25 else "❌ invalid"
                                    print(
                                        f"[Check] kf_id = {kf_id}, "
                                        f"inlier_ratio = {highest_ratio:.4f} -> {status}"
                                    )

                                    if highest_ratio > best_score and highest_ratio > 0.3:
                                        best_score = highest_ratio
                                        best_kf = kf_id

                            if best_kf is not None:
                                print(
                                    f"👉 Selected best keyframe in cluster: "
                                    f"kf_id = {best_kf}, score = {best_score:.4f}"
                                )
                                selected_history_keyframes.append(best_kf)

                        if selected_history_keyframes:
                            self.loopindices.add_batch(
                                selected_history_keyframes,
                                len(self.keyframes) - 1,
                                source="map",
                            )

                            history_feats, history_confs, history_points, history_color, history_graph_idx, \
                            history_dataset_indices, history_pointpose, history_xfeat_desc, history_xfeat_kpt = \
                                self.keyframes.find_oldkf_idx(list(set(selected_history_keyframes)))

                            combined_feats = history_feats + old_feats
                            if new_encoded_feats is not None:
                                combined_feats += list(new_encoded_feats)

                            combined_confs = history_confs + old_confs
                            combined_points = history_points + old_points
                            combined_graph_idx = history_graph_idx + old_graph_idx
                            combined_dataset_indices = history_dataset_indices + old_idx

                            print("len(imgdicts)", len(imgdicts))
                            print("combined_dataset_indices", combined_dataset_indices)

                            combined_imgdicts = [dict(img=torch.zeros_like(imgdicts[0]["img"]))] * len(history_color) + imgdicts

                            new_points, new_rgb, new_confs, encoded_feats, kfpose, K_pred = mapinfer(
                                combined_imgdicts,
                                self.model,
                                torch.device("cuda"),
                                dtype=torch.float32,
                                feat_embeddings=combined_feats,
                            )

                            new_idx = history_dataset_indices + new_idx
                            new_feat_desc = history_xfeat_desc + cfeats_desc
                            new_feat_kpt = history_xfeat_kpt + cfeats_kpt

                            tf_new_to_old, _ = scan_to_map_trans(
                                old_idx, old_points, old_confs, new_points, new_confs, new_idx, K=self.K
                            )
                            new_pointpose = old_pointpose @ tf_new_to_old

                            history_graph_idx = np.array(history_graph_idx)
                            unique_graph_ids = np.unique(history_graph_idx)
                            tf_new_to_history_list = []

                            for gid in unique_graph_ids:
                                indices = np.where(history_graph_idx == gid)[0]

                                selected_dataset_indices = [history_dataset_indices[i] for i in indices]
                                selected_points = [history_points[i] for i in indices]
                                selected_confs = [history_confs[i] for i in indices]

                                tf, _ = scan_to_map_trans(
                                    selected_dataset_indices,
                                    selected_points,
                                    selected_confs,
                                    new_points,
                                    new_confs,
                                    new_idx,
                                    K=self.K,
                                )
                                tf_new_to_history_list.append((gid, tf.to("cpu")))

                            dirty = [True] * len(new_idx)
                            ismapframe = [False] * len(combined_dataset_indices) + [True] * (
                                len(new_idx) - len(combined_dataset_indices)
                            )
                            pointpose = [new_pointpose] * len(new_idx)
                            graph_idx = [self.keyframes.get_max_graph_idx() + 1] * len(new_idx)

                            assert len(new_idx) == len(new_rgb) == len(new_points) == len(new_confs) == \
                                len(kfpose) == len(encoded_feats) == len(dirty) == \
                                len(ismapframe) == len(graph_idx), \
                                f"Length mismatch: idx={len(new_idx)}, rgb={len(new_rgb)}, " \
                                f"point={len(new_points)}, conf={len(new_confs)}, " \
                                f"pose={len(kfpose)}, feat={len(encoded_feats)}, " \
                                f"pointpose={len(pointpose)}, dirty={len(dirty)}, " \
                                f"ismap={len(ismapframe)}, graph={len(graph_idx)}"

                            self.K = K_pred[-1].to("cpu")
                            self.keyframes.update_K(self.K)

                            self.keyframes.append(
                                (
                                    new_idx,
                                    new_rgb,
                                    new_points,
                                    new_confs,
                                    kfpose,
                                    K_pred,
                                    pointpose,
                                    encoded_feats,
                                    dirty,
                                    ismapframe,
                                    graph_idx,
                                    new_feat_kpt,
                                    new_feat_desc,
                                )
                            )
                            self.current_graph += 1

                            self.looper_queue.put(
                                [
                                    "add_odometry_factor",
                                    combined_graph_idx[-1],
                                    graph_idx[-1],
                                    tf_new_to_old.to("cpu"),
                                    new_pointpose.to("cpu"),
                                ]
                            )
                            self.looper_queue.put(
                                [
                                    "add_local_loop_factor",
                                    graph_idx[-1],
                                    tf_new_to_history_list,
                                    torch.eye(7),
                                ]
                            )

                        else:
                            combined_feats = old_feats
                            if new_encoded_feats is not None:
                                combined_feats += list(new_encoded_feats)

                            combined_confs = old_confs
                            combined_points = old_points
                            combined_graph_idx = old_graph_idx
                            combined_dataset_indices = old_idx

                            new_points, new_rgb, new_confs, encoded_feats, kfpose, K_pred = mapinfer(
                                imgdicts,
                                self.model,
                                torch.device("cuda"),
                                dtype=torch.float32,
                                feat_embeddings=combined_feats,
                            )

                            tf_new_to_old, _ = scan_to_map_trans(
                                combined_dataset_indices,
                                combined_points,
                                combined_confs,
                                new_points,
                                new_confs,
                                new_idx,
                                K=self.K,
                            )
                            new_pointpose = old_pointpose @ tf_new_to_old

                            pointpose = [new_pointpose] * len(new_idx)

                            torch.cuda.empty_cache()
                            dirty = [True] * len(new_idx)
                            ismapframe = [False] * len(combined_dataset_indices) + [True] * (
                                len(new_idx) - len(combined_dataset_indices)
                            )
                            self.K = K_pred[-1].to("cpu")
                            self.keyframes.update_K(self.K)
                            graph_idx = [self.keyframes.get_max_graph_idx() + 1] * len(new_idx)

                            assert len(new_idx) == len(new_rgb) == len(new_points) == len(new_confs) == \
                                len(kfpose) == len(encoded_feats) == len(dirty) == \
                                len(ismapframe) == len(graph_idx), \
                                f"Length mismatch: idx={len(new_idx)}, rgb={len(new_rgb)}, " \
                                f"point={len(new_points)}, conf={len(new_confs)}, " \
                                f"pose={len(kfpose)}, feat={len(encoded_feats)}, " \
                                f"pointpose={len(pointpose)}, dirty={len(dirty)}, " \
                                f"ismap={len(ismapframe)}, graph={len(graph_idx)}"

                            self.keyframes.append(
                                (
                                    new_idx,
                                    new_rgb,
                                    new_points,
                                    new_confs,
                                    kfpose,
                                    K_pred,
                                    pointpose,
                                    encoded_feats,
                                    dirty,
                                    ismapframe,
                                    graph_idx,
                                    cfeats_kpt,
                                    cfeats_desc,
                                )
                            )
                            self.current_graph += 1

                            self.looper_queue.put(
                                [
                                    "add_odometry_factor",
                                    combined_graph_idx[-1],
                                    graph_idx[-1],
                                    tf_new_to_old.to("cpu"),
                                    new_pointpose.to("cpu"),
                                ]
                            )

                    else:
                        def print_step_time(step_name, start_time, color="green"):
                            color_dict = {
                                "green": "\033[92m",
                                "cyan": "\033[96m",
                                "magenta": "\033[95m",
                                "yellow": "\033[93m",
                                "red": "\033[91m",
                            }
                            end_time = time.time()
                            print(f"{color_dict.get(color, '')}🕒 {step_name}: {end_time - start_time:.4f} s\033[0m")

                            print(f"{color_dict.get(color, '')}🔍 GPU memory status - {step_name}\033[0m")
                            print(f"    🔹 Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                            print(f"    🔹 Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
                            print(f"    🔹 Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
                            print(f"    🔹 Max reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
                            print("-" * 60)

                        start_time = time.time()
                        print("oldfeat", old_feats[0].shape)

                        new_points, new_rgb, new_confs, encoded_feats, kfpose, K_pred = mapinfer(
                            imgdicts,
                            self.model,
                            torch.device("cuda"),
                            dtype=torch.float32,
                            feat_embeddings=old_feats,
                        )

                        torch.cuda.empty_cache()

                        start_time = time.time()
                        tf_new_to_old, match_idxs = scan_to_map_trans(
                            old_idx, old_points, old_confs, new_points, new_confs, new_idx, K=self.K
                        )
                        new_pointpose = old_pointpose @ tf_new_to_old

                        torch.cuda.empty_cache()

                        start_time = time.time()

                        torch.cuda.empty_cache()

                        dirty = [True] * len(new_idx)
                        ismapframe = [False] * len(old_idx) + [True] * (len(new_idx) - len(old_idx))
                        pointpose = [new_pointpose] * len(new_idx)

                        self.K = K_pred[-1].to("cpu")
                        self.keyframes.update_K(self.K)
                        graph_idx = [self.keyframes.get_max_graph_idx() + 1] * len(new_idx)
                        self.keyframes.append(
                            (
                                new_idx,
                                new_rgb,
                                new_points,
                                new_confs,
                                kfpose,
                                K_pred,
                                pointpose,
                                encoded_feats,
                                dirty,
                                ismapframe,
                                graph_idx,
                                cfeats_kpt,
                                cfeats_desc,
                            )
                        )
                        self.current_graph += 1
                        self.looper_queue.put(
                            [
                                "add_odometry_factor",
                                old_graph_idx[-1],
                                graph_idx[-1],
                                tf_new_to_old.to("cpu"),
                                new_pointpose.to("cpu"),
                            ]
                        )
                        self.tracker_queue.put(["map finished"])

                elif data[0] == "loop_new":
                    loop_mapindex = data[1]
                    self.current_graph += 1

                    num_item = len(loop_mapindex)

                    loop_feats, loop_confs, loop_points, loop_color, loop_graph_idx, \
                    loop_dataset_indices, loop_pointpose, loop_xfeat_desc, loop_xfeat_kpt = \
                        self.keyframes.find_oldkf_idx(loop_mapindex)

                    shapes = tuple(torch.tensor([loop_color[0].shape[:2]]) for _ in range(num_item))
                    loop_imgdicts = [dict(img=torch.zeros_like(imgdicts[0]["img"]))] * num_item

                    assert len(loop_imgdicts) == len(loop_feats)
                    new_points, new_rgb, new_confs, encoded_feats, kfpose, K_pred = mapinfer(
                        loop_imgdicts,
                        self.model,
                        torch.device("cuda"),
                        dtype=torch.float32,
                        feat_embeddings=loop_feats,
                    )

                    tfs = []
                    for gid, loop_point, loop_conf, new_point, new_conf in zip(
                        loop_graph_idx, loop_points, loop_confs, new_points, new_confs
                    ):
                        tf = scan_to_map_loop(loop_point, loop_conf, new_point, new_conf)
                        tfs.append((gid, tf))

                    new_pointpose = loop_pointpose[0] @ tfs[0][1]
                    graph_idx = [self.keyframes.get_max_graph_idx() + 1] * num_item

                    dirty = [False] * num_item
                    ismapframe = [False] * num_item
                    pointpose = [new_pointpose] * num_item

                    self.keyframes.append(
                        (
                            loop_dataset_indices,
                            loop_color,
                            new_points,
                            new_confs,
                            kfpose,
                            K_pred,
                            pointpose,
                            loop_feats,
                            dirty,
                            ismapframe,
                            graph_idx,
                            loop_xfeat_kpt,
                            loop_xfeat_desc,
                        )
                    )
                    self.looper_queue.put(
                        [
                            "add_global_loop_factor_new",
                            graph_idx[-1],
                            new_pointpose.to("cpu"),
                            tfs,
                            loop_mapindex,
                        ]
                    )
                    time.sleep(10)
                    self.tracker_queue.put(["pause"])

                elif data[0] == "SHUTDOWN":
                    print("[mapper] Received shutdown signal, sending it to looper first.")

                    self.looper_queue.put(["SHUTDOWN"])
                    print("[mapper] Sent SHUTDOWN to looper, waiting for ACK...")

                    while True:
                        if not self.looper_queue.empty():
                            looper_response = self.looper_queue.get()
                            if looper_response[0] == "SHUTDOWN_ACK":
                                print("[mapper] Received SHUTDOWN_ACK from looper.")
                                break
                        time.sleep(2)
                        self.looper_queue.put(["SHUTDOWN"])

                    self.tracker_queue.put(["SHUTDOWN_ACK"])
                    print("[mapper] Sent SHUTDOWN_ACK to tracker. Shutdown complete.")

            except queue.Empty:
                print("[mapper] 🧊 EMPTY:")
                pass