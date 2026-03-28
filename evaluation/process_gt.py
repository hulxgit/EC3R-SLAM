import os
from os import path as osp
import numpy as np
import sys
from tqdm import tqdm

SLAM3R_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, SLAM3R_DIR) # noqa: E402

from slam3r.datasets import Replica


def get_replica_gt_pcd(scene_id, save_dir, sample_stride=20):
    os.makedirs(save_dir, exist_ok=True)
    H, W = 224, 224
    dataset = Replica(resolution=(W,H), scene_name=scene_id, num_views=1, sample_freq=sample_stride,ROOT = "/home/lingxianghu/datasets/Replica/")
    print(dataset[0][0]['pts3d'].shape)
    all_pcd = np.zeros([len(dataset),H,W,3])
    valid_masks = np.ones([len(dataset),H,W], dtype=bool)
    for id in tqdm(range(len(dataset))):
        view = dataset[id][0]
        pcd =view['pts3d']
        valid_masks[id] = view['valid_mask']
        all_pcd[id] = pcd

    np.save(os.path.join(save_dir, f"{scene_id}_pcds.npy"), all_pcd)
    np.save(os.path.join(save_dir, f"{scene_id}_valid_masks.npy"), valid_masks)    
    


if __name__ == "__main__":
    for scene_id in ['office0', 'office1', 'office2', 'office3', 'office4', 'room0', 'room1', 'room2']:
        get_replica_gt_pcd(scene_id, sample_stride=1, save_dir="results/gt/replica")