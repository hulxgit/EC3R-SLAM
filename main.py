import torch
import sys
sys.path.append('./accelerated_features')
sys.path.append('./fast3r')

from ec3r.config import load_config, config
import numpy as np
import time
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from ec3r.dataset import load_dataset
from multiprocessing import Manager
import re
import open3d as o3d
import torch.multiprocessing as mp
from ec3r.tracker import FrameTracker
from ec3r.mapper import FrameMapper
from ec3r.looper import FrameLooper
from ec3r.pgo import PoseGraphManager
from ec3r.utils.multiprocessing_utils import FakeQueue
from ec3r.visualization2 import fast_visualization
from ec3r.frame import SharedKeyframes
from ec3r.loopindices import LoopIndicesManager
from save_file import save_keyframes_pointcloud,recover_trajectory,savekf_data
class FastSLAM:
    def __init__(self,config,save_dir=None,sourcepath=None,use_gui = False):
        self.save_dir =save_dir
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        mp.set_start_method("spawn", force=True)  
        start.record()
        self.config = config
        self.sourcepath = sourcepath
        pattern = re.compile(r"(7-scenes|eth3d|euroc|tum|Replica)")  
        match = pattern.search(sourcepath)
        if match:
                dataset_name = match.group(1)
        else:
            dataset_name = None
        self.dataset =load_dataset(
            dataset_name = dataset_name,args=  None,path = self.sourcepath,config =  self.config
            )
        self.use_gui = use_gui
        h,w = self.dataset.get_img_shape()[0]
        print("hw",h,w)
        tracker_queue = mp.Queue()
        mapper_queue = mp.Queue()
        looper_queue = mp.Queue()


        manager = Manager()
        shared_pcd_list = manager.list()
        shared_c2w_list = manager.list()
        shared_imgmatch = manager.list()
        shared_points3d = manager.list()
        currentpose = manager.list(np.eye(4).flatten().tolist())  # 正确：长度为16的list

        self.keyframes = SharedKeyframes(manager,h,w)
        self.pgo = PoseGraphManager(self.keyframes)
        self.loopindices = LoopIndicesManager(manager)


        self.tracker = FrameTracker(self.config,shared_imgmatch,shared_points3d,currentpose,self.keyframes,self.loopindices)
        self.mapper = FrameMapper(self.config,self.keyframes,self.pgo,self.loopindices)
        self.looper = FrameLooper(self.config,self.keyframes,self.pgo,self.loopindices)

        self.tracker.dataset = self.dataset
        self.tracker.use_gui = self.use_gui
        # self.tracker.K =  self.dataset.camera_intrinsics.K_frame
        # self.mapper.K =  self.dataset.camera_intrinsics.K_frame
        # self.looper.K =  self.dataset.camera_intrinsics.K_frame

        self.tracker.tracker_queue = tracker_queue
        self.tracker.mapper_queue = mapper_queue

        self.mapper.tracker_queue = tracker_queue
        self.mapper.looper_queue = looper_queue
        self.mapper.mapper_queue = mapper_queue

        self.looper.looper_queue = looper_queue
        self.looper.mapper_queue = mapper_queue
        self.looper.tracker_queue = tracker_queue

        self.mapper.mainpcd = shared_pcd_list
        self.mapper.c2ws = shared_c2w_list
        # self.tracker.imgmatch = shared_imgmatch
        time.sleep(0.01)
        if self.use_gui:
            self.visualizer = mp.Process(
                target=fast_visualization,
                args=(self.keyframes,self.tracker.shared_imgmatch,self.tracker.point3d,self.tracker.currentpose,)
            )
            self.visualizer.start()

        
        self.looper.start()
        self.mapper.start()
        match_record = self.tracker.run()
        mapper_queue.put(["pause"])
        mapper_queue.put(["stop"])
        looper_queue.put(["stop"])
        self.mapper.join()
        self.looper.join()
        starttime,_ = self.dataset[0] 
        save_keyframes_pointcloud(self.save_dir, self.keyframes, conf_percentile=30)
        recover_trajectory(self.save_dir, self.keyframes,match_record)


        end.record()
        torch.cuda.synchronize() 
        if self.use_gui:
            print("[Main] Terminating Visualizer...")
            self.visualizer.terminate()   # 强制结束
            self.visualizer.join()        # 等待退出
            print("[Main] Visualizer successfully terminated.")
    def run(self):
        pass
    
    


            




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--sourcepath", type=str, required=True, help="Path to source files")
    parser.add_argument("--save_dir", type=str, default="./results", help="Path to save directory (default: ./results)")
    parser.add_argument("--use_gui", action="store_true", help="Enable GUI visualization (default: False)")
    
    args = parser.parse_args(sys.argv[1:])
    load_config(args.config)
    slam = FastSLAM(config, save_dir=args.save_dir, sourcepath=args.sourcepath,use_gui=args.use_gui)
    slam.run()