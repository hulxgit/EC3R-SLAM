import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppost
import pypose as pp
from torch import nn
from torch.optim import Adam
import torch
import numpy as np
from pypose.optim.scheduler import StopOnPlateau
from scipy.spatial.transform import Rotation as R
from ec3r.utils.slam_utils import torch2np,np2torch
def compare_poses_rmse(poses1, poses2):
    """
    计算两条轨迹（两个 pose 列表）之间的整体平移和旋转 RMSE 误差
    参数:
        poses1, poses2: list of (4,4) torch tensors
    返回:
        trans_rmse: 平移 RMSE (m)
        rot_rmse: 旋转 RMSE (°)
    """
    # 1️⃣ 转换为 numpy
    poses1_np = [p.cpu().numpy() for p in poses1]
    poses2_np = [p.cpu().numpy() for p in poses2]
    
    # 2️⃣ 遍历，计算对应位姿的相对变换
    trans_errors = []
    rot_errors = []
    for T1, T2 in zip(poses1_np, poses2_np):
        T1_inv = np.linalg.inv(T1)
        T_rel = T1_inv @ T2
        
        # 平移误差
        trans = T_rel[:3, 3]
        trans_err = np.linalg.norm(trans)
        trans_errors.append(trans_err)
        
        # 旋转误差
        rot_mat = T_rel[:3, :3]
        rot = R.from_matrix(rot_mat)
        rot_err = rot.magnitude() * (180.0 / np.pi)  # 转度数
        rot_errors.append(rot_err)
    
    # 3️⃣ 计算整体 RMSE
    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    
    trans_rmse = np.sqrt(np.mean(trans_errors**2))
    rot_rmse = np.sqrt(np.mean(rot_errors**2))
    
    return trans_rmse, rot_rmse

class PoseGraph(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = pp.Parameter(nodes)

    def forward(self, edges, poses):
        node1 = self.nodes[edges[..., 0]]
        node2 = self.nodes[edges[..., 1]]
        error = poses.Inv() @ node1.Inv() @ node2
        return error.Log().tensor()

class PoseGraphManager:
    def __init__(self,keyframes, device='cpu'):
        self.keyframes = keyframes
        self.device = device
        self.nodes = []
        self.edges = []
        self.edge_poses = []
        self.edge_infos = []

    def add_frame_node(self, pose):
        pose = pose.squeeze(0)
        pose = pp.mat2Sim3(pose)
        self.nodes.append(pose.to(self.device))

    def delete_node_edge(self, node_idx):
        """
        删除与 node_idx 相关的所有边（以及 edge_poses, edge_infos）。
        节点本身不删除，只是删除与其相关的约束。
        """
        # 找出要删除的边的索引
        delete_indices = []
        for i, edge in enumerate(self.edges):
            if node_idx in edge.tolist():
                delete_indices.append(i)

        # 倒序删除（避免索引错乱）
        for i in sorted(delete_indices, reverse=True):
            self.edges.pop(i)
            self.edge_poses.pop(i)
            self.edge_infos.pop(i)        

    def add_pose_prior(self, node_idx, prior_pose, info=None):
        # Add a virtual edge from a fixed identity node to `node_idx`
        identity = pp.mat2Sim3(np.eye(4)).to(torch.float32)
        prior_pose = pp.mat2Sim3(prior_pose)
        self.nodes.insert(0, identity)
        self.nodes.append(prior_pose)

        self.edges.append(torch.tensor([0, node_idx + 1], dtype=torch.int64).to(self.device))

        self.edge_poses.append(pp.mat2Sim3(np.eye(4)).to(torch.float32))
        self.edge_infos.append(info if info is not None else torch.eye(7, device=self.device))

    def add_odometry_factor(self, idx1, idx2, relative_pose, info=None):
        # 创建新的边张量（确保顺序一致）
        new_edge = torch.tensor([idx1, idx2], dtype=torch.int64).to(self.device)

        # 查找是否已有相同边（按内容匹配）
        found = False
        for i, edge in enumerate(self.edges):
            if torch.equal(edge, new_edge):
                # 如果找到已有的边，就更新对应的 pose 和 info
                relative_posepp = pp.mat2Sim3(relative_pose.squeeze(0)).to(self.device)
                self.edge_poses[i] = relative_posepp
                self.edge_infos[i] = info if info is not None else torch.eye(7, device=self.device)
                found = True
                break

        # 如果没有找到，就新增这条边
        if not found:
            self.edges.append(new_edge)
            relative_posepp = pp.mat2Sim3(relative_pose.squeeze(0)).to(self.device)
            self.edge_poses.append(relative_posepp)
            self.edge_infos.append(info if info is not None else torch.eye(7, device=self.device))
    def delete_factor(self, idx1, idx2):
        """
        删除匹配 idx1 和 idx2 的边（以及对应的 edge_poses, edge_infos）。
        """
        delete_indices = []
        for i, edge in enumerate(self.edges):
            if (edge[0].item() == idx1 and edge[1].item() == idx2) or \
            (edge[0].item() == idx2 and edge[1].item() == idx1):
                delete_indices.append(i)

        # 倒序删除，避免索引错乱
        for i in sorted(delete_indices, reverse=True):
            self.edges.pop(i)
            self.edge_poses.pop(i)
            self.edge_infos.pop(i)        
    def add_loop_factor(self, idx1, idx2, relative_pose, info=None):
        self.add_odometry_factor(idx1, idx2, relative_pose, info)

    def optimize_pose_graph(self, max_iters=15, radius=1e4, tol=1e-6):
        import copy
        node_before = copy.deepcopy(self.nodes)  # 深拷贝，确保安全        nodes = torch.stack(self.nodes).to(self.device)

        nodes = torch.stack(self.nodes).to(self.device)
        
        info = torch.stack(self.edge_infos).to(self.device)
        edges = torch.stack(self.edges).to(self.device)
        poses = torch.stack(self.edge_poses).to(torch.float32).to(self.device)
       
        model = PoseGraph(nodes).to(self.device)
        solver = ppos.Cholesky()
        strategy = ppost.TrustRegion(radius=radius)

        optimizer = pp.optim.LM(model, solver=solver, strategy=strategy,
                                min=tol, vectorize=True)
        scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True)

        while scheduler.continual():
            loss = optimizer.step(input=(edges, poses), weight=info)
            scheduler.step(loss)

            title = 'PyPose PGO at the %d step(s) with loss %7f'%(scheduler.steps, loss.item())
            # plot_and_save(graph.nodes.translation(), name+'.png', title, axlim=axlim)

        ### The 2nd implementation: equivalent to the 1st one, but more compact
        scheduler.optimize(input=(edges, poses), weight=info)

        final_pose = []
        opt_nodes = model.state_dict()['nodes']  # shape: [N_opt, 7]
        for i in range(opt_nodes.shape[0]):
            final_pose.append(pp.matrix(opt_nodes[i]))

        posebefore = []
        for i in range(opt_nodes.shape[0]):
            posebefore.append(pp.matrix(node_before[i])) 



        trans_rmse, rot_rmse = compare_poses_rmse(final_pose, posebefore)
        print(f"轨迹整体平移RMSE: {trans_rmse:.6f} m")
        print(f"轨迹整体旋转RMSE: {rot_rmse:.6f} °")
        if  loss.item()<1:
            return final_pose
        else:
            self.nodes  =  node_before#dont change it if loop fall
            return None
    def print_graph(self):
        print("\nPose Graph Summary:")
        print(f"- Number of nodes       : {len(self.nodes)}")
        print(f"- Number of edges       : {len(self.edges)}")
        print(f"- Number of edge poses  : {len(self.edge_poses)}")
        print(f"- Number of infos       : {len(self.edge_infos)}")
        print("\nEdges:")
        for i, edge in enumerate(self.edges):
            print(f"🧩 Edge {i}: {edge.tolist()} ")


    def print_edge_losses(self):
        model = PoseGraph(torch.stack(self.nodes).to(self.device)).to(self.device)
        edges = torch.stack(self.edges).to(self.device)
        poses = torch.stack(self.edge_poses).to(torch.float32).to(self.device)
        infos = torch.stack(self.edge_infos).to(self.device)

        # 计算每条边的残差向量 (N_edges, 8)
        error_vectors = model(edges, poses)  # shape: (N_edges, 8)

        for i in range(len(self.edges)):
            e = error_vectors[i]  # (8,)
            info = infos[i]       # (7,7) or (8,8) 取决于你的定义
            # 如果信息矩阵不是 8x8，需要 pad/截取
            if info.shape[0] != e.shape[0]:
                # 这里假设 info 是 7x7，e 是 8，pad 一个 0 信息维度
                info_padded = torch.eye(e.shape[0], device=self.device, dtype=info.dtype)
                info_padded[:info.shape[0], :info.shape[1]] = info
                info = info_padded
            # loss = e^T * info * e
            loss = e @ info @ e
            print(f"Edge {i} ({self.edges[i].tolist()}): loss={loss.item():.6f}")