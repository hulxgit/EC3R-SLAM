import torch
import pypose as pp
from torch import nn
from pypose.optim import LM
from pypose.optim.kernel import Huber
from pypose.optim.solver import Cholesky
from pypose.optim.strategy import TrustRegion
from pypose.optim.corrector import FastTriggs
from pypose.function.geometry import pixel2point, reprojerr
from pypose.optim.scheduler import StopOnPlateau
from ec3r.utils.slam_utils import torch2np,np2torch
class FullLocalBA(nn.Module):
    def __init__(self, K, pts2d, pts3d, init_T):
        super().__init__()
        self.register_buffer("K", K)
        self.register_buffer("pts2d", pts2d)   # N x 2 2D点观测

        self.T = pp.Parameter(init_T)          # 优化的相机位姿
        self.pts3d = nn.Parameter(pts3d)        # 优化的地图点 (结构)

    def forward(self):
        return  reprojerr(self.pts3d, self.pts2d, self.K, self.T, reduction='none')

def run_full_local_ba(K, points2d, points3d, init_pose, device='cpu'):
    """
    Run Local BA: optimize both pose and structure
    """
    if isinstance(init_pose, torch.Tensor):
        init_pose = pp.mat2SE3(init_pose)
    points2d = points2d.to(device)
    points3d = points3d.to(device)
    K = K.to(device)

    graph = FullLocalBA(K, points2d, points3d, init_pose).to(device)

    kernel = Huber(delta=1.0)
    corrector = FastTriggs(kernel)
    optimizer = LM(graph, solver=Cholesky(),
                          strategy=TrustRegion(radius=1e3),
                          kernel=kernel,
                          corrector=corrector,
                          min=1e-8,
                          reject=128,
                          vectorize=True)
    scheduler = StopOnPlateau(optimizer, steps=25,
                                            patience=4,
                                            decreasing=1e-6,
                                            verbose=True)
    max_steps = 10  # 优化步数可以多一点
    for _ in range(max_steps):
        loss = optimizer.step(input=())
        scheduler.step(loss)
    optimized_pose = pp.SE3(graph.T.data.detach())
    optimized_pts3d = graph.pts3d.data.detach()

    return torch2np(optimized_pose), torch2np(optimized_pts3d)



class StructureOnlyBA(nn.Module):
    def __init__(self, K, pts2d, pts3d, fixed_T):
        super().__init__()
        self.register_buffer("K", K)
        self.register_buffer("pts2d", pts2d)
        self.register_buffer("T", fixed_T)             # 固定相机位姿
        self.pts3d = nn.Parameter(pts3d)               # 优化的结构点

    def forward(self):
        if self.pts3d.shape[0] == 0:
            return torch.tensor([], device=self.pts3d.device)
        return pp.reprojerr(self.pts3d, self.pts2d, self.K, self.T, reduction='none')


def run_structure_only_ba(
    K,
    points2d,
    points3d,
    fixed_pose,
    device='cpu',
    max_steps=10,
    stop_loss_thresh=1e-4
):
    """
    运行结构优化（固定 pose，仅优化 3D 点）

    参数：
        K: 相机内参（tensor）
        points2d: (N,2) tensor
        points3d: (N,3) tensor
        fixed_pose: SE3 pose (LieTensor or 4x4)
        device: 'cpu' or 'cuda'
        max_steps: 最大优化迭代次数
        stop_loss_thresh: 提前终止阈值（loss < 此值则停止）

    返回：
        optimized_pts3d: np.ndarray (N, 3)
    """
    # 预检查
    if points3d.shape[0] == 0 or points2d.shape[0] == 0:
        print("⚠️ [BA] 输入为空，跳过结构优化")
        return torch2np(points3d)

    if torch.isnan(points3d).any() or torch.isnan(points2d).any():
        print("❌ [BA] 输入包含 NaN，跳过结构优化")
        return torch2np(points3d)

    if isinstance(fixed_pose, torch.Tensor):
        fixed_pose = pp.mat2SE3(fixed_pose)

    # 移动到设备
    points2d = points2d.to(device)
    points3d = points3d.to(device)
    K = K.to(device)

    graph = StructureOnlyBA(K, points2d, points3d, fixed_pose).to(device)

    # 设置优化器
    kernel = Huber(delta=1.0)
    corrector = FastTriggs(kernel)
    optimizer = LM(
        graph,
        solver=Cholesky(),
        strategy=TrustRegion(radius=1e3),
        kernel=kernel,
        corrector=corrector,
        min=1e-8,
        reject=128,
        vectorize=True,
    )
    scheduler = StopOnPlateau(
        optimizer,
        steps=25,
        patience=4,
        decreasing=1e-6,
        verbose=False
    )

    # 优化循环
    for i in range(max_steps):
        loss = optimizer.step(input=())
        scheduler.step(loss)
        # print(f"🔧 [BA] Step {i+1} | loss = {loss.item():.4e}")
        if loss.item() < stop_loss_thresh:
            # print(f"🛑 [BA] 提前终止：loss={loss.item():.2e} < {stop_loss_thresh}")
            break

    optimized_pts3d = graph.pts3d.data.detach()
    return torch2np(optimized_pts3d)


def estimate_pose_epnp(intrinsics, points3d, points2d):
    """
    Estimate camera pose using EPnP algorithm.

    Args:
        intrinsics (torch.Tensor): Camera intrinsic matrix of shape (3, 3).
        points3d (torch.Tensor): 3D points in world coordinates of shape (N, 3).
        points2d (torch.Tensor): Corresponding 2D points in image plane of shape (N, 2).

    Returns:
        torch.Tensor: The estimated camera pose as an SE(3) LieTensor.
    """
    # Validate input shapes
    assert points3d.shape[1] == 3, "3D points should have shape (N, 3)"
    assert points2d.shape[1] == 2, "2D points should have shape (N, 2)"
    assert intrinsics.shape == (3, 3), "Intrinsic matrix should be of shape (3, 3)"

    # Initialize EPnP module
    epnp = pp.module.EPnP(np2torch(intrinsics))

    # Estimate the pose
    pose = epnp(np2torch(points3d), np2torch(points2d))
    pose = pp.matrix(pose).cpu().numpy()
    return pose