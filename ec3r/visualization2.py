
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import cv2
import time
import threading
import torch
import gc

def create_camera_frustum(scale=0.03):
    points = np.array([
        [0, 0, 0],
        [1, 0.75, 1],
        [-1, 0.75, 1],
        [-1, -0.75, 1],
        [1, -0.75, 1],
    ]) * scale
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])
    return frustum




def fast_visualization(keyframes, imgmatch, point3d,currentpose, device="cuda:0"):
    class VisualizerApp:
        def __init__(self):
            gui.Application.instance.initialize()
            self.window = gui.Application.instance.create_window("Fast Visualization GUI", 1440, 900)
            self.imgmatch = imgmatch
            self.currentpose = currentpose
            # 3D Scene
            self.scene_widget = gui.SceneWidget()
            self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
            self.scene_widget.scene.set_background([0.4667, 0.4667, 0.4667, 1.0])
            self.window.add_child(self.scene_widget)

            # Right panel
            self.panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
            # 🔵 蓝色圆点和 Key Frame 标签
            self.panel.add_child(gui.Label("   Key Frame"))  # ↓

            # 🔴 红色圆点和 Current Frame 标签
            self.image_widget = gui.ImageWidget()

            self.panel.add_child(self.image_widget)

            self.window.add_child(self.panel)
            self.panel.add_child(gui.Label("   Current Frame"))  # ↑

            self.window.set_on_layout(self._on_layout)

            # 材质
            self.pcd_material = rendering.MaterialRecord()
            self.pcd_material.shader = "defaultUnlit"
            self.pcd_material.point_size = 2.0

            self.line_material = rendering.MaterialRecord()
            self.line_material.shader = "unlitLine"
            self.line_material.line_width = 2.0

            self.static_material = rendering.MaterialRecord()
            self.static_material.shader = "defaultUnlit"
            self.static_material.point_size = 8.0  # 设置静态点云更大一点

            # 状态
            self.keyframes = keyframes
            self.point3d = point3d
            self.last_pcd_idx = 0
            self.last_pose_idx = 0
            self.point3d_added = False

            # Dummy点云
            self.dummy_pcd = o3d.geometry.PointCloud()
            self.dummy_pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
            self.dummy_pcd.colors = o3d.utility.Vector3dVector(np.ones((1, 3)))
            self.scene_widget.scene.add_geometry("init_dummy", self.dummy_pcd, self.pcd_material)

            # 初始相机
            bbox = o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1])
            self.scene_widget.setup_camera(60, bbox, [0, 0, 0])
            self.window.set_on_close(self.close_window)

            self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
            self.update_thread.start()
            gui.Application.instance.run()
        def close_window(self):
            """安全退出"""
            print("🛑 正在关闭可视化窗口...")
            self.update_thread.join()   # 等待线程停止
            del self.point3d            # 彻底清理
            o3d.visualization.gui.Application.instance.quit()
        def _on_layout(self, ctx):
            r = self.window.content_rect
            self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - 310, r.height)
            self.panel.frame = gui.Rect(r.get_right() - 300, r.y, 300, r.height)

        def update_loop(self):
            while True:
                gui.Application.instance.post_to_main_thread(self.window, self.update_scene)
                time.sleep(0.2)  # 5 FPS
        def update_current_pose(self, currentpose):
            try:
                pose = np.array(currentpose).reshape((4, 4))  # 转为 4x4 SE(3)
                if self.scene_widget.scene.has_geometry("current_frustum"):
                    self.scene_widget.scene.remove_geometry("current_frustum")
                frustum = create_camera_frustum(scale=0.05)  # 稍微大一点以示区别
                frustum.paint_uniform_color([1.0, 0.0, 1.0])  # 红色
                frustum.transform(pose)
                self.scene_widget.scene.add_geometry("current_frustum", frustum, self.line_material)
            except Exception as e:
                print(f"[ERROR] Failed to update current pose frustum: {e}")
        def update_scene(self):
            updated = False
            with torch.no_grad():
                with self.keyframes.lock:
                    dirty_index, = torch.where((self.keyframes.dirty.clone()) & (self.keyframes.ismapframe.clone()))

                if len(dirty_index) != 0:
                    # ✅ 只更新 1-2 个 dirty index，避免一次全量更新卡死
                    n_update = min(2, len(dirty_index))
                    dirty_index = dirty_index[:n_update]
                    self.keyframes.dirty[dirty_index] = False

                    poses = torch.index_select(self.keyframes.c2ws, 0, dirty_index).cpu().numpy()
                    for i in range(len(dirty_index)):
                        idx = dirty_index[i].item()
                        pose = poses[i].astype(np.float32)
                        points = self.keyframes.point[idx].cpu().numpy()
                        colors = self.keyframes.color[idx].cpu().numpy()
                        confs = self.keyframes.conf[idx].cpu().numpy()
                        pointpose = self.keyframes.pointpose[idx].cpu().numpy()
                        pose = pose @ pointpose

                        if points.size == 0 or np.isnan(points).any():
                            continue

                        points = points.reshape(-1, 3)
                        N = points.shape[0]
                        points_h = np.hstack([points, np.ones((N, 1))])
                        points_world_h = (pointpose @ points_h.T).T
                        points = points_world_h[:, :3]

                        colors = colors.reshape(-1, 3)
                        confs = confs.reshape(-1)

                        if len(confs) == 0:
                            continue
                        threshold = np.percentile(confs, 80)
                        mask = confs > threshold

                        max_points = 80000
                        if np.sum(mask) > max_points:
                            indices = np.where(mask)[0]
                            sampled = np.random.choice(indices, max_points, replace=False)
                            new_mask = np.zeros_like(mask, dtype=bool)
                            new_mask[sampled] = True
                            mask = new_mask

                        filtered_points = points[mask]
                        filtered_colors = colors[mask]

                        if len(filtered_points) == 0 or np.isnan(filtered_points).any():
                            continue

                        cloud_name = f"keyframe_{idx}"
                        if self.scene_widget.scene.has_geometry(cloud_name):
                            self.scene_widget.scene.remove_geometry(cloud_name)

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(filtered_points)
                        pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
                        self.scene_widget.scene.add_geometry(cloud_name, pcd, self.pcd_material)

                        # 更新相机 frustum
                        frustum = create_camera_frustum(scale=0.05)
                        frustum.transform(pose)
                        cam_name = f"camera_{idx}"
                        if self.scene_widget.scene.has_geometry(cam_name):
                            self.scene_widget.scene.remove_geometry(cam_name)
                        self.scene_widget.scene.add_geometry(cam_name, frustum, self.line_material)

                        updated = True

                # 添加静态点云
                if self.point3d is not None and len(self.point3d) > 0:
                    if self.scene_widget.scene.has_geometry("static_point3d"):
                        self.scene_widget.scene.remove_geometry("static_point3d")

                    pcd_static = o3d.geometry.PointCloud()
                    pcd_static.points = o3d.utility.Vector3dVector(np.array(self.point3d))
                    pcd_static.colors = o3d.utility.Vector3dVector(
                        np.tile(np.array([[0.0, 1.0, 0.0]]), (len(self.point3d), 1))
                    )
                    self.scene_widget.scene.add_geometry("static_point3d", pcd_static, self.static_material)
                    updated = True

                try:
                    if len(self.imgmatch) > 0:
                        img_array = np.array(self.imgmatch[0])
                        if img_array.dtype != np.uint8:
                            img_array = img_array.astype(np.uint8)
                        img_array = cv2.resize(img_array, (240, 320))
                        o3d_img = o3d.geometry.Image(img_array)
                        self.image_widget.update_image(o3d_img)
                except Exception as e:
                    print(f"[ERROR] imgmatch read failed: {e}")

                if self.currentpose is not None:
                    self.update_current_pose(self.currentpose)

                if updated:
                    self.scene_widget.force_redraw()

    VisualizerApp()  