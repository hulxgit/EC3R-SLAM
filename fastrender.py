# from pvd_utils import *
import matplotlib.pyplot as pl
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import plotly.graph_objects as go
import plotly.subplots as sp
import trimesh
from omegaconf import OmegaConf, DictConfig
import os
import time
import copy
import shutil
import open3d as o3d
import rootutils
# from pvd_utils import *
rootutils.setup_root("./fast3r", indicator=".project-root", pythonpath=True)
from fast3r.models.fast3r import Fast3R
from fast3r.dust3r.inference_multiview import inference
from fast3r.dust3r.model import FlashDUSt3R
from image import load_images
from fast3r.dust3r.viz import CAM_COLORS, OPENGL, add_scene_cam, cat_meshes, pts3d_to_trimesh
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import torch
import plotly.graph_objects as go
from fast3r.dust3r.cloud_opt.init_im_poses import fast_pnp
from fast3r.dust3r.viz import auto_cam_size
from fast3r.dust3r.viz_plotly import SceneViz
from fast3r.dust3r.utils.image import rgb  # Assuming you have this utility for image processing
from fast3r.dust3r.inference_multiview_custom import inference_custom
from ec3r.utils.slam_utils import select_keyframes   #dont forget!!!!




pl.ion()


def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')



def get_reconstructed_scene(
    outdir,
    model,
    device,
    silent,
    image_size,
    filelist,
    profiling=False,
    dtype=torch.float32,
    rotate_clockwise_90=False,
    crop_to_landscape=False,
):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    multiple_views_in_one_sample = load_images(filelist, size=image_size)
    for i, sample in enumerate(multiple_views_in_one_sample):
        print(f"\nSample {i}:")
        for key, val in sample.items():
            if hasattr(val, 'shape'):
                print(f"  {key}: shape = {tuple(val.shape)}")
                print(val)
            elif isinstance(val, (list, tuple)):
                print(f"  {key}: list/tuple of length {len(val)}")
            else:
                print(f"  {key}: type = {type(val)}")
                print(val)

    img_ori = (multiple_views_in_one_sample[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [576,1024,3] [0,1]
    print("multiple_views_in_one_sample",multiple_views_in_one_sample[0]["img"].shape)
    # time the inference
    start = time.time()
    # multiple_views_in_one_sample torch.Size([1, 3, 384, 512])

    output = inference(multiple_views_in_one_sample, model, device, dtype=dtype, verbose=not silent, profiling=profiling)
    end = time.time()
    print(f"Time elapsed: {end - start}")

    return output,img_ori

def mutiviewinfer(
    imginput,
    model,
    device,
    silent,
    profiling=False,
    dtype=torch.float32,
):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    # time the inference
    start = time.time()
    output = inference(imginput, model, device, dtype=dtype, verbose=not silent, profiling=profiling)
    end = time.time()
    print(f"⚡️ Time elapsed for network inference: {end - start:.3f}s")

    return output



def plot_rgb_images(views, title="RGB Images", save_image_to_folder=None):
    fig = sp.make_subplots(rows=1, cols=len(views), subplot_titles=[f"View {i} Image" for i in range(len(views))])

    # Plot the RGB images
    for i, view in enumerate(views):
        img_rgb = view['img'].cpu().numpy().squeeze().transpose(1, 2, 0)  # Shape: (224, 224, 3)
         # Rescale RGB values from [-1, 1] to [0, 255]
        img_rgb = ((img_rgb + 1) * 127.5).astype(int).clip(0, 255)
        
        fig.add_trace(go.Image(z=img_rgb), row=1, col=i+1)

        if save_image_to_folder:
            img_path = os.path.join(save_image_to_folder, f"view_{i}.png")
            pl.imsave(img_path, img_rgb.astype(np.uint8))

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # fig.show()

def plot_confidence_maps(preds, title="Confidence Maps", save_image_to_folder=None):
    fig = sp.make_subplots(rows=1, cols=len(preds), subplot_titles=[f"View {i} Confidence" for i in range(len(preds))])

    # Plot the confidence maps
    for i, pred in enumerate(preds):
        conf = pred['conf'].cpu().numpy().squeeze()
        fig.add_trace(go.Heatmap(z=conf, colorscale='turbo', showscale=False), row=1, col=i+1)

        if save_image_to_folder:
            conf_path = os.path.join(save_image_to_folder, f"view_{i}_conf.png")
            pl.imsave(conf_path, conf, cmap='turbo')

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    for i in range(len(preds)):
        fig['layout'][f'yaxis{i+1}'].update(autorange='reversed')

    # fig.show()

def maybe_plot_local_depth_and_conf(preds, title="Local Depth and Confidence Maps", save_image_to_folder=None):
    # Define the number of columns based on available keys
    num_plots = len(preds)
    rows = 2  # one for confidence maps, one for depth maps
    cols = num_plots

    # Create subplots for both confidence and depth maps
    fig = sp.make_subplots(
        rows=rows, 
        cols=cols, 
        subplot_titles=[f"View {i+1} Conf" if 'conf_local' in pred else f"View {i+1} No Conf" for i, pred in enumerate(preds)]
    )

    # Iterate over preds to add confidence and depth maps if the fields exist
    for i, pred in enumerate(preds):
        # Add confidence map if "conf_local" exists
        if 'conf_local' in pred:
            conf_local = pred['conf_local'].cpu().numpy().squeeze()
            fig.add_trace(go.Heatmap(z=conf_local, colorscale='Turbo', showscale=False), row=1, col=i+1)

            if save_image_to_folder:
                conf_local_path = os.path.join(save_image_to_folder, f"view_{i}_conf_local.png")
                pl.imsave(conf_local_path, conf_local, cmap='turbo')
        
        # Add depth map if "pts3d_local" exists
        if 'pts3d_local' in pred:
            # Extract Z values as depth from pts3d_local (XY plane)
            depth_local = pred['pts3d_local'][..., 2].cpu().numpy().squeeze()  # Use the Z-coordinate
            fig.add_trace(go.Heatmap(z=depth_local, colorscale='Greys', showscale=False), row=2, col=i+1)

            if save_image_to_folder:
                depth_local_path = os.path.join(save_image_to_folder, f"view_{i}_depth_local.png")
                pl.imsave(depth_local_path, depth_local, cmap='Greys')
        

    # Update layout for the figure
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Reverse the y-axis for each subplot for consistency
    for i in range(num_plots):
        if 'conf_local' in preds[i]:
            fig['layout'][f'yaxis{i*2+1}'].update(autorange='reversed')
        if 'pts3d_local' in preds[i]:
            fig['layout'][f'yaxis{i*2+2}'].update(autorange='reversed')

    # fig.show()

def plot_3d_points_with_colors(preds, views, title="3D Points Visualization", flip_axes=False, as_mesh=False, min_conf_thr_percentile=80, export_ply_path=None):
    fig = go.Figure()

    all_points = []
    all_colors = []
    
    if as_mesh:
        meshes = []
        for i, pred in enumerate(preds):
            pts3d = pred['pts3d_in_other_view'].cpu().numpy().squeeze()  # Ensure tensor is on CPU and convert to numpy
            img_rgb = views[i]['img'].cpu().numpy().squeeze().transpose(1, 2, 0)  # Shape: (224, 224, 3)
            conf = pred['conf'].cpu().numpy().squeeze()

            # Determine the confidence threshold based on the percentile
            conf_thr = np.percentile(conf, min_conf_thr_percentile)

            # Filter points based on the confidence threshold
            mask = conf > conf_thr

            # Rescale RGB values from [-1, 1] to [0, 255]
            img_rgb = ((img_rgb + 1) * 127.5).astype(np.uint8).clip(0, 255)

            # Generate the mesh for the current view
            mesh_dict = pts3d_to_trimesh(img_rgb, pts3d, valid=mask)
            meshes.append(mesh_dict)

        # Concatenate all meshes
        combined_mesh = trimesh.Trimesh(**cat_meshes(meshes))

        # Flip axes if needed
        if flip_axes:
            combined_mesh.vertices[:, [1, 2]] = combined_mesh.vertices[:, [2, 1]]
            combined_mesh.vertices[:, 2] = -combined_mesh.vertices[:, 2]

        # Export as .ply if the path is provided
        if export_ply_path:
            combined_mesh.export(export_ply_path)

        # Add the combined mesh to the plotly figure
        vertex_colors = combined_mesh.visual.vertex_colors[:, :3]  # Ensure the colors are in RGB format
        # Map vertex colors to face colors
        face_colors = []
        for face in combined_mesh.faces:
            face_colors.append(np.mean(vertex_colors[face], axis=0))
        face_colors = np.array(face_colors).astype(int)
        face_colors = ['rgb({}, {}, {})'.format(r, g, b) for r, g, b in face_colors]

        fig.add_trace(go.Mesh3d(
            x=combined_mesh.vertices[:, 0], 
            y=combined_mesh.vertices[:, 1], 
            z=combined_mesh.vertices[:, 2],
            i=combined_mesh.faces[:, 0], 
            j=combined_mesh.faces[:, 1], 
            k=combined_mesh.faces[:, 2],
            facecolor=face_colors,
            opacity=0.5,
            name="Combined Mesh"
        ))
    else:
        # Loop through each set of points in preds
        for i, pred in enumerate(preds):
            pts3d = pred['pts3d_in_other_view'].cpu().numpy().squeeze()  # Ensure tensor is on CPU and convert to numpy
            img_rgb = views[i]['img'].cpu().numpy().squeeze().transpose(1, 2, 0)  # Shape: (224, 224, 3)
            conf = pred['conf'].cpu().numpy().squeeze()

            # Determine the confidence threshold based on the percentile
            conf_thr = np.percentile(conf, min_conf_thr_percentile)

            # Flatten the points and colors
            x, y, z = pts3d[..., 0].flatten(), pts3d[..., 1].flatten(), pts3d[..., 2].flatten()
            r, g, b = img_rgb[..., 0].flatten(), img_rgb[..., 1].flatten(), img_rgb[..., 2].flatten()
            conf_flat = conf.flatten()

            # Apply confidence mask
            mask = conf_flat > conf_thr
            x, y, z = x[mask], y[mask], z[mask]
            r, g, b = r[mask], g[mask], b[mask]

            # Collect points and colors for exporting
            all_points.append(np.vstack([x, y, z]).T)
            all_colors.append(np.vstack([r, g, b]).T)



        # # 可视化
            # Rescale RGB values from [-1, 1] to [0, 255]
            r = ((r + 1) * 127.5).astype(int).clip(0, 255)
            g = ((g + 1) * 127.5).astype(int).clip(0, 255)
            b = ((b + 1) * 127.5).astype(int).clip(0, 255)

            colors = ['rgb({}, {}, {})'.format(r[j], g[j], b[j]) for j in range(len(r))]
            
            # Check the flag and flip axes if needed
            if flip_axes:
                x, y, z = x, z, y
                z = -z

            # Add points to the plot
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=2, opacity=0.8, color=colors),
                name=f"View {i}"
            ))

        # Export as .ply if the path is provided
        if export_ply_path:
            all_points = np.vstack(all_points)
            all_colors = np.vstack(all_colors)
            point_cloud = trimesh.PointCloud(vertices=all_points, colors=all_colors)
            point_cloud.export(export_ply_path)
    xyz_min = np.min(all_points, axis=0)
    xyz_max = np.max(all_points, axis=0)
    ranges = xyz_max - xyz_min
    max_range = np.max(ranges)
    aspect_ratio = dict(x=ranges[0] / max_range,
                        y=ranges[1] / max_range,
                        z=ranges[2] / max_range)


    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='manual',
            aspectratio=aspect_ratio  # ✅ 严格等比例缩放！
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=1000
    )

    fig.show()



# Function to visualize 3D points and camera poses with SceneViz
def plot_3d_points_with_estimated_camera_poses(preds, views, title="3D Points and Camera Poses", flip_axes=False, min_conf_thr_percentile=80, export_ply_path=None, export_html_path=None):
    # Initialize SceneViz for visualization
    viz = SceneViz()

    # Flip axes if requested
    if flip_axes:
        preds = copy.deepcopy(preds)
        for i, pred in enumerate(preds):
            pts3d = pred['pts3d_in_other_view']
            pts3d = pts3d[..., [0, 2, 1]]  # Swap Y and Z axes
            pts3d[..., 2] *= -1  # Flip the sign of the Z axis
            pred['pts3d_in_other_view'] = pts3d  # Reassign the modified points back to pred

    # Estimate camera poses and focal lengths
    poses_c2w, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(preds, niter_PnP=10)
    poses_c2w = poses_c2w[0]  # batch size is 1
    estimated_focals = estimated_focals[0]  # batch size is 1
    cam_size = max(auto_cam_size(poses_c2w), 0.05)  # Auto-scale based on the point cloud

    # Set up point clouds and visualization
    for i, (pred, pose_c2w) in enumerate(zip(preds, poses_c2w)):
        pts3d = pred['pts3d_in_other_view'].cpu().numpy().squeeze()  # (224, 224, 3)
        img_rgb = rgb(views[i]['img'].cpu().numpy().squeeze().transpose(1, 2, 0))  # Shape: (224, 224, 3)
        conf = pred['conf'].cpu().numpy().squeeze()

        # Determine the confidence threshold based on the percentile
        conf_thr = np.percentile(conf, min_conf_thr_percentile)
        mask = conf > conf_thr

        # Add the point cloud directly to the SceneViz object
        viz.add_pointcloud(pts3d, img_rgb, mask=mask, point_size=1.0, view_idx=i)
        color = tuple(np.random.randint(0, 255, size=3).tolist())  # 转换成元组

        # Add camera to the visualization
        viz.add_camera(
            pose_c2w=pose_c2w,  # Estimated camera-to-world pose
            focal=estimated_focals[i],  # Estimated focal length for each view
            color=np.random.randint(0, 255, size=3),  # Generate a random RGB color for each camera # Generate a random RGB color for each camera
            image=img_rgb,  # Image of the view
            cam_size=cam_size,  # Auto-scaled camera size
            view_idx=i
        )

    # Export point clouds and meshes if the path is provided
    if export_ply_path:
        all_points = []
        all_colors = []
        for i, pred in enumerate(preds):
            pts3d = pred['pts3d_in_other_view'].cpu().numpy().squeeze()
            img_rgb = views[i]['img'].cpu().numpy().squeeze().transpose(1, 2, 0)
            conf = pred['conf'].cpu().numpy().squeeze()
            conf_thr = np.percentile(conf, min_conf_thr_percentile)
            mask = conf > conf_thr
            all_points.append(pts3d[mask])
            all_colors.append(img_rgb[mask])
        
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        point_cloud = trimesh.PointCloud(vertices=all_points, colors=all_colors)
        point_cloud.export(export_ply_path)
    
    if export_html_path:
        viz.export_html(export_html_path)

    # Show the visualization
    viz.show()

def save_pointmaps_and_camera_parameters_to_folder(preds,views, save_folder, niter_PnP=100, focal_length_estimation_method='individual'):
    """
    Saves pointmaps and estimated camera parameters to a folder.

    Args:
        preds (list): List of prediction dictionaries containing point maps and confidence scores.
        save_folder (str): Path to the folder where the numpy data structure will be saved.
    """
    # Estimate camera poses and focal lengths
    poses_c2w, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(preds, niter_PnP=niter_PnP, focal_length_estimation_method=focal_length_estimation_method)
    poses_c2w = poses_c2w[0]  # Assuming batch size is 1
    estimated_focals = estimated_focals[0]  # Assuming batch size is 1

    # Initialize lists to hold the data
    global_pointmap = []
    global_confidence = []
    local_pointmap = []
    local_aligned_to_global_pointmap = []
    local_confidence = []
    estimated_focals_list = []
    estimated_poses_c2w_list = []
    img_rgb_list = []
    # Loop over predictions and extract required data
    for i, pred in enumerate(preds):
        # Extract global point map
        pts3d_in_other_view = pred['pts3d_in_other_view'].cpu().numpy().squeeze()  # Shape: H x W x 3
        global_pointmap.append(pts3d_in_other_view)
        
        # Extract global confidence map
        conf = pred['conf'].cpu().numpy().squeeze()  # Shape: H x W
        global_confidence.append(conf)
        
        # Extract local point map
        pts3d_local = pred['pts3d_local'].cpu().numpy().squeeze()  # Shape: H x W x 3
        local_pointmap.append(pts3d_local)

        # Extract local aligned to global point map
        # pts3d_local_aligned = pred['pts3d_local_aligned_to_global'].cpu().numpy().squeeze()  # Shape: H x W x 3
        # local_aligned_to_global_pointmap.append(pts3d_local_aligned)
        
        # Extract local confidence map
        conf_local = pred['conf_local'].cpu().numpy().squeeze()
        local_confidence.append(conf_local)

        # Append estimated focal length and camera pose
        focal = estimated_focals[i].item() if isinstance(estimated_focals[i], torch.Tensor) else estimated_focals[i]
        estimated_focals_list.append(focal)
        pose = poses_c2w[i].cpu().numpy() if isinstance(poses_c2w[i], torch.Tensor) else poses_c2w[i]
        estimated_poses_c2w_list.append(pose)
        
        img_rgb = (views[i]['img']* 0.5 + 0.5).cpu().numpy().squeeze().transpose(1, 2, 0)  # Shape: (H, W, 3)
        img_rgb_list.append(img_rgb)

    return {
        "global_pointmaps": global_pointmap,
        # "global_confidence_maps": global_confidence,
        # "local_pointmaps": local_pointmap,
        # "local_aligned_to_global_pointmaps": local_aligned_to_global_pointmap,
        "local_confidence_maps": local_confidence,
        # "estimated_focals": estimated_focals_list,
        "estimated_poses_c2w": estimated_poses_c2w_list,
        "rgb": img_rgb_list
    }

def export_combined_ply(preds, views, export_ply_path=None, 
                        pts3d_key_to_visualize="pts3d_local_aligned_to_global",
                        conf_key_to_visualize="conf_local",
                        min_conf_thr_percentile=0, flip_axes=False, max_num_points=None, sampling_strategy='uniform'):
    all_points = []
    all_colors = []

    # Loop through each set of points in preds
    for i, pred in enumerate(preds):
        pts3d = pred[pts3d_key_to_visualize].cpu().numpy().squeeze()  # Ensure tensor is on CPU and convert to numpy
        img_rgb = views[i]['img'].cpu().numpy().squeeze().transpose(1, 2, 0)  # Shape: (H, W, 3)
        conf = pred[conf_key_to_visualize].cpu().numpy().squeeze()

        # Determine the confidence threshold based on the percentile
        conf_thr = np.percentile(conf, min_conf_thr_percentile)

        # Flatten the points and colors
        x, y, z = pts3d[..., 0].flatten(), pts3d[..., 1].flatten(), pts3d[..., 2].flatten()
        r, g, b = img_rgb[..., 0].flatten(), img_rgb[..., 1].flatten(), img_rgb[..., 2].flatten()
        conf_flat = conf.flatten()

        # Apply confidence mask
        mask = conf_flat > conf_thr
        x, y, z = x[mask], y[mask], z[mask]
        r, g, b = r[mask], g[mask], b[mask]

        # Rescale RGB values from [-1, 1] to [0, 255]
        r = ((r + 1) * 127.5).astype(np.uint8).clip(0, 255)
        g = ((g + 1) * 127.5).astype(np.uint8).clip(0, 255)
        b = ((b + 1) * 127.5).astype(np.uint8).clip(0, 255)

        # Collect points and colors for exporting
        points = np.vstack([x, y, z]).T
        colors = np.vstack([r, g, b]).T

        # Check the flag and flip axes if needed
        if flip_axes:
            points = points[:, [0, 2, 1]]  # Swap y and z
            points[:, 2] = -points[:, 2]   # Invert z-axis

        all_points.append(points)
        all_colors.append(colors)

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    # If max_num_points is specified, downsample the point cloud using the selected sampling strategy
    if max_num_points is not None and len(all_points) > max_num_points:
        if sampling_strategy == 'uniform':
            # Uniform random sampling
            indices = np.random.choice(len(all_points), size=max_num_points, replace=False)
            all_points = all_points[indices]
            all_colors = all_colors[indices]
        elif sampling_strategy == 'voxel':
            # Voxel grid downsampling
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_points)
            pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64) / 255.0)
            
            # Estimate a voxel size to achieve the desired number of points
            # This is a heuristic and may need adjustment
            bounding_box = pcd.get_axis_aligned_bounding_box()
            extent = bounding_box.get_extent()
            volume = extent[0] * extent[1] * extent[2]
            voxel_size = (volume / max_num_points) ** (1/3)

            down_pcd = pcd.voxel_down_sample(voxel_size)

            # Extract downsampled points and colors
            all_points = np.asarray(down_pcd.points)
            all_colors = (np.asarray(down_pcd.colors) * 255.0).astype(np.uint8)
        elif sampling_strategy == 'farthest_point':
            # Farthest point downsampling using Open3D
            # Note: May be slow for large point clouds
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_points)
            pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64) / 255.0)

            down_pcd = pcd.farthest_point_down_sample(max_num_points)

            # Extract downsampled points and colors
            all_points = np.asarray(down_pcd.points)
            all_colors = (np.asarray(down_pcd.colors) * 255.0).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")

    # Export as .ply if the path is provided
    if export_ply_path:
        point_cloud = trimesh.PointCloud(vertices=all_points, colors=all_colors)
        point_cloud.export(export_ply_path)

    return all_points, all_colors




# multi-cam dynamic scenes
# Function to get sorted file list from a directory
def get_sorted_file_list(directory):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


# display the images
def display_images(filelist, title, rotate_clockwise_90=False, crop_to_landscape=False):
    fig, axes = plt.subplots(1, len(filelist), figsize=(30, 4))
    fig.suptitle(title)
    for ax, filepath in zip(axes if hasattr(axes, '__iter__') else [axes], filelist):
        img = Image.open(filepath)
        if rotate_clockwise_90:
            img = img.rotate(-90, expand=True)
        if crop_to_landscape:
            # Crop to a landscape aspect ratio (e.g., 16:9)
            desired_aspect_ratio = 4 / 3
            width, height = img.size
            current_aspect_ratio = width / height

            if current_aspect_ratio > desired_aspect_ratio:
                # Wider than landscape: crop width
                new_width = int(height * desired_aspect_ratio)
                left = (width - new_width) // 2
                right = left + new_width
                top = 0
                bottom = height
            else:
                # Taller than landscape: crop height
                new_height = int(width / desired_aspect_ratio)
                top = (height - new_height) // 2
                bottom = top + new_height
                left = 0
                right = width
            
            img = img.crop((left, top, right, bottom))
        
        ax.imshow(img)
        ax.axis('off')
    plt.show()


def generate_traj_interp(c2ws,H,W,fs,c,ns,device):

    c2ws = interp_traj(c2ws,n_inserts= ns,device=device)
    num_views = c2ws.shape[0] 
    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))

    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)

    fs = interpolate_sequence(fs,ns-2,device=device)
    c = interpolate_sequence(c,ns-2,device=device)
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    
    return cameras, num_views,w2c



def run_render(pcd, imgs,masks, H, W, camera_traj,num_views,nbv=False):
    render_setup = setup_renderer(camera_traj, image_size=(H,W))
    renderer = render_setup['renderer']
    render_results, depthmap,viewmask = render_pcd(pcd, imgs, masks, num_views,renderer,"cuda:0",nbv=False)
    return render_results,depthmap, viewmask

def load_images_to_numpy(image_folder):
    """
    Load all PNG images from a folder and convert them to NumPy arrays.

    Args:
        image_folder (str): Path to the folder containing images.

    Returns:
        numpy.ndarray: NumPy array of shape (N, H, W, C) where N is the number of images.
    """
    image_list = []
    image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])

    for filename in image_filenames:
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
        image_array = np.array(image, dtype=np.uint8)  # Convert to NumPy array
        image_list.append(image_array)

    return np.stack(image_list, axis=0)  # Stack into (N, H, W, C) array


def fastrecon(save_dir):
    # # Display train images
    # display_images(filelist_train, 'Train Images')
    # directory = "./re10k_2/gt_rgb"
    imgdir = os.path.join(save_dir,"img")

    filelist_test = get_sorted_file_list(imgdir)
    videodir = os.path.join(save_dir,"video")
    os.makedirs(videodir,exist_ok=True)
    sfmdir = os.path.join(videodir,"sfm")
    os.makedirs(sfmdir,exist_ok=True)
    # skip this
    device = torch.device("cuda")

    checkpoint_root = "jedyang97/Fast3R_ViT_Large_512"

    model = Fast3R.from_pretrained(checkpoint_root).to(device)
    model = model.to(device)
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)

    # Set model to evaluation mode
    model.eval()
    lit_module.eval()


    output,img_ori = get_reconstructed_scene(
        outdir =sfmdir,
        model = model,
        device = device,
        silent = False,
        # image_size = 224,
        image_size = 512,
        filelist = filelist_test,
        profiling=True,
        dtype = torch.float32,
    )
    # views = output[1]['views']
    # preds=output[1]['preds']
    output = output[0]
    # local to global alignment
    # before fix: lit_module.align_local_pts3d_to_global(preds=output['preds'], views=output['views'], min_conf_thr_percentile=0)
    # lit_module.align_local_pts3d_to_global(preds=output['preds'], views=output['views'], min_conf_thr_percentile=85)
    # Camera pose evaluation on RealEstate10K
    # local to global alignment
        # before fix: lit_module.align_local_pts3d_to_global(preds=output['preds'], views=output['views'], min_conf_thr_percentile=0)
    lit_module.align_local_pts3d_to_global(preds=output['preds'], views=output['views'], min_conf_thr_percentile=85)

    dirs_to_create = [sfmdir, os.path.join(sfmdir, "rgb_images"), os.path.join(sfmdir, "global_confidence_maps"), os.path.join(sfmdir, "local_depth_and_confidence_maps")]

    if os.path.exists(sfmdir):
        shutil.rmtree(sfmdir)

    for d in dirs_to_create:
        if not os.path.exists(d):
            os.makedirs(d)

    plot_rgb_images(output['views'], save_image_to_folder=os.path.join(sfmdir, "rgb_images"))

    # Plot the confidence maps
    plot_confidence_maps(output['preds'], save_image_to_folder=os.path.join(sfmdir, "global_confidence_maps"))

    # Plot the local depth and confidence maps
    maybe_plot_local_depth_and_conf(output['preds'], save_image_to_folder=os.path.join(sfmdir, "local_depth_and_confidence_maps"))

    export_combined_ply(
        preds=output['preds'],
        views=output['views'],
        pts3d_key_to_visualize="pts3d_local_aligned_to_global",
        conf_key_to_visualize="conf_local",
        export_ply_path=os.path.join(sfmdir, "combined_pointcloud.ply"),
        min_conf_thr_percentile=45,
        flip_axes=True,
        max_num_points=1_000_000,          # Set your desired maximum number of points here
        sampling_strategy='uniform'    # Choose 'uniform', 'voxel', or 'farthest_point'
    )
    save_pointmaps_and_camera_parameters_to_folder(preds=output['preds'],views=output['views'], save_folder=sfmdir, niter_PnP=100, focal_length_estimation_method='first_view_from_global_head')


    return output


def fast3rmap(filelist_test):

 
    # skip this
    device = torch.device("cuda")

    checkpoint_root = "jedyang97/Fast3R_ViT_Large_512"

    model = Fast3R.from_pretrained(checkpoint_root).to(device)
    model = model.to(device)
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)

    # Set model to evaluation mode
    model.eval()
    lit_module.eval()


    output,img_ori = get_reconstructed_scene(
        outdir =None,
        model = model,
        device = device,
        silent = False,
        # image_size = 224,
        image_size = 512,
        filelist = filelist_test,
        profiling=True,
        dtype = torch.float32,
    )

    output = output[0]
    lit_module.align_local_pts3d_to_global(preds=output['preds'], views=output['views'], min_conf_thr_percentile=85)

    fastresult = save_pointmaps_and_camera_parameters_to_folder(preds=output['preds'],views=output['views'], save_folder=None, niter_PnP=100, focal_length_estimation_method='first_view_from_global_head')
    return fastresult

def fast3rinfer(imgs,device,model):

 
    output = mutiviewinfer(
    imgs,
    model,
    device,
    False,
    profiling=False,
    dtype=torch.float32,
    )

    fastresult = save_pointmaps_and_camera_parameters_to_folder(preds=output['preds'],views=output['views'], save_folder=None, niter_PnP=100, focal_length_estimation_method='first_view_from_global_head')

    return fastresult



def postprocess(preds,views, niter_PnP=100, focal_length_estimation_method='individual'):
    """
    Saves pointmaps and estimated camera parameters to a folder.

    Args:
        preds (list): List of prediction dictionaries containing point maps and confidence scores.
        save_folder (str): Path to the folder where the numpy data structure will be saved.
    """
    # Estimate camera poses and focal lengths
    poses_c2w, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(preds, niter_PnP=niter_PnP, focal_length_estimation_method=focal_length_estimation_method)
    poses_c2w = poses_c2w[0]  # Assuming batch size is 1
    estimated_focals = estimated_focals[0]  # Assuming batch size is 1

    # Initialize lists to hold the data
    global_pointmap = []
    global_confidence = []
    local_pointmap = []
    local_aligned_to_global_pointmap = []
    local_confidence = []
    estimated_focals_list = []
    estimated_poses_c2w_list = []
    img_rgb_list = []
    # Loop over predictions and extract required data
    for i, pred in enumerate(preds):
        # Extract global point map
        pts3d_in_other_view = pred['pts3d_in_other_view'].cpu().numpy().squeeze()  # Shape: H x W x 3
        global_pointmap.append(pts3d_in_other_view)
        
        # Extract global confidence map
        conf = pred['conf'].cpu().numpy().squeeze()  # Shape: H x W
        global_confidence.append(conf)
        
        # Extract local point map
        pts3d_local = pred['pts3d_local'].cpu().numpy().squeeze()  # Shape: H x W x 3
        local_pointmap.append(pts3d_local)

        
        # Extract local confidence map
        conf_local = pred['conf_local'].cpu().numpy().squeeze()
        local_confidence.append(conf_local)

        # Append estimated focal length and camera pose
        focal = estimated_focals[i].item() if isinstance(estimated_focals[i], torch.Tensor) else estimated_focals[i]
        estimated_focals_list.append(focal)
        pose = poses_c2w[i].cpu().numpy() if isinstance(poses_c2w[i], torch.Tensor) else poses_c2w[i]
        estimated_poses_c2w_list.append(pose)
        
        img_rgb = (views[i]['img']* 0.5 + 0.5).cpu().numpy().squeeze().transpose(1, 2, 0)  # Shape: (H, W, 3)
        img_rgb_list.append(img_rgb)
    
    
    
    return {
        "global_pointmaps": global_pointmap,
        # "global_confidence_maps": global_confidence,
        # "local_pointmaps": local_pointmap,
        # "local_aligned_to_global_pointmaps": local_aligned_to_global_pointmap,
        "local_confidence_maps": local_confidence,
        # "estimated_focals": estimated_focals_list,
        "estimated_poses_c2w": estimated_poses_c2w_list,
        "rgb": img_rgb_list
    }





def adjust_pointcloud_with_intrinsics(points, K_pred, K_real):
    """
    使用已知的相机内参 K_real 对预测的点云进行缩放

    参数:
    - points: (N, 3) 神经网络预测的点云
    - K_pred: (3, 3) 神经网络估计的相机内参
    - K_real: (3, 3) 真实的相机内参

    返回:
    - new_points: (N, 3) 调整后的点云
    """
    # 转换到像素坐标
    points = np.array(points).reshape(-1,3)
    uvs = (K_pred @ points.T).T  # (N, 3)

    # 归一化
    uvs[:, 0] /= uvs[:, 2]
    uvs[:, 1] /= uvs[:, 2]
    uvs[:, 2] = 1.0

    # 使用真实的相机内参反投影到新空间
    K_real_inv = np.linalg.inv(K_real)
    new_points = (K_real_inv @ uvs.T).T

    # 恢复深度
    new_points *= points[:, 2].reshape(-1, 1)

    return new_points

def init_infer(imgs,model,device,dtype):
    output, encoded_feats, positions = inference_custom(
        imgs,
        model,
        device,
       dtype,   # ✅ 明确指定 dtype，不要放在第四个位置
        feat_embeddings=None,
        pose_embeddings=None,
        shapes= None,
                )
    torch.cuda.empty_cache()

    poses_c2w, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(preds=output['preds'], niter_PnP=100, focal_length_estimation_method='first_view_from_global_head')



    # Initialize lists to hold the data
    global_pointmap = []
    global_confidence = []
    local_pointmap = []
    local_confidence = []
    estimated_focals_list = []
    estimated_poses_c2w_list = []
    img_rgb_list = []
    # Loop over predictions and extract required data
    preds = output['preds']
    views=output['views']
    for i, pred in enumerate(preds):
        # Extract global point map
        pts3d_in_other_view = pred['pts3d_in_other_view'].squeeze()  # Shape: H x W x 3
        global_pointmap.append(pts3d_in_other_view)
        
        # Extract global confidence map
        conf = pred['conf'].squeeze()  # Shape: H x W
        global_confidence.append(conf)
        
        # Extract local point map
        pts3d_local = pred['pts3d_local'].squeeze()  # Shape: H x W x 3
        local_pointmap.append(pts3d_local)
        conf_local = pred['conf_local'].squeeze()
        local_confidence.append(conf_local)

        # Append estimated focal length and camera pose
        img_rgb = (views[i]['img']* 0.5 + 0.5).squeeze().permute(1, 2, 0) # Shape: (H, W, 3)
        img_rgb_list.append(img_rgb)

    # adjusted_points = adjust_pointcloud_with_intrinsics(global_pointmap, K_pred, K_real)
    # adjusted_points = adjusted_points.reshape(np.array(global_pointmap).shape)
    # adjusted_points = [adjusted_points[i] for i in range(adjusted_points.shape[0])]
    # adjusted_points = [global_pointmap[i].cpu().numpy() for i in range(len(global_pointmap))]
    H,W,_ = global_pointmap[0].shape
    kfpose = []
    K_pred = np.array([
    [estimated_focals[0][0], 0., W/2],
    [0., estimated_focals[0][0], H/2],
    [0., 0., 1.]
])

    for conf, pts3d in zip(global_confidence, global_pointmap):
        # 转换为 numpy.float32
        H, W = conf.shape
        conf = conf.cpu().numpy().reshape(-1).astype(np.float32)        # [H*W]
        pts3d = pts3d.cpu().numpy().reshape(-1, 3).astype(np.float32)   # [H*W, 3]

        # 生成像素坐标 [u, v]
        pixels = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).reshape(-1, 2).astype(np.float32)  # [H*W, 2]

        # 选置信度最高的前 1/3 点
        N = H * W
        num_select = N // 3
        top_indices = np.argsort(-conf)[:num_select]
        mask = np.zeros(N, dtype=bool)
        mask[top_indices] = True

        # 执行 PnP + RANSAC
        success, R_vec, T, inliers = cv2.solvePnPRansac(
            pts3d[mask],                  # objectPoints [N, 3]
            pixels[mask],                # imagePoints  [N, 2]
            K_pred.astype(np.float32),   # cameraMatrix
            None,                        # distCoeffs
            None, None, False,           # useExtrinsicGuess = False
            100,                         # iterationsCount
            5.0,                         # reprojectionError
            0.99,                        # confidence
            None,                        # inliers
            cv2.SOLVEPNP_SQPNP           # method flag
        )

        assert success, "PnP failed on current frame"

        # 转换为 [4x4] 的位姿矩阵（cam-to-world）
        R = cv2.Rodrigues(R_vec)[0]      # [3x3]
        T = T.reshape(3, 1)              # [3x1]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3:] = T
        pose_inv = np.linalg.inv(pose)   # 从 world→cam 转为 cam→world

        kfpose.append(pose_inv)

    return kfpose,global_pointmap,img_rgb_list,global_confidence,encoded_feats,positions,K_pred



def mapinfer(imgs,model,device,dtype,feat_embeddings,pose_embeddings,shapes,encoder_only):
    if feat_embeddings is not None:
        feat_embeddings=[f.to(device) for f in feat_embeddings]
        pose_embeddings=[p.to(device) for p in pose_embeddings]
    
    output, encoded_feats, positions = inference_custom(
        imgs,
        model,
        device,
       dtype,   # ✅ 明确指定 dtype，不要放在第四个位置
        feat_embeddings= feat_embeddings,
        pose_embeddings=pose_embeddings,
        shapes = shapes,
        encoder_only = encoder_only
                )
    
    torch.cuda.synchronize()     # 确保所有 CUDA 操作完成
    torch.cuda.empty_cache()     # 释放未使用的缓存显存
    if not encoder_only:
        # Initialize lists to hold the data
        global_pointmap = []
        global_confidence = []
        img_rgb_list = []
        # Loop over predictions and extract required data
        preds = output['preds']
        views=output['views']
        for i, pred in enumerate(preds):
            # Extract global point map
            pts3d_in_other_view = pred['pts3d_in_other_view'].squeeze()  # Shape: H x W x 3
            global_pointmap.append(pts3d_in_other_view)
            
            # Extract global confidence map
            conf = pred['conf'].squeeze()  # Shape: H x W
            global_confidence.append(conf)
            

            # Append estimated focal length and camera pose
            img_rgb = (views[i]['img']* 0.5 + 0.5).squeeze().permute(1, 2, 0) # Shape: (H, W, 3)
            img_rgb_list.append(img_rgb)

        return global_pointmap,img_rgb_list,global_confidence,encoded_feats,positions

    else:
        return output, encoded_feats,positions

def image_similarity(feat1, feat2, mode="cross_topk", top_k=5):
    """
    计算两个图像特征之间的相似度。

    参数:
        feat1, feat2: [N, D] CroCo/VIT patch embeddings
        mode: str, 相似度计算模式，可选：
            - "mean_cosine": 全局平均后余弦相似度
            - "patch_mean": 同位置 patch 平均余弦相似度
            - "cross_max": cross-patch 最大相似度
            - "cross_topk": cross-patch top-k 平均相似度（鲁棒）
        top_k: int, 仅在 "cross_topk" 模式下使用

    返回:
        float, 相似度（越大越相似）
    """
    if mode == "mean_cosine":
        emb1 = feat1.mean(dim=0)
        emb2 = feat2.mean(dim=0)
        return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()

    elif mode == "patch_mean":
        return torch.nn.functional.cosine_similarity(feat1, feat2, dim=1).mean().item()

    elif mode == "cross_max":
        feat1 = feat1 / feat1.norm(dim=1, keepdim=True)
        feat2 = feat2 / feat2.norm(dim=1, keepdim=True)
        sim_matrix = torch.matmul(feat1, feat2.T)
        return sim_matrix.max().item()

    elif mode == "cross_topk":
        feat1 = feat1 / feat1.norm(dim=1, keepdim=True)
        feat2 = feat2 / feat2.norm(dim=1, keepdim=True)
        sim_matrix = torch.matmul(feat1, feat2.T)  # shape: [N, N]

        # 双向 top-k 平均相似度
        topk_sim1 = torch.topk(sim_matrix, top_k, dim=1).values.mean(dim=1)  # 每行 top-k
        topk_sim2 = torch.topk(sim_matrix, top_k, dim=0).values.mean(dim=0)  # 每列 top-k

        similarity_score = (topk_sim1.mean() + topk_sim2.mean()) / 2
        return similarity_score.item()

    else:
        raise ValueError(f"Unknown mode: {mode}")
