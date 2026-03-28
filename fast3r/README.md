<div align="center">

# ⚡️Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass


${{\color{Red}\Huge{\textsf{  CVPR\ 2025\ \}}}}\$


[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2501.13928)
[![Project Website](https://img.shields.io/badge/Fast3R-Website-4CAF50?logo=googlechrome&logoColor=white)](https://fast3r-3d.github.io/)
[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-orange?style=flat&logo=Gradio&logoColor=red)](https://fast3r.ngrok.app/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/jedyang97/Fast3R_ViT_Large_512/)
</div>

![Teaser Image](assets/teaser.png)

Official implementation of **Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass**, CVPR 2025

*[Jianing Yang](https://jedyang.com/), [Alexander Sax](https://alexsax.github.io/), [Kevin J. Liang](https://kevinjliang.github.io/), [Mikael Henaff](https://www.mikaelhenaff.net/), [Hao Tang](https://tanghaotommy.github.io/), [Ang Cao](https://caoang327.github.io/), [Joyce Chai](https://web.eecs.umich.edu/~chaijy/), [Franziska Meier](https://fmeier.github.io/), [Matt Feiszli](https://www.linkedin.com/in/matt-feiszli-76b34b/)*

## Installation

```bash
# clone project
git clone https://github.com/facebookresearch/fast3r
cd fast3r

# create conda environment
conda create -n fast3r python=3.11 cmake=3.14.0 -y
conda activate fast3r

# install PyTorch (adjust cuda version according to your system)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia

# install PyTorch3D from source (the compilation will take a while)
# export MAX_JOBS=6 # un-comment this if your machine is low on RAM (e.g., 16GB) when compiling PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# unset MAX_JOBS

# install requirements
pip install -r requirements.txt

# install fast3r as a package (so you can import fast3r and use it in your own project)
pip install -e .
```

Note: Please make sure to NOT install the cuROPE module like in DUSt3R - it would mess up the model's prediction.

<details>
<summary>Installation Troubleshooting</summary>

If you encounter the error `fatal error: cuda_runtime.h: No such file or directory` when installing PyTorch3D, try setting `CUDA_HOME` before installing PyTorch3D:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
</details>

## Demo

Use the following command to run the demo:

```bash
python fast3r/viz/demo.py
```
This will automatically download the pre-trained model weights and config from [Hugging Face Model](https://huggingface.co/jedyang97/Fast3R_ViT_Large_512).

The demo is a Gradio interface where you can upload images or a video and visualize the 3D reconstruction and camera pose estimation.

`fast3r/viz/demo.py` also serves as an example of how to use the model for inference.

<div>
  <img src="assets/fast3r_demo_upload.gif" width="45%" alt="Demo GIF 1" />
  <img src="assets/fast3r_demo_control.gif" width="45%" alt="Demo GIF 2" style="margin-left: 5%;" />
  <br>
  <em>Left: Upload a video. Right: Visualize the 3D Reconstruction</em>
</div>

<details>
<summary>Click here to see example of: visualize confidence heatmap + play frame by frame + render a GIF</summary>
<div style="display: flex; justify-content: center;">
  <img src="assets/fast3r_demo_coloring.gif" width="100%" alt="Demo GIF 3" />
</div>
</details>

## Using Fast3R in Your Own Project

To use Fast3R in your own project, you can import the `Fast3R` class from `fast3r.models.fast3r` and use it as a regular PyTorch model.

```python
import torch
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

# --- Setup ---
# Load the model from Hugging Face
model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")  # If you have networking issues, try pre-download the HF checkpoint dir and change the path here to a local directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create a lightweight lightning module wrapper for the model.
# This provides functions to estimate camera poses, evaluate 3D reconstruction, etc.
lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)

# Set model to evaluation mode
model.eval()
lit_module.eval()

# --- Load Images ---
# Provide a list of image file paths. Images can come from different cameras and aspect ratios.
filelist = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"]
images = load_images(filelist, size=512, verbose=True)

# --- Run Inference ---
# The inference function returns a dictionary with predictions and view information.
output_dict, profiling_info = inference(
    images,
    model,
    device,
    dtype=torch.float32,  # or use torch.bfloat16 if supported
    verbose=True,
    profiling=True,
)

# --- Estimate Camera Poses ---
# This step estimates the camera-to-world (c2w) poses for each view using PnP.
poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
    output_dict['preds'],
    niter_PnP=100,
    focal_length_estimation_method='first_view_from_global_head'
)
# poses_c2w_batch is a list; the first element contains the estimated poses for each view.
camera_poses = poses_c2w_batch[0]

# Print camera poses for all views.
for view_idx, pose in enumerate(camera_poses):
    print(f"Camera Pose for view {view_idx}:")
    print(pose.shape)  # np.array of shape (4, 4), the camera-to-world transformation matrix

# --- Extract 3D Point Clouds for Each View ---
# Each element in output_dict['preds'] corresponds to a view's point map.
for view_idx, pred in enumerate(output_dict['preds']):
    point_cloud = pred['pts3d_in_other_view'].cpu().numpy()
    print(f"Point Cloud Shape for view {view_idx}: {point_cloud.shape}")  # shape: (1, 368, 512, 3), i.e., (1, Height, Width, XYZ)
```

## Training

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python fast3r/train.py experiment=super_long_training/super_long_training
```

You can override any parameter from command line following [Hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/):

```bash
python fast3r/train.py experiment=super_long_training/super_long_training trainer.max_epochs=20 trainer.num_nodes=2
```

To submit a multi-node training job with Slurm, use the following command:

```bash
python scripts/slurm/submit_train.py --nodes=<NODES> --experiment=<EXPERIMENT>
```

After training, you can run the demo with a lightning checkpoint with the following command:
```bash
python fast3r/viz/demo.py --is_lightning_checkpoint --checkpoint_dir=/path/to/super_long_training_999999
```

## Evaluation

To evaluate on 3D reconstruction or camera pose estimation tasks, run:

```bash
python fast3r/eval.py eval=<eval_config>
```
`<eval_config>` can be any of the evaluation configurations in [configs/eval/](configs/eval/). For example:
- `ablation_recon_better_inference_hp/ablation_recon_better_inference_hp` evaluates the 3D reconstruction on DTU, 7-Scenes and Neural-RGBD datasets.
- `eval_cam_pose/eval_cam_pose_10views` evaluates the camera pose estimation on 10 views on CO3D dataset.


To evaluate camera poses on RealEstate10K dataset, run:

```bash
python scripts/fast3r_re10k_pose_eval.py  --subset_file scripts/re10k_test_1800.txt
```

To evaluate multi-view depth estimation on Tanks and Temples, ETH-3D, DTU, and ScanNet datasets, follow the data download and preparation guide of [robustmvd](https://github.com/lmb-freiburg/robustmvd), install that repo's `requirements.txt` into the current conda environment, and run:

```bash
python scripts/robustmvd_eval.py
```

## Dataset Preprocessing

Please follow [DUSt3R's data preprocessing instructions](https://github.com/naver/dust3r/tree/main?tab=readme-ov-file#datasets) to prepare the data for training and evaluation. The pre-processed data is compatible with the [multi-view dataloaders](fast3r/dust3r/datasets) in this repo.

For preprocessing the DTU, 7-Scene, and NRGBD datasets for evaluation, we follow [Spann3r's data processing instructions](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md).

## FAQ

- Q: `httpcore.ConnectError: All connection attempts failed` when launching the demo?
  - See [#34](https://github.com/facebookresearch/fast3r/issues/34). Download the example videos into a local directory.
- Q: Data pre-processing for BlendedMVS, `train_list.txt` is missing?
  - See [#33](https://github.com/facebookresearch/fast3r/issues/33).
- Q: Loading checkpoint to fine-tune Fast3R?
  - See [#25](https://github.com/facebookresearch/fast3r/issues/25)
- Q: Running demo on Windows? (TypeError: cannot pickle '_thread.RLock' object)
  - See [#28](https://github.com/facebookresearch/fast3r/issues/28). It seems that some more work is needed to make the demo compatible with Windows - we hope the community could contribute a PR!
- Q: Completely messed-up point cloud output?
  - See [#21](https://github.com/facebookresearch/fast3r/issues/21). Please make sure the cuROPE module is NOT installed.
- Q: My GPU doesn't support FlashAttention?
  - See [#17](https://github.com/facebookresearch/fast3r/issues/17). Use `attn_implementation=pytorch_auto` option instead.
- Q: `TypeError: Fast3R.__init__() missing 3 required positional arguments: 'encoder_args', 'decoder_args', and 'head_args'`
  - See See [#7](https://github.com/facebookresearch/fast3r/issues/7). It is caused by a networking issue with downloading the model from Huggingface in some countries (e.g., China) - please pre-download the model checkpoint with a working networking configuration, and use a local path to load the model instead.
## License

The code and models are licensed under the [FAIR NC Research License](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citation

```
@InProceedings{Yang_2025_Fast3R,
    title={Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass},
    author={Jianing Yang and Alexander Sax and Kevin J. Liang and Mikael Henaff and Hao Tang and Ang Cao and Joyce Chai and Franziska Meier and Matt Feiszli},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2025},
}
```

## Acknowledgement

Fast3R is built upon a foundation of remarkable open-source projects. We deeply appreciate the contributions of these projects and their communities, whose efforts have significantly advanced the field and made this work possible.

- [DUSt3R](https://dust3r.europe.naverlabs.com/)
- [Spann3R](https://hengyiwang.github.io/projects/spanner)
- [Viser](https://viser.studio/main/)
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=facebookresearch/fast3r&type=Date)](https://star-history.com/#facebookresearch/fast3r&Date)
