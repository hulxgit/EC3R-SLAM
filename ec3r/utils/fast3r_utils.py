import PIL
from PIL import Image
import numpy as np
import os
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def loadfast3r(path=None, device="cuda"):
    checkpoint_root = "/home/lingxianghu/EC3R-SLAM/checkpoints/model.safetensors"
    model = Fast3R.from_pretrained(checkpoint_root).to(device)
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(device)
    model.eval()
    lit_module.eval()
    return model,lit_module


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)

import torchvision.transforms as tvf

# 定义ToTensor转换器（用于将图像转为0-1范围的张量）
to_tensor = tvf.ToTensor()

def resize_img(img, target_size, return_transformation=False):
    """
    参照load_and_preprocess_images函数逻辑，处理图像并将backend_img转换为0-1范围
    
    Args:
        img: 输入的numpy数组图像（已归一化到0-1范围）
        target_size: 目标宽度尺寸
        return_transformation: 是否返回变换参数
        
    Returns:
        预处理后的图像数据字典，包含：
        - img: 归一化处理后的张量（[-1,1]范围）
        - true_shape: 图像形状
        - unnormalized_img: 原始未处理图像
        - backend_img: 缩放裁剪后的图像（0-1范围张量）
        若return_transformation=True，还返回缩放和裁剪参数
    """
    # 将0-1范围的numpy数组转换为0-255的PIL图像
    img_pil = Image.fromarray(np.uint8(img * 255))
    ori_img = img_pil.copy()
    W1, H1 = img_pil.size  # 原始宽度和高度
    
    # 计算缩放比例，保持宽高比（优先调整宽度到目标尺寸）
    scale = target_size / W1
    new_width = target_size
    new_height = round(H1 * scale)
    
    # 确保高度是14的倍数（模型兼容性处理）
    new_height = (new_height // 14) * 14
    if new_height == 0:
        new_height = 14  # 避免过小尺寸
    
    # 缩放图像
    resized_img = img_pil.resize((new_width, new_height), Image.Resampling.BICUBIC)
    W, H = resized_img.size  # 缩放后的宽度和高度
    
    # 中心裁剪：如果高度超过目标尺寸，裁剪到目标尺寸
    if H > target_size:
        cy = H // 2
        halfh = target_size // 2
        resized_img = resized_img.crop((0, cy - halfh, W, cy + halfh))
        H = target_size  # 更新裁剪后的高度
    
    # 关键步骤：使用ToTensor将PIL图像转为0-1范围的张量
    # ToTensor会自动完成：PIL图像(0-255) → 张量(0.0-1.0)，并调整维度为[C, H, W]
    backend_img_tensor = to_tensor(resized_img)
    # 准备返回结果
    res = dict(
        true_shape=np.int32([resized_img.size[::-1]]),  # (高度, 宽度)
        unnormalized_img=np.asarray(resized_img),  # 原始图像数组(0-255)
        backend_img=backend_img_tensor  # 0-1范围的张量[C, H, W]
    )
    # 计算变换参数（缩放和裁剪偏移）
    if return_transformation:
        scale_w = W1 / new_width  # 宽度缩放比例（原始/缩放后）
        scale_h = H1 / new_height  # 高度缩放比例（原始/缩放后）
        
        # 裁剪偏移（如果有裁剪）
        crop_offset_y = (new_height - H) // 2 if new_height > H else 0
        crop_offset_x = 0  # 宽度未裁剪
        
        return res, (scale_w, scale_h, crop_offset_x, crop_offset_y)
    
    return res
    