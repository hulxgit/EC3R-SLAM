import torch
import cv2
import numpy as np
from typing import List, Tuple

def preprocess_images(kfimgs, img_cur):
    """图像预处理: 自动处理torch tensor和numpy数组的转换，转成 BGR uint8"""
    # 当前图像
    if isinstance(img_cur, torch.Tensor):
        img_cur = img_cur.cpu().numpy()
    img_cur = ((img_cur) * 255).clip(0, 255).astype(np.uint8)
    img_cur_bgr = cv2.cvtColor(img_cur, cv2.COLOR_RGB2BGR)
    
    # 关键帧图像
    processed_kfimgs = []
    for kfimg in kfimgs:
        if isinstance(kfimg, torch.Tensor):
            kfimg = kfimg.cpu().numpy()
        kfimg = ((kfimg) * 255).clip(0, 255).astype(np.uint8)
        kfimg_bgr = cv2.cvtColor(kfimg, cv2.COLOR_RGB2BGR)
        processed_kfimgs.append(kfimg_bgr)
    
    return processed_kfimgs, img_cur_bgr

def orb_match_and_verify(img1, img2, visualize=False, ratio_threshold=0.3):
    """
    使用 ORB 特征匹配 + RANSAC 验证两帧是否为回环 (基于内点比例)
    
    返回:
        is_loop: 是否是回环
        inlier_ratio: 内点比例
        match_img: 匹配可视化图像（如果 visualize=True）
    """
    # 1️⃣ ORB 特征提取
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return False, 0.0, None
    
    # 2️⃣ 特征匹配 (BFMatcher + Hamming)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # 3️⃣ 选出前 N 个最佳匹配
    num_good_matches = min(100, len(matches))
    good_matches = matches[:num_good_matches]
    
    if len(good_matches) < 10:
        return False, 0.0, None
    
    # 4️⃣ 匹配点
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    
    # 5️⃣ RANSAC 验证 (基础矩阵)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 3.0)
    if mask is None:
        return False, 0.0, None
    
    inliers = mask.ravel().tolist().count(1)
    inlier_ratio = inliers / len(good_matches)
    
    # 6️⃣ 判定是否是回环：内点比例
    is_loop = inlier_ratio > ratio_threshold
    
    # 7️⃣ 可视化
    match_img = None
    if visualize:
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, 
                          matchesMask=mask.ravel().tolist(), flags=2)
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
    
    return is_loop, inlier_ratio, match_img

def detect_loop_closure(kfimgs, img_cur, visualize=False):
    """回环判断: 当至少3个关键帧的内点比例超过阈值则判定为回环"""
    processed_kfimgs, processed_img_cur = preprocess_images(kfimgs, img_cur)
    
    successful_count = 0
    inlier_ratios = []
    
    print(f"🔍 开始回环检测，共有 {len(processed_kfimgs)} 个关键帧")
    
    for i, kfimg in enumerate(processed_kfimgs):
        is_loop, inlier_ratio, match_img = orb_match_and_verify(
            processed_img_cur, kfimg, visualize=visualize, ratio_threshold=0.25
        )
        
        emoji = "✅" if is_loop else "❌"
        print(f"{emoji} 关键帧 {i+1}: 内点比例 = {inlier_ratio:.2f}, 是否成功 = {is_loop}")
        inlier_ratios.append(inlier_ratio)
        
        if is_loop:
            successful_count += 1
    
    # 回环条件: 至少3个成功
    is_loop_closure = successful_count >= 2
    loop_emoji = "🎯" if is_loop_closure else "🚫"
    print(f"{loop_emoji} 回环检测结果: 成功验证 {successful_count} 个关键帧，是否回环 = {is_loop_closure}")
    
    return is_loop_closure

# 🎯 使用示例
if __name__ == "__main__":
    # 示例: img_cur, kfimgs 是 0-1 torch.Tensor, (H,W,3)
    img_cur = torch.rand((384, 512, 3))
    kfimgs = [torch.rand((384, 512, 3)) for _ in range(5)]
    
    is_loop, count, inlier_ratios = detect_loop_closure(kfimgs, img_cur, visualize=True)
    print("最终回环结果:", is_loop)
