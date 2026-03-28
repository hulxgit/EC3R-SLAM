import sys
sys.path.append("vggt/")
from vggt.models.vggt import VGGT

from vggt.utils.load_fn import load_and_preprocess_images_vggt

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map,expand_to_homogeneous
import torch
from torchvision import transforms as TF
from PIL import Image

import numpy as np
def get_length(var):
    """获取不同类型变量的长度（第一维大小）"""
    if isinstance(var, list):
        return len(var)
    elif isinstance(var, np.ndarray):
        return var.shape[0]
    elif isinstance(var, torch.Tensor):
        return var.shape[0]

def init_infer(imgs,model,device,dtype):
    images = [i['img'].to(device) for i in imgs]
    images = torch.stack(images).to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            embeddings = model.forward_encoder(images)

            predictions,embeddings = model.forward_decoder(images,embeddings)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic

            # Convert tensors to numpy
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].squeeze(0)  # remove batch dimension
            predictions['pose_enc_list'] = None # remove pose_enc_list

            # Generate world points from depth map
            depth_map = predictions["depth"]  # (S, H, W, 1)
            world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
            # Clean up
            torch.cuda.empty_cache()
            extrinsic = expand_to_homogeneous(predictions["extrinsic"])
    images = [i.permute(1,2,0) for i in images]
    assert get_length(extrinsic) == get_length(world_points) == get_length(images) == get_length(predictions['depth_conf']) == get_length(embeddings) == get_length(predictions["intrinsic"]), \
    "变量长度不一致"    
    return extrinsic,world_points,images,predictions['depth_conf'],embeddings,predictions["intrinsic"]
    #return kfpose,global_pointmap,img_rgb_list,global_confidence,encoded_feats,positions,K_pred


def encoder_infer(imgs,model,device,dtype,feat_embeddings):
    assert len(imgs) >= len(feat_embeddings)
    images = [i['img'].to(device) for i in imgs]
    images = torch.stack(images).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):    
            if feat_embeddings:
                embeddings_new = model.forward_encoder(images[len(feat_embeddings):])
    return embeddings_new

def mapinfer(imgs,model,device,dtype,feat_embeddings):
    assert len(imgs) >= len(feat_embeddings)
    images = [i['img'].to(device) for i in imgs]
    images = torch.stack(images).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):  
            if   len(imgs) > len(feat_embeddings):
                if feat_embeddings:
                    embeddings_new = model.forward_encoder(images[len(feat_embeddings):])
                    #embeddings = feat_embeddings +embeddings_new
                    # 转换张量为列表
                    print(f"feat_embeddings 类型: {type(feat_embeddings)}")
                    print(f"embeddings_new 类型: {type(embeddings_new)}")
                    print(f"embeddings_new 形状: {embeddings_new.shape}")
                
                    # 1. 将列表转换为张量
                    if isinstance(feat_embeddings, list) and len(feat_embeddings)>0:
                        feat_embeddings = [
                    tensor.to(device) if tensor.device != device else tensor
                    for tensor in feat_embeddings
                ]
                        feat_embeddings_tensor = torch.stack(feat_embeddings).to(device)
                    #     # 假设列表中的元素是张量或可转换为张量的数据
                    #     #feat_embeddings_tensor = torch.stack(feat_embeddings)
                    # else:
                    #     feat_embeddings_tensor = torch.tensor(feat_embeddings)

                    # 再进行拼接 （n,99,1024）
                    embeddings = torch.cat([feat_embeddings_tensor.to(device), embeddings_new], dim=0)

                else:
                    embeddings = model.forward_encoder(images)
            else:

                feat_embeddings = [
                    tensor.to(device) if tensor.device != device else tensor
                    for tensor in feat_embeddings
                ]

                # 再执行拼接
                embeddings = torch.stack(feat_embeddings).to(device)
            predictions,embeddings = model.forward_decoder(images,embeddings)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic

            # Convert tensors to numpy
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].squeeze(0)  # remove batch dimension
            predictions['pose_enc_list'] = None # remove pose_enc_list

            # Generate world points from depth map
            depth_map = predictions["depth"]  # (S, H, W, 1)
            world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
            # Clean up
            torch.cuda.empty_cache()
            extrinsic = expand_to_homogeneous(predictions["extrinsic"])
    images = [i.permute(1,2,0) for i in images]
    assert get_length(extrinsic) == get_length(world_points) == get_length(images) == get_length(predictions['depth_conf']) == get_length(embeddings) == get_length(predictions["intrinsic"]), \
    "变量长度不一致"      
    return (
        world_points.float(),  # 转换为float32
        images,        # 转换为float32
        predictions['depth_conf'].to("cpu").float(),  # 先移到CPU再转float32
        embeddings.to("cpu").float(),                 # 先移到CPU再转float32
        extrinsic.to("cpu").float(),                  # 先移到CPU再转float32
        predictions["intrinsic"].to("cpu").float()    # 先移到CPU再转float32
    )
