import cv2
import numpy as np
import torch

from .multitask import MeshInfo

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FoundationPose.Utils import nvdiffrast_render

def get_depth(depth_file: str, zfar = np.inf):
    '''
    来自 YcbineoatReader.get_depth
    读取深度图片 (处理深度图片是除以了 1e3, 原因未知, 单位转换?)
    '''
    depth = cv2.imread(depth_file, -1) / 1e3
    depth[(depth<0.001) | (depth>=zfar)] = 0

    return depth

def letter_box_resize(img: np.ndarray, size: tuple, blank: tuple = (0.5, 0.5, 0.5)):
    '''
    通过缩放的方式将图片以占据尽可能多空间的方式放入 size 空间中, 放入时居中图片  
    使用 (h, w) 的原则

    返回值包含调整后图片的 resize_w, resize_h, offset_w, offset_h
    '''
    w = img.shape[1]
    h = img.shape[0]
    scale = 1
    offset_w = 0
    offset_h = 0

    resize_w = size[1]
    resize_h = size[0]

    if w > h:
        # 宽与缩放后的图片对齐, 高居中
        scale = size[1] / w

        resize_h = h * scale
        offset_h = (size[0] - h * scale) / 2
    else:
        scale = size[0] / h

        resize_w = w * scale
        offset_w = (size[1] - w * scale) / 2

    trans = np.array(
        [
            [scale, 0, offset_w],
            [0, scale, offset_h]
        ]
    )

    return cv2.warpAffine(img, trans, size, borderValue = blank), resize_w, resize_h, offset_w, offset_h

def render(mesh_info: MeshInfo, ob_in_cams, H, W, K, glctx = None):
    res = nvdiffrast_render(
        K = K,
        H = H,
        W = W,
        ob_in_cams = torch.tensor(ob_in_cams, dtype = torch.float32, device = "cuda"),
        mesh_tensors = mesh_info.mesh_tensors,
        use_light=True,
        glctx=glctx
    )

    color = torch.squeeze(res[0]).cpu().numpy()
    depth = torch.squeeze(res[1]).cpu().numpy()
    return (color, depth)
