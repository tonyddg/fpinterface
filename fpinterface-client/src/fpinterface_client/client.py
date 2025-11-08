from typing import List, Optional
import numpy as np
import requests
import io

from scipy.spatial.transform import Rotation as R
from .utility import pack_numpy, depth_filter

import logging
logger = logging.getLogger(__name__)

class FoundationPoseClient:

    API_GET_CFG = "/get_cfg"
    API_INFER = "/infer"
    API_SET_K = "/set_cam_k"
    API_GET_BBOX = "/get_mesh_bbox"
    
    def __init__(
        self,
        host: str = "http://127.0.0.1",
        port: int = 8000,
        cam_k: Optional[np.ndarray] = None
    ) -> None:
        
        self.api_base = f"{host}:{port}"

        self.api_get_cfg = self.api_base + self.API_GET_CFG
        self.api_infer = self.api_base + self.API_INFER
        self.api_set_k = self.api_base + self.API_SET_K
        self.api_get_bbox = self.api_base + self.API_GET_BBOX

        try:
            self.infer_cfg = dict(requests.get(
                url = self.api_get_cfg,
                timeout = 10
            ).json())
        except Exception as e:
            raise RuntimeError(f"FoundationPose 连接失败, 检查是否启动服务, 错误为: {e}")

        self.z_far = self.infer_cfg["z_far"]
        self.z_near = self.infer_cfg["z_near"]

        if cam_k is not None:
            requests.post(
                url = self.api_set_k,
                json = {"cam_k": cam_k.flatten().tolist()},
                timeout = 10
            )
            self.cam_k = cam_k
        else:
            self.cam_k = np.array(self.infer_cfg["cam_k"]).reshape((3, 3))
    
    def infer(
        self,
        target: str,
        # RGB 三通道 0-255
        rgb: np.ndarray,
        # 单通道 float32
        depth: np.ndarray,
        # 单通道 bool
        mask: np.ndarray,
        # 是否使用包容盒坐标系
        is_bbox_pose: bool = True
    ):
        '''
        Args:
            target (str): 目标模型名称，对应配置文件中键 `mesh_cnf_dict` 下的字典
            rgb (np.ndarray): 三通道 Uint8 RGB 图片，对于 opencv 读取的图片需要使用 `rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 转化为 RGB 格式
            depth (np.ndarray): 单通道 float 图片，像素值表示以 m 为单位的深度值，单位不正确时需要转化
            mask (np.ndarray): 单通道 bool 图片，像素值为 True 时表示掩膜区域，可通过 mask = np.asarray(mask > 0, dtype = np.bool_) 转化

        Returns:
            out (np.ndarray): 目标 4 x 4 齐次矩阵

        '''
        if target not in self.infer_cfg["mesh_cnf_dict"].keys():
            raise RuntimeError(f"目标 {target} 不存在")

        pack_arr = pack_numpy(rgb, depth, mask, self.z_far, self.z_near)
        buf = io.BytesIO()
        np.savez_compressed(buf, pack_arr = pack_arr)

        respond = dict(requests.post(
            self.api_infer,
            files = {
                "pack_file": (
                    "data.npz", buf.getvalue(), "application/octet-stream"
                )
            },
            params = {"target": target, "is_bbox_pose": is_bbox_pose},
            timeout = 10
        ).json())

        logger.info(f"get respond: {respond}")

        # 将推理结果转换为位姿矩阵（默认为四元数）
        position = np.asarray(respond["position"], dtype = np.float64)
        quat = np.asarray(respond["quat"], dtype = np.float64)
        rot = R.from_quat(quat).as_matrix()

        pose_mat = np.identity(4)
        pose_mat[:3, :3] = rot
        pose_mat[:3, 3] = position

        return pose_mat

    def get_bbox(
        self,
        target: str
    ):
        respond = dict(requests.get(
            url = self.api_get_bbox,
            params = {"target": target},
            timeout = 10
        ).json())

        return (np.asarray(respond["min_xyz"]), np.asarray(respond["max_xyz"]))
