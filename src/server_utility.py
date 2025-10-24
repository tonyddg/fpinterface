from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from typing import Dict, List, Union
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from scipy.spatial.transform import Rotation as R

from .multitask import MeshInfo, FPMultiTask

CandidatesQualityLevelMap = {
    "l": { # 84
        "min_n_views": 40, 
        "inplane_step": 240,
    },
    "m": { # 126
        "min_n_views": 40, 
        "inplane_step": 120,
    },
    "h": { # 252 原始配置
        "min_n_views": 40, 
        "inplane_step": 60,
    }
}

def pack_numpy(
    # RGB 三通道 0-255
    rgb: np.ndarray,
    # 单通道 float32
    depth: np.ndarray,
    # 单通道 bool
    mask: np.ndarray,
    # 深度调整参数
    z_far: float = 1.5,
    z_near: float = 0.1
):
    '''
    将推理输入整合为一个 5 通道 uint8 数组
    '''
    
    assert rgb.shape[2] == 3, "RGB 图片数组格式错误"
    assert (rgb.shape[:2] == mask.shape[:2]) and (rgb.shape[:2] == depth.shape[:2]), "图片大小不统一"
    assert (rgb.dtype == np.uint8) and (depth.dtype == np.float32 or depth.dtype == np.float64) and (mask.dtype == np.uint8 or mask.dtype == np.bool_), "数据格式错误"

    pack_arr = np.zeros((depth.shape[0], depth.shape[1], 5), dtype = np.uint8)
    pack_arr[:, :, :3] = rgb
    
    depth_norm = (depth - z_near) / (z_far - z_near)
    depth_norm[depth_norm > 1] = 1
    depth_norm[depth_norm < 0] = 0
    pack_arr[:, :, 3] = np.asarray(depth_norm * 256, dtype = np.uint8)

    pack_arr[:, :, 4] = mask > 0

    return pack_arr

def unpack_numpy(
    pack_arr: np.ndarray,
    # 深度调整参数
    z_far: float = 1.5,
    z_near: float = 0.1
):
    '''
    将整合的输入解包为 rgb, depth, mask 三模态数据
    '''

    assert pack_arr.shape[2] == 5, f"图片包通道数 {pack_arr.shape[2]} 不为 5"
    assert pack_arr.dtype == np.uint8, f"图片包数组格式 {pack_arr.dtype} 错误"

    rgb = np.asarray(pack_arr[:, :, :3], dtype = np.uint8)

    depth_norm = np.asarray(pack_arr[:, :, 3], dtype = np.float32) / 256
    depth = depth_norm * (z_far - z_near) + z_near

    mask = np.asarray(pack_arr[:, :, 4] > 0, dtype = np.bool_)

    return (rgb, depth, mask)

@dataclass
class FPServerCfg:
    # 相机内参矩阵
    cam_k_path: str = "/dataset/mine/cam_K.txt"
    # 位姿估计质量参数
    candidate_quality: str = 'l'                # 候选位姿数
    refine_iteration: int = 3                   # 候选位姿迭代次数
    post_track: int = 5                         # 检测位姿迭代次数
    # 模型配置字典，模型名 : 模型路径
    mesh_cnf_dict: Dict[str, str] = field(default_factory = lambda : {})
    # 是否使用 bbox 中心作为模型坐标系原点（默认为模型坐标系）
    is_bbox_pose: bool = False
    # 深度调整参数
    z_far: float = 1.5
    z_near: float = 0.1

    @classmethod
    def from_cfg(
        cls,
        cfg_path: Union[str, Path]
    ):
        base_cfg = OmegaConf.structured(cls)
        yaml_cfg = OmegaConf.load(cfg_path)
        mix_cfg = OmegaConf.merge(base_cfg, yaml_cfg)
        
        res = OmegaConf.to_object(mix_cfg)
        assert isinstance(res, cls), f"类型 {type(res)} 与 {cls.__name__} 不匹配"
        return res

class FPRespone(BaseModel):
    # 坐标
    position: List[float] = Field(min_length = 3, max_length = 3)
    # 四元数，w 位于最后
    quat: List[float] = Field(min_length = 4, max_length = 4)

    @classmethod
    def T2Respone(
        cls, 
        pose_mat: np.ndarray
    ):
        assert pose_mat.shape == (4, 4), f"姿态矩阵形状 {pose_mat.shape} 不合法"

        position = pose_mat[:3, 3]
        rot = pose_mat[:3, :3]
        quat = R.from_matrix(rot).as_quat()

        return FPRespone(
            position = [float(p) for p in position],
            quat = [float(q) for q in quat]
        )

class FPServer:
    def __init__(
        self,
        cfg: Union[FPServerCfg, str, Path]
    ) -> None:
        
        if not isinstance(cfg, FPServerCfg):
            cfg = FPServerCfg.from_cfg(cfg)
        
        self.cfg = cfg
        self.est = FPMultiTask()
        self.K = np.loadtxt(self.cfg.cam_k_path).reshape(3,3)

        for mesh_name, mesh_path in cfg.mesh_cnf_dict.items():
            self.est.add_target(
                mesh_name,
                MeshInfo(
                    mesh_path, False, None,
                    min_n_views = CandidatesQualityLevelMap[self.cfg.candidate_quality]["min_n_views"],
                    inplane_step = CandidatesQualityLevelMap[self.cfg.candidate_quality]["inplane_step"],
                )
            )

    def infer(
        self,
        target_name: str,
        pack_arr: np.ndarray
    ):
        (rgb, depth, mask) = unpack_numpy(
            pack_arr,
            z_far = self.cfg.z_far,
            z_near = self.cfg.z_near
        )

        self.est.register_time_analyze(target_name, self.K, rgb, depth, mask, self.cfg.refine_iteration)
        if self.cfg.post_track > 0:
            self.est.track_one(target_name, rgb, depth, self.K, self.cfg.post_track)

        if self.cfg.is_bbox_pose:
            pose_mat = self.est.get_target_bbox_pose(target_name)
        else:
            pose_mat = self.est.get_target_origin_pose(target_name)

        return FPRespone.T2Respone(pose_mat)
