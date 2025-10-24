import numpy as np
import requests
import io

from scipy.spatial.transform import Rotation as R

import logging
logger = logging.getLogger(__name__)

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

API_BASE = "http://127.0.0.1:8000"
API_GET_CFG = "/get_cfg"
API_INFER = "/infer"

class FoundationPoseClient:
    
    def __init__(
        self,
        api_url: str = API_BASE
    ) -> None:
        
        self.api_get_cfg = api_url + API_GET_CFG
        self.api_infer = api_url + API_INFER

        self.infer_cfg = dict(requests.get(
            url = self.api_get_cfg,
            timeout = 10
        ).json())

        self.z_far = self.infer_cfg["z_far"]
        self.z_near = self.infer_cfg["z_near"]
    
    def infer(
        self,
        target: str,
        # RGB 三通道 0-255
        rgb: np.ndarray,
        # 单通道 float32
        depth: np.ndarray,
        # 单通道 bool
        mask: np.ndarray,
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

        pack_arr = pack_numpy(rgb, depth, mask)
        buf = io.BytesIO()
        np.savez_compressed(buf, pack_arr = pack_arr)

        respond = dict(requests.post(
            self.api_infer,
            files = {
                "pack_file": (
                    "data.npz", buf.getvalue(), "application/octet-stream"
                )
            },
            params = {"target": target},
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

if __name__ == "__main__":

    from pathlib import Path
    import os
    IS_IN_CONTAINER = os.environ.get("IN_CONTAINER", False)

    def get_depth(depth_file: str, zfar = np.inf):
        '''
        来自 YcbineoatReader.get_depth
        读取深度图片 (读入的深度图像素单位为 m，此处认为读取的深度图像素单位为 mm)
        '''
        depth = cv2.imread(depth_file, -1) / 1e3
        depth[(depth<0.001) | (depth>=zfar)] = 0

        return depth

    from dataclasses import dataclass, field
    import tyro
    
    app_dir = Path(__file__).parent
    @dataclass
    class CliArgs:
        # RGB 文件路径
        COLOR_IMG: str = app_dir.joinpath("../example/example/color.png").as_posix()
        # 掩膜图片路径
        MASK_IMG: str = app_dir.joinpath("../example/example/workspace_mask.png").as_posix()
        # 深度图片路径
        DEPTH_IMG: str = app_dir.joinpath("../example/example/depth.png").as_posix()
        # 相机内参矩阵文件
        K_PATH: str = app_dir.joinpath("../example/cam_K.txt").as_posix()
        # 模型文件路径
        MESH_FILE_ORIGIN: str = app_dir.joinpath("../example/blue_big/blue_big.PLY").as_posix()
    config = tyro.cli(CliArgs)

    import cv2
    import time

    vis = cv2.imread(config.COLOR_IMG) # 用于绘制
    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB) # 用于分析
    # 来自 YcbineoatReader.get_mask
    mask = cv2.imread(config.MASK_IMG, -1).astype(bool)
    # 获取深度图
    depth = get_depth(config.DEPTH_IMG)

    fpc = FoundationPoseClient()

    start_time = time.perf_counter()
    pose_mat = fpc.infer(
        "blue_big",
        rgb, depth, mask
    )
    use_time = time.perf_counter() - start_time
    print(f"use time: {use_time:.3f}s")

    print(pose_mat)

    # 如果在容器内环境，则再可视化相关结果
    if IS_IN_CONTAINER:
        # 绘制结果部分

        from matplotlib import pyplot as plt

        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from src.multitask import MeshInfo
        from src.utility import render

        K = np.loadtxt(config.K_PATH).reshape(3,3)

        mesh_info = MeshInfo(config.MESH_FILE_ORIGIN, False, min_n_views = 10, inplane_step = 120)
        # 绘制混合图像
        rgb_render_raw = np.zeros(rgb.shape)
        single_render, _ = render(mesh_info, pose_mat, depth.shape[0], depth.shape[1], K)
        rgb_render_raw += single_render
        mix_estimate_rgb = cv2.addWeighted(rgb, 0.5, (rgb_render_raw * 255).astype(np.uint8), 0.8, 0)
        # mix_estimate_vis = cv2.cvtColor(mix_estimate_rgb, cv2.COLOR_BGR2RGB) # 用于分析

        fig, axe = plt.subplot_mosaic([[0, 1], [2, 3]])
        axe[0].imshow(rgb)
        axe[1].imshow(depth)
        axe[2].imshow(mask)
        axe[3].imshow(mix_estimate_rgb)

        fig.savefig(app_dir.joinpath("../result.png").as_posix())
        plt.show()
