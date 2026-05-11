import os
import sys
from typing import List
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utility import *
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

from dataclasses import dataclass, field
import tyro

@dataclass
class CliArgs:
    # 展示模型
    model_path: str = "/app/example/yellow_middle/yellow_middle.PLY"
    # 相机内参矩阵文件
    cam_k_path: str = "/app/example/cam_K.txt"
    # 模型坐标
    position: List[float] = field(default_factory = lambda: [0, 0, 0.5])
    # 模型欧拉角
    euler: List[float] = field(default_factory = lambda: [0, 0, 0])

if __name__ == "__main__":

    config = tyro.cli(CliArgs)

    mi_or = MeshInfo(config.model_path, False, min_n_views = 10)
    K = np.loadtxt(config.cam_k_path).reshape(3,3)

    OB_IN_CAMS = np.identity(4)
    OB_IN_CAMS[:3, 3] = config.position

    OB_IN_CAMS[:3, :3] = R.from_euler("xyz", config.euler).as_matrix()

    res_or = render(
        mi_or, 
        OB_IN_CAMS,
        K = K,
        H = 480,
        W = 640,
    )

    fig, axes = plt.subplot_mosaic([["A", "B"]])

    axes["A"].imshow(res_or[0])
    axes["B"].imshow(res_or[1])

    plt.show()
