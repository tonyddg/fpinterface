'''
基本功能测试
'''

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FoundationPose.estimater import *
from FoundationPose.datareader import *

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.multitask import *
from src.utility import *

import time
from matplotlib import pyplot as plt

from dataclasses import dataclass
import tyro

@dataclass
class CliArgs:
    # 模型文件路径
    MESH_FILE_ORIGIN: str = "/app/example/blue_big/blue_big.PLY"
    # RGB 文件路径
    COLOR_IMG: str = "/app/fpinterface-client/example/color.png"
    # 掩膜图片路径
    MASK_IMG: str = "/app/fpinterface-client/example/workspace_mask.png"
    # 深度图片路径
    DEPTH_IMG: str = "/app/fpinterface-client/example/depth.png"
    # 相机内参矩阵文件
    K_PATH: str = "/app/example/cam_K.txt"

if __name__ == "__main__":

    config = tyro.cli(CliArgs)

    ### 获取预测所需的图片
    # 来自 YcbineoatReader.get_color (读取图片应为 RGB)
    vis = cv2.imread(config.COLOR_IMG) # 用于绘制 (YOLO 预测)
    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB) # 用于分析
    # 来自 YcbineoatReader.get_mask
    mask = cv2.imread(config.MASK_IMG, -1).astype(bool)
    # 获取深度图
    depth = get_depth(config.DEPTH_IMG)
    print(depth)
    # 获取相机内参矩阵
    K = np.loadtxt(config.K_PATH).reshape(3,3)

    s = time.perf_counter()
    est = FPMultiTask()
    print(f"Load Use time: {time.perf_counter() - s}")

    # 位姿估计

    s = time.perf_counter()

    mesh_info = MeshInfo(config.MESH_FILE_ORIGIN, False, min_n_views = 10, inplane_step = 120)
    est.add_target("11", mesh_info)

    print(f"Load Mesh time: {time.perf_counter() - s}")

    est.register_time_analyze("11", K, rgb, depth, mask, 5)
    print(est.get_target_origin_pose("11"))

    vis_estimate = est.draw_last_pose(vis, K, "11", (255, 0, 0))
    vis_estimate = cv2.cvtColor(vis_estimate, cv2.COLOR_BGR2RGB)

    # 位姿修正

    s = time.perf_counter()
    est.track_one("11", rgb, depth, K, 5)
    print(f"Track Use time: {time.perf_counter() - s}")
    print(est.get_target_origin_pose("11"))

    vis_track = est.draw_last_pose(vis, K, "11")
    vis_track = cv2.cvtColor(vis_track, cv2.COLOR_BGR2RGB)

    # 绘制混合图像

    rgb_render_raw = np.zeros(rgb.shape)
    single_render, _ = render(mesh_info, est.get_target_origin_pose("11"), depth.shape[0], depth.shape[1], K)
    rgb_render_raw += single_render
    vis_mix = cv2.addWeighted(rgb, 0.2, (rgb_render_raw * 255).astype(np.uint8), 0.8, 0)

    # 绘制结果

    fig, axes = plt.subplot_mosaic([["A", "B"], ["C", "D"]])

    axes["A"].imshow(depth)
    axes["A"].set_title("Depth")

    axes["B"].imshow(vis_estimate)
    axes["B"].set_title("Estimate")

    axes["C"].imshow(vis_track)
    axes["C"].set_title("Refine")

    axes["D"].imshow(vis_mix)
    axes["D"].set_title("Mix")

    plt.show()


