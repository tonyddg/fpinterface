from .client import FoundationPoseClient
from .utility import depth_filter
from .visualize import *

if __name__ == "__main__":

    from pathlib import Path
    from dataclasses import dataclass

    try:
        from matplotlib import pyplot as plt
        import tyro
    except:
        raise RuntimeError("运行示例前请先安装 tyro 与 matplotlib")
    
    app_dir = Path(__file__).parent.joinpath("../../../")
    @dataclass
    class CliArgs:
        # RGB 文件路径
        COLOR_IMG: str = app_dir.joinpath("example/example/color.png").as_posix()
        # 掩膜图片路径
        MASK_IMG: str = app_dir.joinpath("example/example/workspace_mask.png").as_posix()
        # 深度图片路径
        DEPTH_IMG: str = app_dir.joinpath("example/example/depth.png").as_posix()
        # 相机内参矩阵文件
        K_PATH: str = app_dir.joinpath("example/cam_K.txt").as_posix()
        # 模型文件路径
        MESH_FILE_ORIGIN: str = app_dir.joinpath("example/blue_big/blue_big.PLY").as_posix()
        # 是否绘制原始坐标系
        IS_DRAW_ORIN_POSE: bool = False
        # 绘制目标
        TARGET_LABEL: str = "blue_big"
    config = tyro.cli(CliArgs)

    import cv2
    import time

    rgb = cv2.cvtColor(cv2.imread(config.COLOR_IMG), cv2.COLOR_BGR2RGB) # 用于分析
    # 来自 YcbineoatReader.get_mask
    mask = cv2.imread(config.MASK_IMG, -1).astype(bool)
    # 获取深度图
    depth = cv2.imread(config.DEPTH_IMG, -1) / 1e3
    depth = depth_filter(depth)

    fpc = FoundationPoseClient()

    start_time = time.perf_counter()
    bbox_pose = fpc.infer(
        config.TARGET_LABEL,
        rgb, depth, mask, is_bbox_pose = True
    )
    use_time = time.perf_counter() - start_time
    print(f"use time: {use_time:.3f}s")
    print(bbox_pose)

    if config.IS_DRAW_ORIN_POSE:
        origin_pose = fpc.infer(
            config.TARGET_LABEL,
            rgb, depth, mask, is_bbox_pose = False
        )
    else: 
        origin_pose = None

    bbox_rgb = rgb.copy()
    bbox_min_xyz, bbox_max_xyz = fpc.get_bbox(config.TARGET_LABEL) 
    bbox_rgb = draw_mesh_axis_bbox(
        bbox_rgb, bbox_pose, fpc.cam_k, bbox_min_xyz=bbox_min_xyz, bbox_max_xyz = bbox_max_xyz, label = config.TARGET_LABEL, origin_pose = origin_pose
    )

    mask_rgb = rgb.copy()
    mask_rgb[mask == 0] = 0

    fig, axe = plt.subplot_mosaic([[0, 1], [2, 3]])
    axe[0].imshow(rgb)
    axe[1].imshow(depth)
    axe[2].imshow(mask_rgb)
    axe[3].imshow(bbox_rgb)

    plt.show()

