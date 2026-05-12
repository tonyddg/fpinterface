from typing import Optional

from fpinterface_client.client import FoundationPoseClient
from fpinterface_client.utility import depth_filter
from fpinterface_client.visualize import draw_mesh_axis_bbox

if __name__ == "__main__":

    from pathlib import Path
    from dataclasses import dataclass

    try:
        # Python >= 3.9
        from importlib.resources import files, as_file
    except ImportError:
        # Python 3.8
        from importlib_resources import files, as_file
        raise RuntimeError("运行示例前请通过 pip install 'fpinterface-client[cli] @ ...' 或 uv sync --all-extras 安装开发依赖 importlib_resources")

    try:
        from matplotlib import pyplot as plt
        import tyro
    except:
        raise RuntimeError("运行示例前请通过 pip install 'fpinterface-client[cli] @ ...' 或 uv sync --all-extras 安装开发依赖 tyro 与 matplotlib")

    @dataclass
    class CliArgs:

        @dataclass
        class ImgPath:
            ''' 图片路径, 不给出时使用示例图片'''
            # RGB 文件路径
            color: str
            # 掩膜图片路径
            mask: str
            # 深度图片路径
            depth: str

        # 图片路径, 不给出时使用示例图片
        src: Optional[ImgPath] = None
        # 是否绘制原始坐标系
        is_draw_orin_pose: bool = False
        # 绘制目标
        target_label: str = "blue_big"
        # 服务端地址
        host: str = "http://127.0.0.1"
        # 服务端端口
        port: int = 8000
    config = tyro.cli(CliArgs)

    import cv2
    import time

    if (config.src is None):
        
        # 加载示例图片
        
        example_color_img = files("fpinterface_client.assets").joinpath("color.png")
        example_mask_img = files("fpinterface_client.assets").joinpath("workspace_mask.png")
        example_depth_img = files("fpinterface_client.assets").joinpath("depth.png")

        with as_file(example_color_img) as example_color_src:
            rgb = cv2.cvtColor(cv2.imread(example_color_src.as_posix()), cv2.COLOR_BGR2RGB)
        with as_file(example_mask_img) as example_mask_src:
            mask = cv2.imread(example_mask_src.as_posix(), -1).astype(bool)
        with as_file(example_depth_img) as example_depth_src:
            depth = cv2.imread(example_depth_src.as_posix(), -1) / 1e3

    else:
        rgb = cv2.cvtColor(cv2.imread(config.src.color), cv2.COLOR_BGR2RGB) # 用于分析
        # 来自 YcbineoatReader.get_mask
        mask = cv2.imread(config.src.mask, -1).astype(bool)
        # 获取深度图
        depth = cv2.imread(config.src.depth, -1) / 1e3

    depth = depth_filter(depth)

    fpc = FoundationPoseClient(
        host = config.host,
        port = config.port
    )

    start_time = time.perf_counter()
    bbox_pose = fpc.infer(
        config.target_label,
        rgb, depth, mask, is_bbox_pose = True
    )
    use_time = time.perf_counter() - start_time
    print(f"use time: {use_time:.3f}s")
    print(bbox_pose)

    if config.is_draw_orin_pose:
        origin_pose = fpc.infer(
            config.target_label,
            rgb, depth, mask, is_bbox_pose = False
        )
    else: 
        origin_pose = None

    bbox_rgb = rgb.copy()
    bbox_min_xyz, bbox_max_xyz = fpc.get_bbox(config.target_label) 
    bbox_rgb = draw_mesh_axis_bbox(
        bbox_rgb, bbox_pose, fpc.cam_k, bbox_min_xyz=bbox_min_xyz, bbox_max_xyz = bbox_max_xyz, label = config.target_label, origin_pose = origin_pose
    )

    mask_rgb = rgb.copy()
    mask_rgb[mask == 0] = 0

    fig, axe = plt.subplot_mosaic([[0, 1], [2, 3]])
    axe[0].imshow(rgb)
    axe[1].imshow(depth)
    axe[2].imshow(mask_rgb)
    axe[3].imshow(bbox_rgb)

    plt.show()

