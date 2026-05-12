import numpy as np

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

def depth_filter(
    depth: np.ndarray,
    z_far: float = np.inf
):
    '''
    简单处理原始深度图, 去除无效深度值
    '''
    depth[(depth < 0.001) | (depth >= z_far)] = 0
    return depth

def quat_to_rotmat(q):
    """
    将 w 位于最后的四元数转为 3x3 旋转矩阵。

    参数:
        q: numpy array, shape (4,)
           四元数格式为 [x, y, z, w]

    返回:
        R: numpy array, shape (3, 3)
           旋转矩阵
    """
    q = np.asarray(q, dtype = np.float64)

    # 归一化，避免数值误差
    q = q / np.linalg.norm(q)

    x, y, z, w = q

    R = np.array([
        [1 - 2 * (y*y + z*z),     2 * (x*y - z*w),     2 * (x*z + y*w)],
        [    2 * (x*y + z*w), 1 - 2 * (x*x + z*z),     2 * (y*z - x*w)],
        [    2 * (x*z - y*w),     2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]
    ], dtype = np.float64)

    return R