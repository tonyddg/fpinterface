from typing import Optional, Tuple
import numpy as np
import cv2

def to_homo(pts):
    '''
    来自 FoundationPose, 将二维或三维的点坐标转为齐次坐标
    
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    '''
    assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
    homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
    return homo

def project_3d_to_2d(pt,K,ob_in_cam):
    '''
    来自 FoundationPose, 将物体坐标系上的点投影到像平面, 得到像素坐标

    Args:
        pt (np.ndarray): 物体坐标系上点的 3 维齐次坐标 (第四个元素为 1)
        K (np.ndarray): 相机内参矩阵
        ob_in_cam (np.ndarray): 相机坐标系描述下的物体位姿 (外参)

    '''
    pt = pt.reshape(4,1)
    projected = K @ ((ob_in_cam@pt)[:3,:])
    projected = projected.reshape(-1)
    projected = projected/projected[2]
    return np.asarray(projected.reshape(-1)[:2].round(), dtype = np.int32)

def draw_xyz_axis(
        color: np.ndarray, 
        ob_in_cam: np.ndarray, 
        scale: float = 0.1, 
        K: np.ndarray = np.eye(3), 
        thickness: int = 3, 
        transparency: float = 0,
        is_input_rgb: bool = False,

        is_draw_axis_label: bool = True,
        label_size: int = 2,
    ):
    '''
    来自 FoundationPose, 绘制物体坐标系, 红色为 x 轴, 绿色为 y 轴, 蓝色为 z 轴

    @color: BGR
    '''
    if is_input_rgb:
        color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
    xx = np.array([1,0,0,1]).astype(float)
    yy = np.array([0,1,0,1]).astype(float)
    zz = np.array([0,0,1,1]).astype(float)
    xx[:3] = xx[:3]*scale
    yy[:3] = yy[:3]*scale
    zz[:3] = zz[:3]*scale
    origin = tuple(project_3d_to_2d(np.array([0,0,0,1]), K, ob_in_cam))
    xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
    yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
    zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
    line_type = cv2.LINE_AA
    arrow_len = 0
    tmp = color.copy()
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, xx, color=(0,0,255), thickness=thickness,line_type=line_type, tipLength=arrow_len)
    if is_draw_axis_label:
        tmp1 = cv2.putText(tmp1, "x", xx, cv2.FONT_HERSHEY_PLAIN, label_size, (0,0,255), 2, cv2.LINE_AA)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)

    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, yy, color=(0,255,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
    if is_draw_axis_label:
        tmp1 = cv2.putText(tmp1, "y", yy, cv2.FONT_HERSHEY_PLAIN, label_size, (0,255,0), 2, cv2.LINE_AA)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)

    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, zz, color=(255,0,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
    if is_draw_axis_label:
        tmp1 = cv2.putText(tmp1, "z", zz, cv2.FONT_HERSHEY_PLAIN, label_size, (255,0,0), 2, cv2.LINE_AA)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)

    tmp = tmp.astype(np.uint8)
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)

    return tmp

def draw_posed_3d_box(
        K: np.ndarray, 
        img: np.ndarray, 
        ob_in_cam: np.ndarray, 
        bbox_min_xyz: np.ndarray, 
        bbox_max_xyz: np.ndarray, 
        line_color: Tuple = (0,255,0), 
        linewidth: int = 2
    ):
    '''
    来自 FoundationPose, 绘制物体包容盒

    Revised from 6pack dataset/inference_dataset_nocs.py::projection
    @bbox: (2,3) min/max
    @line_color: RGB
    '''
    xmin, ymin, zmin = bbox_min_xyz
    xmax, ymax, zmax = bbox_max_xyz

    def draw_line3d(start,end,img):
        pts = np.stack((start,end),axis=0).reshape(-1,3)
        pts = (ob_in_cam@to_homo(pts).T).T[:,:3]   #(2,3)
        projected = (K@pts.T).T
        uv = np.round(projected[:,:2]/projected[:,2].reshape(-1,1)).astype(int)   #(2,2)
        img = cv2.line(img, uv[0].tolist(), uv[1].tolist(), color=line_color, thickness=linewidth, lineType=cv2.LINE_AA)
        return img

    for y in [ymin,ymax]:
        for z in [zmin,zmax]:
            start = np.array([xmin,y,z])
            end = start+np.array([xmax-xmin,0,0])
            img = draw_line3d(start,end,img)

    for x in [xmin,xmax]:
        for z in [zmin,zmax]:
            start = np.array([x,ymin,z])
            end = start+np.array([0,ymax-ymin,0])
            img = draw_line3d(start,end,img)

    for x in [xmin,xmax]:
        for y in [ymin,ymax]:
            start = np.array([x,y,zmin])
            end = start+np.array([0,0,zmax-zmin])
            img = draw_line3d(start,end,img)

    return img

def draw_mesh_axis_bbox(
        img: np.ndarray, 
        bbox_pose: np.ndarray,
        K: np.ndarray, 
        bbox_min_xyz: np.ndarray, 
        bbox_max_xyz: np.ndarray, 
        label: str, 
        origin_pose: Optional[np.ndarray] = None, 
        line_color: tuple = (0, 255, 0), 
        line_sacle: float = 0.05, 
        font_color: tuple = (255, 0, 255), 
        font_size: int = 2, 
        is_input_rgb = True
    ):
    '''
    * `img` RGB 格式图片
    '''
    vis = draw_posed_3d_box(K, img=img, ob_in_cam=bbox_pose, bbox_min_xyz=bbox_min_xyz, bbox_max_xyz = bbox_max_xyz, line_color = line_color)

    if origin_pose is None:
        use_pose = bbox_pose
    else:
        use_pose = origin_pose

    vis = draw_xyz_axis(img, ob_in_cam=use_pose, scale=line_sacle, K=K, thickness=3, transparency=0, is_input_rgb = is_input_rgb)
    
    center = project_3d_to_2d(np.array([0,0,0,1]), K, use_pose) + np.array([20, -20])
    vis = cv2.putText(vis, label, tuple(center), cv2.FONT_HERSHEY_PLAIN, font_size, font_color, 2, cv2.LINE_AA)

    return vis
