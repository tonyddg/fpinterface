'''
适用于多目标追踪的 Foundation Pose
'''

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FoundationPose.Utils import *
from FoundationPose.datareader import *
import itertools
from FoundationPose.learning.training.predict_score import *
from FoundationPose.learning.training.predict_pose_refine import *
import yaml

from typing import Dict, Optional
from dataclasses import dataclass

def guess_translation(depth, mask, K):
    '''
    来自 FoundationPose.guess_translation
    估计初始中心位置
    '''
    vs,us = np.where(mask>0)
    if len(us)==0:
        logging.warning(f'mask is all zero')
        return np.zeros((3))
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    valid = mask.astype(bool) & (depth>=0.001)
    if not valid.any():
        logging.warning(f"valid is empty")
        return np.zeros((3))

    zc = np.median(depth[valid])
    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1))*zc

    # if self.debug>=2:
    #     pcd = toOpen3dCloud(center.reshape(1,3))
    #     o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

    return center.reshape(3)

class MeshInfo:
    def __init__(self, mesh_path: str, is_mm_unit: bool = False, symmetry_tfs = None, min_n_views: int = 40, inplane_step: float = 60):
        '''
        * `mesh_path` 模型文件路径
        * `is_mm_unit` 是否使用 mm 单位 (如 Linemod 的模型)
        * `symmetry_tfs` 对称性信息见 Utils.py:symmetry_tfs_from_info
        * `min_n_views` 初始围绕目标的观测角度
        * `inplane_step` 相机旋转角度间隔

        生成假设姿态数为 (min_n_views + 2) * (360 / inplane_step)
        '''
        # force 参数用于保证 obj 模型具有纹理
        mesh = trimesh.load(mesh_path, force='mesh')
        if is_mm_unit:
            mesh.vertices *= 1e-3 # 对于 linemod 的 ply 模型需要转换单位为 m

        self.loadmesh(mesh, symmetry_tfs)
        self.make_rotation_grid(min_n_views = min_n_views, inplane_step = inplane_step)
        # 来自 FoundationPose 待确定的初始姿态
        self.pose_last: Optional[torch.Tensor] = None   # Used for tracking; per the centered mesh

        self.to_device()

    def loadmesh(self, mesh, symmetry_tfs=None):
        '''
        来自 FoundationPose.reset_object
        获取模型相关信息
        '''
        model_pts = mesh.vertices 
        model_normals = mesh.vertex_normals

        max_xyz = mesh.vertices.max(axis=0)
        min_xyz = mesh.vertices.min(axis=0)
        self.model_center = (min_xyz+max_xyz)/2
        if mesh is not None:
            self.mesh_ori = mesh.copy()
            mesh = mesh.copy()
            mesh.vertices = mesh.vertices - self.model_center.reshape(1,3)

        model_pts = mesh.vertices # 参数 model_pts 目的不明
        self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
        self.vox_size = max(self.diameter/20.0, 0.003)
        logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
        self.dist_bin = self.vox_size/2
        self.angle_bin = 20  # Deg
        pcd = toOpen3dCloud(model_pts, normals=model_normals)
        pcd = pcd.voxel_down_sample(self.vox_size)
        self.max_xyz = np.asarray(pcd.points).max(axis=0)
        self.min_xyz = np.asarray(pcd.points).min(axis=0)
        self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
        self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
        logging.info(f'self.pts:{self.pts.shape}')
        self.mesh_path = None
        self.mesh = mesh
        if self.mesh is not None:
            self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
            self.mesh.export(self.mesh_path)
            self.mesh_tensors = make_mesh_tensors(self.mesh)

        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4).float().cuda()[None]
        else:
            self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

        logging.info("mesh load done")

        # 获取模型旋转包容框的边界
        bbox_to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.origin_to_bbox = np.linalg.inv(bbox_to_origin)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    def make_rotation_grid(self, min_n_views: int = 40, inplane_step: float = 60):
        '''
        来自 FoundationPose.make_rotation_grid
        生成一系列初始姿态估计
        * `min_n_views` 初始围绕目标的观测角度
        * `inplane_step` 相机旋转角度间隔

        生成假设姿态数为 (min_n_views + 2) * (360 / inplane_step)
        '''
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        logging.info(f'smin_n_views: {min_n_views} and sample cam_in_obs:{cam_in_obs.shape}')
        rot_grid = []
        for i in range(len(cam_in_obs)):
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                cam_in_ob = cam_in_obs[i]
                R_inplane = euler_matrix(0,0,inplane_rot)
                cam_in_ob = cam_in_ob@R_inplane
                ob_in_cam = np.linalg.inv(cam_in_ob)
                rot_grid.append(ob_in_cam)

        rot_grid = np.asarray(rot_grid)
        logging.info(f"rot_grid:{rot_grid.shape}")
        # 根据角度与距离以及对称性剔除相近的估计姿态
        rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
        rot_grid = np.asarray(rot_grid)
        # logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
        self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
        # logging.info(f"self.rot_grid: {self.rot_grid.shape}")

    def to_device(self, s='cuda:0'):
        '''
        来自 FoundationPose.to_device  
        将所有数据转移到 GPU 中
        '''
        for k in self.__dict__:
            self.__dict__[k] = self.__dict__[k] # 含义不明
            if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
                # logging.info(f"Moving {k} to device {s}")
                self.__dict__[k] = self.__dict__[k].to(s)
        for k in self.mesh_tensors:
            # logging.info(f"Moving {k} to device {s}")
            self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
        # if self.refiner is not None:
        #     self.refiner.model.to(s)
        # if self.scorer is not None:
        #     self.scorer.model.to(s)
        # if self.glctx is not None:
        #     self.glctx = dr.RasterizeCudaContext(s)

    def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
        '''
        来自 FoundationPose.generate_random_pose_hypo
        @scene_pts: torch tensor (N,3)
        '''
        ob_in_cams = self.rot_grid.clone()
        center = guess_translation(depth=depth, mask=mask, K=K)
        ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
        return ob_in_cams

    def get_tf_to_centered_mesh(self):
        '''
        来自 FoundationPose.get_tf_to_centered_mesh  
        获取模型中心的平动变换
        '''
        tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
        tf_to_center[:3,3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
        return tf_to_center

    def draw_mesh_axis_bbox(
            self, img: np.ndarray, K: np.ndarray, bbox_pose: np.ndarray, label: str, 
            line_color: tuple = (0, 255, 0), line_sacle: float = 0.02, 
            font_color: tuple = (255, 0, 255), font_size: int = 3, is_input_rgb = True
                            ):
        '''
        * `img` RGB 格式图片
        '''
        vis = draw_posed_3d_box(K, img=img, ob_in_cam=bbox_pose, bbox=self.bbox, line_color = line_color)
        vis = draw_xyz_axis(img, ob_in_cam=bbox_pose, scale=line_sacle, K=K, thickness=3, transparency=0, is_input_rgb = is_input_rgb)
        
        center = project_3d_to_2d(np.array([0,0,0,1]), K, bbox_pose)
        vis = cv2.putText(vis, label, center, cv2.FONT_HERSHEY_PLAIN, font_size, font_color, 2, cv2.LINE_AA)

        return vis
        
class FPMultiTask:

    @dataclass
    class TraceTarget:
        mesh_info: MeshInfo
        pose_last: Optional[torch.Tensor] = None

    def __init__(
            self, 
            # debug = 0, 
            # debug_dir = '/home/bowen/debug/novel_pose_debug/'
        ):
        ### 来自 FoundationPose.__init__
        # self.ignore_normal_flip = True
        # self.debug = debug
        # self.debug_dir = debug_dir
        # os.makedirs(debug_dir, exist_ok=True)

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        # 渲染上下文直接使用 Cuda, 不考虑使用 OpenGL
        self.glctx = dr.RasterizeCudaContext()

        # 来自 FoundationPose.to_device, 不考虑使用其他设备
        self.scorer.model.to("cuda")
        self.refiner.model.to("cuda")

        ### 关于多目标追踪的目标字典 (使用字符串标记目标)
        self.target_dict: Dict[str, FPMultiTask.TraceTarget] = {}

    def add_target(
            self, 
            target_name: str, 
            mesh_info: MeshInfo
        ):
        '''
        添加追踪目标  
        * `target_name` 目标名称
        '''
        if target_name in self.target_dict:
            logging.info("target name exist")
            return
        self.target_dict[target_name] = FPMultiTask.TraceTarget(mesh_info)

    # def register(self, target_name: str, K, rgb, depth, ob_mask, iteration=5):
    #     '''Copmute pose from given pts to self.pcd
    #     @pts: (N,3) np array, downsampled scene points
    #     '''
    #     if target_name not in self.target_dict.keys():
    #         raise RuntimeError(f"Target: {target_name} not found")

    #     set_seed(0)
    #     # logging.info('Welcome')

    #     # if self.glctx is None:
    #     #     if glctx is None:
    #     #         self.glctx = dr.RasterizeCudaContext()
    #     #         # self.glctx = dr.RasterizeGLContext()
    #     #     else:
    #     #         self.glctx = glctx

    #     # 对深度图进行腐蚀与滤波
    #     depth = erode_depth(depth, radius=2, device='cuda')
    #     depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    #     # if self.debug>=2:
    #     #     xyz_map = depth2xyzmap(depth, K)
    #     #     valid = xyz_map[...,2]>=0.001
    #     #     pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
    #     #     o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
    #     #     cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    #     # 获取掩膜提取的目标信息
    #     normal_map = None
    #     valid = (depth>=0.001) & (ob_mask>0)
    #     if valid.sum()<4:
    #         logging.warning(f'valid too small, return')
    #         pose = np.eye(4)
    #         pose[:3,3] = guess_translation(depth=depth, mask=ob_mask, K=K)
    #         return pose

    #     # if self.debug>=2:
    #     #     imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
    #     #     cv2.imwrite(f'{self.debug_dir}/depth.png', (depth*1000).astype(np.uint16))
    #     #     valid = xyz_map[...,2]>=0.001
    #     #     pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
    #     #     o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply',pcd)

    #     # 保存观测图片信息与相机
    #     self.H, self.W = depth.shape[:2]
    #     self.K = K
    #     # DEBUG 作用未知, 暂时注释
    #     # self.ob_mask = ob_mask

    #     # 生成假设位置与姿态
    #     poses = self.target_dict[target_name].mesh_info.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
    #     poses = poses.data.cpu().numpy()
    #     # logging.info(f'numbers of hypo pose:{poses.shape}')
    #     # 可优化, 函数 generate_random_pose_hypo 已生成 guess_translation
    #     center = guess_translation(depth=depth, mask=ob_mask, K=K)

    #     poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    #     poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')

    #     # add_errs = self.compute_add_err_to_gt_pose(poses)
    #     # logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")

    #     # 优化假设
    #     xyz_map = depth2xyzmap(depth, K)
    #     poses, vis = self.refiner.predict(mesh=self.target_dict[target_name].mesh_info.mesh, mesh_tensors=self.target_dict[target_name].mesh_info.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.target_dict[target_name].mesh_info.diameter, iteration=iteration)# , get_vis=self.debug>=2)
    #     # if vis is not None:
    #     #     imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

    #     # 
    #     scores, vis = self.scorer.predict(mesh=self.target_dict[target_name].mesh_info.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.target_dict[target_name].mesh_info.mesh_tensors, glctx=self.glctx, mesh_diameter=self.target_dict[target_name].mesh_info.diameter) #, get_vis=self.debug>=2)
    #     # if vis is not None:
    #     #     imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

    #     # add_errs = self.compute_add_err_to_gt_pose(poses)
    #     # logging.info(f"final, add_errs min:{add_errs.min()}")

    #     # ids = torch.as_tensor(scores).argsort(descending=True)
    #     # logging.info(f'sort ids:{ids}')
    #     # scores = scores[ids]
    #     # poses = poses[ids]

    #     # logging.info(f'sorted scores:{scores}')

    #     # 获取最优姿态
    #     id_best = int(scores.argmax().item())

    #     # 保存上一姿态
    #     self.target_dict[target_name].pose_last = poses[id_best]
    #     # self.best_id = ids[0]

    #     # self.poses = poses
    #     # self.scores = scores

    #     return poses[id_best]

    def track_one(self, target_name: str, rgb, depth, K, iteration):
        if target_name not in self.target_dict.keys():
            raise RuntimeError(f"Target: {target_name} not found")
        
        pose_last = self.target_dict[target_name].pose_last
        if not isinstance(pose_last, torch.Tensor):
            raise RuntimeError(f"Target: {target_name} has not regist")
        logging.info("Welcome")

        depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')
        logging.info("depth processing done")

        xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

        pose, vis = self.refiner.predict(mesh=self.target_dict[target_name].mesh_info.mesh, mesh_tensors=self.target_dict[target_name].mesh_info.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=pose_last.reshape(1,4,4).data.cpu().numpy(), normal_map=None, xyz_map=xyz_map, mesh_diameter=self.target_dict[target_name].mesh_info.diameter, glctx=self.glctx, iteration=iteration)#, get_vis=self.debug>=2)
        logging.info("pose done")
        # if self.debug>=2:
        # extra['vis'] = vis

        # 将目标位姿平移到包容框中心, 可通过相加代替 ?
        self.target_dict[target_name].pose_last = pose[0]
        return pose[0]
    
    def get_target_raw_pose(self, target_name: str):
        '''
        获取基于模型原始坐标系的上一次预测结果
        '''
        if not isinstance(self.target_dict[target_name].pose_last, torch.Tensor):
            raise RuntimeError("Mesh has not register")

        # pose_last 为用于预测的姿态, 还需要 get_tf_to_centered_mesh 移动到包容盒中心
        return self.target_dict[target_name].pose_last.cpu().numpy()

    def get_target_origin_pose(self, target_name: str):
        '''
        获取基于模型原始坐标系的上一次预测结果
        '''
        if not isinstance(self.target_dict[target_name].pose_last, torch.Tensor):
            raise RuntimeError("Mesh has not register")

        # pose_last 为用于预测的姿态, 还需要 get_tf_to_centered_mesh 移动到包容盒中心
        return np.asarray((self.target_dict[target_name].pose_last @ self.target_dict[target_name].mesh_info.get_tf_to_centered_mesh()).cpu().numpy(), dtype = np.float64)

    def get_target_bbox_pose(self, target_name: str):
        '''
        获取基于模型包容盒坐标系的上一次预测结果
        '''
        if not isinstance(self.target_dict[target_name].pose_last, torch.Tensor):
            raise RuntimeError("Mesh has not register")

        return np.asarray(self.get_target_raw_pose(target_name) @ self.target_dict[target_name].mesh_info.origin_to_bbox, dtype = np.float64)

    def draw_last_pose(
            self, vis: np.ndarray, K: np.ndarray, target_name: str, 
            line_color: tuple = (0, 255, 0), line_sacle: float = 0.02, 
            font_color: tuple = (255, 0, 255), font_size: int = 3, is_input_rgb = False
            ):
        '''
        * `vis` RGB 格式的图片
        '''
        if not isinstance(self.target_dict[target_name].pose_last, torch.Tensor):
            raise RuntimeError("Mesh has not register")

        pose = self.get_target_bbox_pose(target_name)
        return self.target_dict[target_name].mesh_info.draw_mesh_axis_bbox(vis, K, pose, target_name, line_color, line_sacle, font_color, font_size, is_input_rgb)

    def register_time_analyze(
            self, 
            target_name: str, 
            K, 
            rgb, 
            depth, 
            ob_mask, 
            refine_iteration=5
        ):
        '''Copmute pose from given pts to self.pcd
        @pts: (N,3) np array, downsampled scene points
        '''
        if target_name not in self.target_dict.keys():
            raise RuntimeError(f"Target: {target_name} not found")

        # set_seed(0)

        time_init = time.perf_counter()
        time_start = 0
        time_consume = 0
        time_start = time.perf_counter()

        ### 对深度图进行腐蚀与滤波
        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')

        # 获取掩膜提取的目标信息
        normal_map = None
        valid = (depth>=0.001) & (ob_mask>0)
        if valid.sum()<4:
            logging.warning(f'valid too small, return')
            pose = np.eye(4)
            pose[:3,3] = guess_translation(depth=depth, mask=ob_mask, K=K)
            return pose

        time_consume = int((time.perf_counter() - time_start) * 1000)
        logging.info(f"depth erode and guess: {time_consume} ms")

        # 保存观测图片信息与相机
        self.H, self.W = depth.shape[:2]
        self.K = K

        ### 生成假设位置与姿态
        time_start = time.perf_counter()        

        poses = self.target_dict[target_name].mesh_info.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
        poses = poses.data.cpu().numpy()
        # logging.info(f'numbers of hypo pose:{poses.shape}')
        # 可优化, 函数 generate_random_pose_hypo 已生成 guess_translation
        center = guess_translation(depth=depth, mask=ob_mask, K=K)
        logging.info(f"guess translation: {center}")

        poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
        poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')

        time_consume = int((time.perf_counter() - time_start) * 1000)
        logging.info(f"give hypo: {time_consume} ms")

        ### 优化假设
        time_start = time.perf_counter()        

        xyz_map = depth2xyzmap(depth, K)
        poses, _ = self.refiner.predict(mesh=self.target_dict[target_name].mesh_info.mesh, mesh_tensors=self.target_dict[target_name].mesh_info.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.target_dict[target_name].mesh_info.diameter, iteration=refine_iteration)# , get_vis=self.debug>=2)

        time_consume = int((time.perf_counter() - time_start) * 1000)
        logging.info(f"refine hypo: {time_consume} ms")

        ### 评估假设分数
        time_start = time.perf_counter()        

        scores, _ = self.scorer.predict(mesh=self.target_dict[target_name].mesh_info.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.target_dict[target_name].mesh_info.mesh_tensors, glctx=self.glctx, mesh_diameter=self.target_dict[target_name].mesh_info.diameter) #, get_vis=self.debug>=2)
        
        # 获取最优姿态
        id_best = int(scores.argmax().item())

        time_consume = int((time.perf_counter() - time_start) * 1000)
        logging.info(f"best score: {scores[id_best]}")
        logging.info(f"score hypo: {time_consume} ms")

        # 保存上一姿态
        self.target_dict[target_name].pose_last = poses[id_best]

        time_consume = int((time.perf_counter() - time_init) * 1000)
        logging.info(f"regist total: {time_consume} ms")

        return poses[id_best]
