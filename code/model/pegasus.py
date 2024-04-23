import math
from functools import partial

import torch
import torch.nn as nn
from flame.FLAME import FLAME, FLAME_lightning
from pytorch3d.ops import knn_points
from pytorch3d.renderer import (AlphaCompositor,
                                PerspectiveCameras,
                                PointsRasterizationSettings,
                                PointsRasterizer,
                                )
from pytorch3d.structures import Pointclouds
# from model.point_cloud import PointCloud, PointCloudSceneLatent, PointCloudSceneLatentThreeStages

from functorch import jacfwd, vmap, grad, jacrev

# from model.geometry_network import GeometryNetwork, GeometryNetworkSceneLatent
# from model.deformer_network import ForwardDeformer, ForwardDeformerSceneLatent, ForwardDeformerSceneLatentThreeStages, ForwardDeformerSceneLatentDeepThreeStages
# from model.texture_network import RenderingNetwork, RenderingNetworkSceneLatentThreeStages
from utils import general as utils
from einops import rearrange
import os, sys
face_parsing_path = os.path.abspath("../preprocess/face-parsing.PyTorch")
sys.path.append(face_parsing_path)
from model_parsing import BiSeNet
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2
from utils.rotation_converter import batch_rodrigues

print_flushed = partial(print, flush=True)

def chamfer_distance(A, B):
    """
    Compute the Chamfer Distance between two point sets A and B.

    Args:
    A: Tensor of shape [N, D] where N is the number of points in set A, and D is the dimension of each point.
    B: Tensor of shape [M, D] where M is the number of points in set B.

    Returns:
    A scalar tensor with the Chamfer Distance.
    """
    # Expand A and B for broadcasting
    A_expanded = A.unsqueeze(1)  # Shape: [N, 1, D]
    B_expanded = B.unsqueeze(0)  # Shape: [1, M, D]

    # Compute squared distances between all points in A and all points in B
    distances = torch.sum((A_expanded - B_expanded) ** 2, dim=2)

    # Compute minimum distances from A to B and B to A
    min_distances_A_to_B = torch.min(distances, dim=1)[0]  # Closest B point for each A point
    min_distances_B_to_A = torch.min(distances, dim=0)[0]  # Closest A point for each B point

    # Compute Chamfer Distance
    chamfer_dist = torch.mean(min_distances_A_to_B) + torch.mean(min_distances_B_to_A)

    return chamfer_dist

# class SceneLatentThreeStagesModel(nn.Module):
#     # NOTE singleGPU를 기반으로 만들었음. SingleGPU와의 차이점을 NOTE로 기술함.
#     # deform_cc를 추가하였음. 반드시 켜져있어야함. 
#     def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background, checkpoint_path, pcd_init=None):
#         super().__init__()
#         self.optimize_latent_code = conf.get_bool('train.optimize_latent_code')
#         self.optimize_scene_latent_code = conf.get_bool('train.optimize_scene_latent_code')

#         # FLAME_lightning
#         self.FLAMEServer = utils.get_class(conf.get_string('model.FLAME_class'))(conf=conf,
#                                                                                 flame_model_path='./flame/FLAME2020/generic_model.pkl', 
#                                                                                 lmk_embedding_path='./flame/FLAME2020/landmark_embedding.npy',
#                                                                                 n_shape=100,
#                                                                                 n_exp=50,
#                                                                                 shape_params=shape_params,
#                                                                                 canonical_expression=canonical_expression,
#                                                                                 canonical_pose=canonical_pose)                           # NOTE cuda 없앴음
        
#         # NOTE original code
#         # self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
#         #     self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
#         # self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)

#         self.prune_thresh = conf.get_float('model.prune_thresh', default=0.5)

#         # NOTE custom #########################
#         # scene latent를 위해 변형한 모델들이 들어감.
#         self.latent_code_dim = conf.get_int('model.latent_code_dim')
#         print('[DEBUG] latent_code_dim:', self.latent_code_dim)
#         # GeometryNetworkSceneLatent
#         self.geometry_network = utils.get_class(conf.get_string('model.geometry_class'))(optimize_latent_code=self.optimize_latent_code,
#                                                                                          optimize_scene_latent_code=self.optimize_scene_latent_code,
#                                                                                          latent_code_dim=self.latent_code_dim,
#                                                                                          **conf.get_config('model.geometry_network'))
#         # ForwardDeformerSceneLatentThreeStages
#         self.deformer_network = utils.get_class(conf.get_string('model.deformer_class'))(FLAMEServer=self.FLAMEServer,
#                                                                                         optimize_scene_latent_code=self.optimize_scene_latent_code,
#                                                                                         latent_code_dim=self.latent_code_dim,
#                                                                                         **conf.get_config('model.deformer_network'))
#         # RenderingNetworkSceneLatentThreeStages
#         self.rendering_network = utils.get_class(conf.get_string('model.rendering_class'))(optimize_scene_latent_code=self.optimize_scene_latent_code,
#                                                                                             latent_code_dim=self.latent_code_dim,
#                                                                                             **conf.get_config('model.rendering_network'))
#         #######################################

#         self.ghostbone = self.deformer_network.ghostbone

#         # NOTE custom #########################
#         if checkpoint_path is not None:
#             print('[DEBUG] init point cloud from previous checkpoint')
#             # n_init_point를 checkpoint으로부터 불러오기 위해..
#             data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
#             n_init_points = data['state_dict']['model.pc.points'].shape[0]
#             init_radius = data['state_dict']['model.radius'].item()
#         elif pcd_init is not None:
#             print('[DEBUG] init point cloud from meta learning')
#             n_init_points = pcd_init['n_init_points']
#             init_radius = pcd_init['init_radius']
#         else:
#             print('[DEBUG] init point cloud from scratch')
#             n_init_points = 400
#             init_radius = 0.5

#         # PointCloudSceneLatentThreeStages
#         self.pc = utils.get_class(conf.get_string('model.pointcloud_class'))(n_init_points=n_init_points,
#                                                                             init_radius=init_radius,
#                                                                             **conf.get_config('model.point_cloud'))     # NOTE .cuda() 없앴음
#         #######################################

#         n_points = self.pc.points.shape[0]
#         self.img_res = img_res
#         self.use_background = use_background
#         if self.use_background:
#             init_background = torch.zeros(img_res[0] * img_res[1], 3).float()                   # NOTE .cuda() 없앰
#             # self.background = nn.Parameter(init_background)                                   # NOTE singleGPU코드에서는 이렇게 작성했지만,
#             self.register_parameter('background', nn.Parameter(init_background))                # NOTE 이렇게 수정해서 혹시나하는 버그를 방지해보고자 한다.
#         else:
#             # self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()           # NOTE singleGPU코드에서는 이렇게 작성했지만,
#             self.register_buffer('background', torch.ones(img_res[0] * img_res[1], 3).float())  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

#         # NOTE original code
#         self.raster_settings = PointsRasterizationSettings(image_size=img_res[0],
#                                                            radius=self.pc.radius_factor * (0.75 ** math.log2(n_points / 100)),
#                                                            points_per_pixel=10)
#         self.register_buffer('radius', torch.tensor(self.raster_settings.radius))
        
#         # keypoint rasterizer is only for debugging camera intrinsics
#         self.raster_settings_kp = PointsRasterizationSettings(image_size=self.img_res[0],
#                                                               radius=0.007,
#                                                               points_per_pixel=1)

#         # NOTE ablation #########################################
#         self.enable_prune = conf.get_bool('train.enable_prune')

#         # self.visible_points = torch.zeros(n_points).bool().cuda()                             # NOTE singleGPU 코드에서는 이렇게 작성했지만,
#         if self.enable_prune:
#             if checkpoint_path is not None:
#                 # n_init_point를 checkpoint으로부터 불러오기 위해..
#                 data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
#                 visible_points = data['state_dict']['model.visible_points']                         # NOTE 이거 안해주면 visible이 0이 되어서 훈련이 안됨.
#             else:
#                 visible_points = torch.zeros(n_points).bool()
#             self.register_buffer('visible_points', visible_points)                                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.
#         # self.compositor = AlphaCompositor().cuda()                                            # NOTE singleGPU 코드에서는 이렇게 작성했지만,
#         self.compositor = AlphaCompositor()                                                   # NOTE cuda 할당이 자동으로 되도록 수정해본다.


#     def _compute_canonical_normals_and_feature_vectors(self, condition):
#         p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
#         p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
#         # randomly sample some points in the neighborhood within 0.25 distance

#         # eikonal_points = torch.cat([p, p + (torch.rand(p.shape).cuda() - 0.5) * 0.5], dim=0)                          # NOTE original code, eikonal_points.shape: [6400, 3]
#         eikonal_points = torch.cat([p, p + (torch.rand(p.shape, device=p.device) - 0.5) * 0.5], dim=0)                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

#         if self.optimize_scene_latent_code:
#             condition['scene_latent_gradient'] = torch.cat([condition['scene_latent'], condition['scene_latent']], dim=0).detach()

#         eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points.detach(), condition)
#         n_points = self.pc.points.shape[0] # 400
#         canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1) # 400, 3

#         p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
#         p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
#         geometry_output = self.geometry_network(p, condition)  # not using SDF to regularize point location, 3200, 4
#         sdf_values = geometry_output[:, 0]

#         feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector

#         if self.training and hasattr(self, "_output"):
#             self._output['sdf_values'] = sdf_values
#             self._output['grad_thetas'] = grad_thetas
#         if not self.training:
#             self._output['pnts_albedo'] = feature_vector

#         return canonical_normals, feature_vector # (400, 3), (400, 3) -> (400, 3) (3200, 3)

#     def _render(self, point_cloud, cameras, render_kp=False):
#         rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
#         fragments = rasterizer(point_cloud)
#         r = rasterizer.raster_settings.radius
#         dists2 = fragments.dists.permute(0, 3, 1, 2)
#         alphas = 1 - dists2 / (r * r)
#         images, weights = self.compositor(
#             fragments.idx.long().permute(0, 3, 1, 2),
#             alphas,
#             point_cloud.features_packed().permute(1, 0),
#         )
#         images = images.permute(0, 2, 3, 1)
#         weights = weights.permute(0, 2, 3, 1)
#         # batch_size, img_res, img_res, points_per_pixel
#         if self.enable_prune and self.training and not render_kp:
#             n_points = self.pc.points.shape[0]
#             # the first point for each pixel is visible
#             visible_points = fragments.idx.long()[..., 0].reshape(-1)
#             visible_points = visible_points[visible_points != -1]

#             visible_points = visible_points % n_points
#             self.visible_points[visible_points] = True

#             # points with weights larger than prune_thresh are visible
#             visible_points = fragments.idx.long().reshape(-1)[weights.reshape(-1) > self.prune_thresh]
#             visible_points = visible_points[visible_points != -1]

#             n_points = self.pc.points.shape[0]
#             visible_points = visible_points % n_points
#             self.visible_points[visible_points] = True

#         return images

#     def forward(self, input):
#         self._output = {}
#         intrinsics = input["intrinsics"].clone()
#         cam_pose = input["cam_pose"].clone()
#         R = cam_pose[:, :3, :3]
#         T = cam_pose[:, :3, 3]
#         flame_pose = input["flame_pose"]
#         expression = input["expression"]
#         batch_size = flame_pose.shape[0]
#         verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)

#         if self.ghostbone:
#             # identity transformation for body
#             # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
#             transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

#         # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
#         cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

#         # make sure the cameras focal length is logged too
#         focal_length = intrinsics[:, [0, 1], [0, 1]]
#         cameras.focal_length = focal_length
#         cameras.principal_point = cameras.get_principal_point()

#         n_points = self.pc.points.shape[0]
#         total_points = batch_size * n_points
#         # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

#         # NOTE custom #########################
#         if self.optimize_latent_code or self.optimize_scene_latent_code:
#             network_condition = dict()
#         else:
#             network_condition = None

#         if self.optimize_latent_code:
#             network_condition['latent'] = input["latent_code"] # [1, 32]
#         if self.optimize_scene_latent_code:
#             network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)   
#         ######################################

#         canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(network_condition)      # NOTE network_condition 추가

#         transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
#                                                                                                                      normals=canonical_normals,
#                                                                                                                      feature_vectors=feature_vector,
#                                                                                                                      pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
#                                                                                                                      betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
#                                                                                                                      transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
#                                                                                                                      cond=network_condition) # NOTE network_condition 추가

#         p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
#         p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
#         # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach(), cond=network_condition) # NOTE network_condition 추가
#         shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
#         # NOTE transformed_points: x_d
#         transformed_points = transformed_points.reshape(batch_size, n_points, 3)
#         rgb_points = rgb_points.reshape(batch_size, n_points, 3)
#         # point feature to rasterize and composite
#         features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)
#         if not self.training:
#             # render normal image
#             normal_begin_index = features.shape[-1]
#             normals_points = normals_points.reshape(batch_size, n_points, 3)
#             features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

#             shading_begin_index = features.shape[-1]
#             albedo_begin_index = features.shape[-1] + 3
#             albedo_points = torch.clamp(albedo_points, 0., 1.)
#             features = torch.cat([features, shading_points.reshape(batch_size, n_points, 3), albedo_points.reshape(batch_size, n_points, 3)], dim=-1)

#         transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

#         images = self._render(transformed_point_cloud, cameras)

#         if not self.training:
#             # render landmarks for easier camera format debugging
#             landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
#             transformed_verts = Pointclouds(points=landmarks2d, features=torch.ones_like(landmarks2d))
#             rendered_landmarks = self._render(transformed_verts, cameras, render_kp=True)

#         foreground_mask = images[..., 3].reshape(-1, 1)
#         if not self.use_background:
#             rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
#         else:
#             bkgd = torch.sigmoid(self.background * 100)
#             rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)

#         knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
#         flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
#         index_batch = index_batch.reshape(-1)

#         rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

#         # training outputs
#         output = {
#             'img_res': self.img_res,
#             'batch_size': batch_size,
#             'predicted_mask': foreground_mask,  # mask loss
#             'rgb_image': rgb_image,
#             'canonical_points': pnts_c_flame,
#             # for flame loss
#             'index_batch': index_batch,
#             'posedirs': posedirs,
#             'shapedirs': shapedirs,
#             'lbs_weights': lbs_weights,
#             'flame_posedirs': self.FLAMEServer.posedirs,
#             'flame_shapedirs': self.FLAMEServer.shapedirs,
#             'flame_lbs_weights': self.FLAMEServer.lbs_weights,
#         }

#         if not self.training:
#             output_testing = {
#                 'normal_image': images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
#                 'shading_image': images[..., shading_begin_index:shading_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
#                 'albedo_image': images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
#                 'rendered_landmarks': rendered_landmarks.reshape(-1, 3),
#                 'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
#                 'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
#                 'deformed_verts': verts.reshape(-1, 3),
#                 'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
#                 'pnts_normal_deformed': normals_points.reshape(batch_size, n_points, 3),
#                 #'pnts_normal_canonical': canonical_normals,
#             }
#             if self.deformer_network.deform_c:
#                 output_testing['unconstrained_canonical_points'] = self.pc.points
#             output.update(output_testing)
#         output.update(self._output)
#         if self.optimize_scene_latent_code:
#             output['scene_latent_code'] = input["scene_latent_code"]

#         return output


#     def get_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None):
#         if pnts_c.shape[0] == 0:
#             return pnts_c.detach()
#         pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

#         pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

#         total_points = betas.shape[0]
#         batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
#         n_points = pnts_c.shape[0]                                                  # NOTE 400
#         # pnts_c: n_points, 3
#         def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
#             pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
#             # NOTE custom #########################
#             condition = {}
#             condition['scene_latent'] = scene_latent
#             #######################################
#             shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
#             shapedirs = shapedirs.expand(batch_size, -1, -1)
#             posedirs = posedirs.expand(batch_size, -1, -1)
#             lbs_weights = lbs_weights.expand(batch_size, -1)
#             pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
#             pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights) # FLAME-based deformed
#             pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
#             return pnts_d, pnts_d

#         normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
#         betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
#         transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
#         pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
#         if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
#             scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
#             grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent)
#             # pnts_c: [400, 3], betas: [400, 1, 50], transformations: [400, 1, 6, 4, 4], pose_feature: [400, 1, 36], scene_latent: [400, 1, 32]
#         else:
#             grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

#         pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
#         grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

#         grads_inv = grads_batch.inverse()
#         normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
#         feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
#         # some relighting code for inference
#         # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
#         # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)
#         shading = self.rendering_network(normals_d, cond) # TODO 여기다가 condition을 추가하면 어떻게 될까?????
#         albedo = feature_vectors
#         rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
#         return pnts_d, rgb_vals, albedo, shading, normals_d
    


class SceneLatentThreeStagesModel(nn.Module):
    # NOTE singleGPU를 기반으로 만들었음. SingleGPU와의 차이점을 NOTE로 기술함.
    # deform_cc를 추가하였음. 반드시 켜져있어야함. 
    def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background, checkpoint_path, pcd_init=None):
        super().__init__()
        self.optimize_latent_code = conf.get_bool('train.optimize_latent_code')
        self.optimize_scene_latent_code = conf.get_bool('train.optimize_scene_latent_code')

        # FLAME_lightning
        self.FLAMEServer = utils.get_class(conf.get_string('model.FLAME_class'))(conf=conf,
                                                                                flame_model_path='./flame/FLAME2020/generic_model.pkl', 
                                                                                lmk_embedding_path='./flame/FLAME2020/landmark_embedding.npy',
                                                                                n_shape=100,
                                                                                n_exp=50,
                                                                                shape_params=shape_params,
                                                                                canonical_expression=canonical_expression,
                                                                                canonical_pose=canonical_pose)                           # NOTE cuda 없앴음
        
        # NOTE original code
        # self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
        #     self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        # self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)

        self.prune_thresh = conf.get_float('model.prune_thresh', default=0.5)

        # NOTE custom #########################
        # scene latent를 위해 변형한 모델들이 들어감.
        self.latent_code_dim = conf.get_int('model.latent_code_dim')
        print('[DEBUG] latent_code_dim:', self.latent_code_dim)
        # GeometryNetworkSceneLatent
        self.geometry_network = utils.get_class(conf.get_string('model.geometry_class'))(optimize_latent_code=self.optimize_latent_code,
                                                                                         optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                         latent_code_dim=self.latent_code_dim,
                                                                                         **conf.get_config('model.geometry_network'))
        # ForwardDeformerSceneLatentThreeStages
        self.deformer_network = utils.get_class(conf.get_string('model.deformer_class'))(FLAMEServer=self.FLAMEServer,
                                                                                        optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                        latent_code_dim=self.latent_code_dim,
                                                                                        **conf.get_config('model.deformer_network'))
        # RenderingNetworkSceneLatentThreeStages
        self.rendering_network = utils.get_class(conf.get_string('model.rendering_class'))(optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                            latent_code_dim=self.latent_code_dim,
                                                                                            **conf.get_config('model.rendering_network'))
        #######################################

        self.ghostbone = self.deformer_network.ghostbone

        # NOTE custom #########################
        self.normal = True if conf.get_float('loss.normal_weight') > 0 and self.training else False
        if checkpoint_path is not None:
            print('[DEBUG] init point cloud from previous checkpoint')
            # n_init_point를 checkpoint으로부터 불러오기 위해..
            data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            n_init_points = data['state_dict']['model.pc.points'].shape[0]
            init_radius = data['state_dict']['model.radius'].item()
        elif pcd_init is not None:
            print('[DEBUG] init point cloud from meta learning')
            n_init_points = pcd_init['n_init_points']
            init_radius = pcd_init['init_radius']
        else:
            print('[DEBUG] init point cloud from scratch')
            n_init_points = 400
            init_radius = 0.5

        # PointCloudSceneLatentThreeStages
        self.pc = utils.get_class(conf.get_string('model.pointcloud_class'))(n_init_points=n_init_points,
                                                                            init_radius=init_radius,
                                                                            **conf.get_config('model.point_cloud'))    # NOTE .cuda() 없앴음
        #######################################

        n_points = self.pc.points.shape[0]
        self.img_res = img_res
        self.use_background = use_background
        if self.use_background:
            init_background = torch.zeros(img_res[0] * img_res[1], 3).float()                   # NOTE .cuda() 없앰
            # self.background = nn.Parameter(init_background)                                   # NOTE singleGPU코드에서는 이렇게 작성했지만,
            self.register_parameter('background', nn.Parameter(init_background))                # NOTE 이렇게 수정해서 혹시나하는 버그를 방지해보고자 한다.
        else:
            # self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()           # NOTE singleGPU코드에서는 이렇게 작성했지만,
            self.register_buffer('background', torch.ones(img_res[0] * img_res[1], 3).float())  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # NOTE original code
        self.raster_settings = PointsRasterizationSettings(image_size=img_res[0],
                                                           radius=self.pc.radius_factor * (0.75 ** math.log2(n_points / 100)),
                                                           points_per_pixel=10,
                                                           bin_size=0)              # NOTE warning이 뜨길래 bin_size=0 추가함.
        self.register_buffer('radius', torch.tensor(self.raster_settings.radius))
        
        # keypoint rasterizer is only for debugging camera intrinsics
        self.raster_settings_kp = PointsRasterizationSettings(image_size=self.img_res[0],
                                                              radius=0.007,
                                                              points_per_pixel=1)  

        # NOTE ablation #########################################
        self.enable_prune = conf.get_bool('train.enable_prune')

        # self.visible_points = torch.zeros(n_points).bool().cuda()                             # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        if self.enable_prune:
            if checkpoint_path is not None:
                # n_init_point를 checkpoint으로부터 불러오기 위해..
                data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                visible_points = data['state_dict']['model.visible_points']                         # NOTE 이거 안해주면 visible이 0이 되어서 훈련이 안됨.
            else:
                visible_points = torch.zeros(n_points).bool()
            self.register_buffer('visible_points', visible_points)                                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.
        # self.compositor = AlphaCompositor().cuda()                                            # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        self.compositor = AlphaCompositor()                                                     # NOTE cuda 할당이 자동으로 되도록 수정해본다.


    def _compute_canonical_normals_and_feature_vectors(self, condition):
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
        # randomly sample some points in the neighborhood within 0.25 distance

        # eikonal_points = torch.cat([p, p + (torch.rand(p.shape).cuda() - 0.5) * 0.5], dim=0)                          # NOTE original code, eikonal_points.shape: [6400, 3]
        eikonal_points = torch.cat([p, p + (torch.rand(p.shape, device=p.device) - 0.5) * 0.5], dim=0)                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        if self.optimize_scene_latent_code:
            condition['scene_latent_gradient'] = torch.cat([condition['scene_latent'], condition['scene_latent']], dim=0).detach()

        eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points.detach(), condition)
        n_points = self.pc.points.shape[0] # 400
        canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1) # 400, 3

        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
        geometry_output = self.geometry_network(p, condition)  # not using SDF to regularize point location, 3200, 4
        sdf_values = geometry_output[:, 0]

        feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector

        if self.training and hasattr(self, "_output"):
            self._output['sdf_values'] = sdf_values
            self._output['grad_thetas'] = grad_thetas
        if not self.training:
            self._output['pnts_albedo'] = feature_vector

        return canonical_normals, feature_vector # (400, 3), (400, 3) -> (400, 3) (3200, 3)

    def _render(self, point_cloud, cameras, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)
        r = rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        images, weights = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            point_cloud.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
        # batch_size, img_res, img_res, points_per_pixel
        if self.enable_prune and self.training and not render_kp:
            n_points = self.pc.points.shape[0]
            # the first point for each pixel is visible
            visible_points = fragments.idx.long()[..., 0].reshape(-1)
            visible_points = visible_points[visible_points != -1]

            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

            # points with weights larger than prune_thresh are visible
            visible_points = fragments.idx.long().reshape(-1)[weights.reshape(-1) > self.prune_thresh]
            visible_points = visible_points[visible_points != -1]

            n_points = self.pc.points.shape[0]
            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

        return images

    def forward(self, input):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        batch_size = flame_pose.shape[0]
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)

        if self.ghostbone:
            # identity transformation for body
            # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
            transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
        cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        # NOTE custom #########################
        if self.optimize_latent_code or self.optimize_scene_latent_code:
            network_condition = dict()
        else:
            network_condition = None

        if self.optimize_latent_code:
            network_condition['latent'] = input["latent_code"] # [1, 32]
        if self.optimize_scene_latent_code:
            network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)   
        ######################################

        canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(network_condition)      # NOTE network_condition 추가

        transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                                                     normals=canonical_normals,
                                                                                                                     feature_vectors=feature_vector,
                                                                                                                     pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                                     cond=network_condition) # NOTE network_condition 추가

        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach(), cond=network_condition) # NOTE network_condition 추가
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        # NOTE transformed_points: x_d
        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        rgb_points = rgb_points.reshape(batch_size, n_points, 3)
        # point feature to rasterize and composite
        features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)        # NOTE [batch_size, num of PCD, num of RGB features (4)]
        
        if self.normal and self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)
            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)                # NOTE [batch_size, num of PCD, num of normal features (3)]

        if not self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)
            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

            shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1] + 3
            albedo_points = torch.clamp(albedo_points, 0., 1.)
            features = torch.cat([features, shading_points.reshape(batch_size, n_points, 3), albedo_points.reshape(batch_size, n_points, 3)], dim=-1)   # NOTE shading: [3], albedo: [3]

        transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

        images = self._render(transformed_point_cloud, cameras)

        if not self.training:
            # render landmarks for easier camera format debugging
            landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
            transformed_verts = Pointclouds(points=landmarks2d, features=torch.ones_like(landmarks2d))
            rendered_landmarks = self._render(transformed_verts, cameras, render_kp=True)

        foreground_mask = images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)

        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)

        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # training outputs
        output = {
            'img_res': self.img_res,
            'batch_size': batch_size,
            'predicted_mask': foreground_mask,  # mask loss
            'rgb_image': rgb_image,
            'canonical_points': pnts_c_flame,
            # for flame loss
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights,
        }

        if self.normal and self.training:
            output['normal_image'] = (images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask)).reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        if not self.training:
            output_testing = {
                'normal_image': images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'shading_image': images[..., shading_begin_index:shading_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'albedo_image': images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'rendered_landmarks': rendered_landmarks.reshape(-1, 3),
                'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
                'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
                'deformed_verts': verts.reshape(-1, 3),
                'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
                'pnts_normal_deformed': normals_points.reshape(batch_size, n_points, 3),
                #'pnts_normal_canonical': canonical_normals,
            }
            if self.deformer_network.deform_c:
                output_testing['unconstrained_canonical_points'] = self.pc.points
            output.update(output_testing)
        output.update(self._output)

        if self.optimize_scene_latent_code:
            output['scene_latent_code'] = input["scene_latent_code"]

        return output


    def get_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

        pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
        n_points = pnts_c.shape[0]                                                  # NOTE 400
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
            pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
            # NOTE custom #########################
            condition = {}
            condition['scene_latent'] = scene_latent
            #######################################
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
            shapedirs = shapedirs.expand(batch_size, -1, -1)
            posedirs = posedirs.expand(batch_size, -1, -1)
            lbs_weights = lbs_weights.expand(batch_size, -1)
            pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
            pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights) # FLAME-based deformed
            pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
            return pnts_d, pnts_d

        normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
            scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent)
            # pnts_c: [400, 3], betas: [400, 1, 50], transformations: [400, 1, 6, 4, 4], pose_feature: [400, 1, 36], scene_latent: [400, 1, 32]
        else:
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        grads_inv = grads_batch.inverse()
        normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)
        shading = self.rendering_network(normals_d, cond) # TODO 여기다가 condition을 추가하면 어떻게 될까?????
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
        return pnts_d, rgb_vals, albedo, shading, normals_d




class SceneLatentThreeStagesBlendingModel(nn.Module):
    # NOTE SceneLatentThreeStages모델을 기반으로 만들었음. beta cancel이 적용되어있고 target human도 함께 들어가도록 설계되어있다.
    # deform_cc를 추가하였음. 반드시 켜져있어야함.
    def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background, checkpoint_path, pcd_init=None):
        super().__init__()
        shape_params = None
        self.optimize_latent_code = conf.get_bool('train.optimize_latent_code')
        self.optimize_scene_latent_code = conf.get_bool('train.optimize_scene_latent_code')

        # FLAME_lightning
        self.FLAMEServer = utils.get_class(conf.get_string('model.FLAME_class'))(conf=conf,
                                                                                flame_model_path='./flame/FLAME2020/generic_model.pkl', 
                                                                                lmk_embedding_path='./flame/FLAME2020/landmark_embedding.npy',
                                                                                n_shape=100,
                                                                                n_exp=50,
                                                                                # shape_params=shape_params,                            # NOTE BetaCancel
                                                                                canonical_expression=canonical_expression,
                                                                                canonical_pose=canonical_pose)                           # NOTE cuda 없앴음
        
        # NOTE original code
        # self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
        #     self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        # self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)

        self.prune_thresh = conf.get_float('model.prune_thresh', default=0.5)

        # NOTE custom #########################
        # scene latent를 위해 변형한 모델들이 들어감.
        self.latent_code_dim = conf.get_int('model.latent_code_dim')
        print('[DEBUG] latent_code_dim:', self.latent_code_dim)
        # GeometryNetworkSceneLatent
        self.geometry_network = utils.get_class(conf.get_string('model.geometry_class'))(optimize_latent_code=self.optimize_latent_code,
                                                                                         optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                         latent_code_dim=self.latent_code_dim,
                                                                                         **conf.get_config('model.geometry_network'))
        # ForwardDeformerSceneLatentThreeStages
        self.deformer_network = utils.get_class(conf.get_string('model.deformer_class'))(FLAMEServer=self.FLAMEServer,
                                                                                        optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                        latent_code_dim=self.latent_code_dim,
                                                                                        **conf.get_config('model.deformer_network'))
        # RenderingNetworkSceneLatentThreeStages
        self.rendering_network = utils.get_class(conf.get_string('model.rendering_class'))(optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                            latent_code_dim=self.latent_code_dim,
                                                                                            **conf.get_config('model.rendering_network'))
        #######################################

        self.ghostbone = self.deformer_network.ghostbone

        # NOTE custom #########################
        # self.test_target_finetuning = conf.get_bool('test.target_finetuning')
        self.normal = True if conf.get_float('loss.normal_weight') > 0 and self.training else False
        self.target_training = conf.get_bool('train.target_training', default=False)
        # self.segment = True if conf.get_float('loss.segment_weight') > 0 else False
        if checkpoint_path is not None:
            print('[DEBUG] init point cloud from previous checkpoint')
            # n_init_point를 checkpoint으로부터 불러오기 위해..
            data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            n_init_points = data['state_dict']['model.pc.points'].shape[0]
            try:
                init_radius = data['state_dict']['model.pc.radius'].item()
            except:
                init_radius = data['state_dict']['model.radius'].item()
        elif pcd_init is not None:
            print('[DEBUG] init point cloud from meta learning')
            n_init_points = pcd_init['n_init_points']
            init_radius = pcd_init['init_radius']
        else:
            print('[DEBUG] init point cloud from scratch')
            n_init_points = 400
            init_radius = 0.5
            # init_radius = self.pc.radius_factor * (0.75 ** math.log2(n_points / 100))

        # PointCloudSceneLatentThreeStages
        self.pc = utils.get_class(conf.get_string('model.pointcloud_class'))(n_init_points=n_init_points,
                                                                            init_radius=init_radius,
                                                                            **conf.get_config('model.point_cloud'))    # NOTE .cuda() 없앴음
        #######################################

        n_points = self.pc.points.shape[0]
        self.img_res = img_res
        self.use_background = use_background
        if self.use_background:
            init_background = torch.zeros(img_res[0] * img_res[1], 3).float()                   # NOTE .cuda() 없앰
            # self.background = nn.Parameter(init_background)                                   # NOTE singleGPU코드에서는 이렇게 작성했지만,
            self.register_parameter('background', nn.Parameter(init_background))                # NOTE 이렇게 수정해서 혹시나하는 버그를 방지해보고자 한다.
        else:
            # self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()           # NOTE singleGPU코드에서는 이렇게 작성했지만,
            self.register_buffer('background', torch.ones(img_res[0] * img_res[1], 3).float())  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # NOTE original code
        self.raster_settings = PointsRasterizationSettings(image_size=img_res[0],
                                                           radius=init_radius,
                                                           points_per_pixel=10,
                                                           bin_size=0)              # NOTE warning이 뜨길래 bin_size=0 추가함.
        # self.register_buffer('radius', torch.tensor(self.raster_settings.radius))
        
        # keypoint rasterizer is only for debugging camera intrinsics
        self.raster_settings_kp = PointsRasterizationSettings(image_size=self.img_res[0],
                                                              radius=0.007,
                                                              points_per_pixel=1)  

        # NOTE ablation #########################################
        self.enable_prune = conf.get_bool('train.enable_prune')

        # self.visible_points = torch.zeros(n_points).bool().cuda()                             # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        if self.enable_prune:
            if checkpoint_path is not None:
                # n_init_point를 checkpoint으로부터 불러오기 위해..
                data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                visible_points = data['state_dict']['model.visible_points']                         # NOTE 이거 안해주면 visible이 0이 되어서 훈련이 안됨.
            else:
                visible_points = torch.zeros(n_points).bool()
            self.register_buffer('visible_points', visible_points)                                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.
        # self.compositor = AlphaCompositor().cuda()                                            # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        self.compositor = AlphaCompositor()                                                     # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        self.num_views = 100
        # NOTE 모든 view에 대해서 segment된 point indices를 다 저장한다.
        self.voting_table = torch.zeros((len(conf.get_list('dataset.train.sub_dir')), n_points, self.num_views))
        # NOTE voting_table의 마지막 view의 index를 지정하기 위해, 각 sub_dir이 몇번 나왔는지 세기 위해서 만들었다.
        self.count_sub_dirs = {}
        self.binary_segmentation = conf.get_bool('model.binary_segmentation', default=False)


    def _compute_canonical_normals_and_feature_vectors(self, p, condition):
        # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        # p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
        # randomly sample some points in the neighborhood within 0.25 distance

        # eikonal_points = torch.cat([p, p + (torch.rand(p.shape).cuda() - 0.5) * 0.5], dim=0)                          # NOTE original code, eikonal_points.shape: [6400, 3]
        eikonal_points = torch.cat([p, p + (torch.rand(p.shape, device=p.device) - 0.5) * 0.5], dim=0)                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        if self.optimize_scene_latent_code:
            condition['scene_latent_gradient'] = torch.cat([condition['scene_latent'], condition['scene_latent']], dim=0).detach()

        eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points.detach(), condition)
        n_points = self.pc.points.shape[0] # 400
        canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1) # 400, 3

        # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        # p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
        geometry_output = self.geometry_network(p, condition)  # not using SDF to regularize point location, 3200, 4
        sdf_values = geometry_output[:, 0]

        if not self.binary_segmentation:
            feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector
        else:
            feature_vector = torch.sigmoid(geometry_output[:, 1:-1] * 10)  # albedo vector
            binary_segment = torch.sigmoid(geometry_output[:, -1:])         # dim = 1
        # feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector
        # segment_probability = geometry_output[:, -10:]  # segment probability                       # NOTE segment를 위해 추가했음.

        if self.training and hasattr(self, "_output"):
            self._output['sdf_values'] = sdf_values
            self._output['grad_thetas'] = grad_thetas
        if self.binary_segmentation:
            self._output['binary_segment'] = binary_segment
        if not self.training:
            self._output['pnts_albedo'] = feature_vector

        return canonical_normals, feature_vector # (400, 3), (400, 3) -> (400, 3) (3200, 3)

    def _segment(self, point_cloud, cameras, mask, render_debug=True, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)

        # mask = mask > 127.5
        mask = mask > 0.5
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask.cpu().numpy().astype(np.uint8), kernel, iterations=1)
        mask = torch.tensor(mask, device=point_cloud.device).bool()
        on_pixels = fragments.idx[0, mask.reshape(1, 512, 512)[0]].long()

        unique_on_pixels = torch.unique(on_pixels[on_pixels >= 0])      
        # segmented_point = point[0, unique_on_pixels].unsqueeze(0)           # NOTE point.shape: [1, 108724, 3]

        # total = torch.tensor(list(range(point.shape[1])), device=unique_on_pixels.device)
        # mask = ~torch.isin(total, unique_on_pixels)
        # unsegmented_mask = total[mask]
        # unsegmented_point = point[0, unsegmented_mask].unsqueeze(0)

        if render_debug:
            red_color = torch.tensor([1, 0, 0], device=point_cloud.device)  # RGB for red
            for indices in on_pixels:
                for idx in indices:
                    # Check for valid index (i.e., not -1)
                    if idx >= 0:
                        point_cloud.features_packed()[idx, :3] = red_color

            fragments = rasterizer(point_cloud)
            r = rasterizer.raster_settings.radius
            dists2 = fragments.dists.permute(0, 3, 1, 2)
            alphas = 1 - dists2 / (r * r)
            images, weights = self.compositor(
                fragments.idx.long().permute(0, 3, 1, 2),
                alphas,
                point_cloud.features_packed().permute(1, 0),
            )
            images = images.permute(0, 2, 3, 1)
            return unique_on_pixels, images
        else:
            return unique_on_pixels
    
    def _render(self, point_cloud, cameras, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)
        r = rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        images, weights = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            point_cloud.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
        
        mask_hole = (fragments.idx.long()[..., 0].reshape(-1) == -1).reshape(self.img_res)
        if not render_kp:
            self._output['mask_hole'] = mask_hole

        # batch_size, img_res, img_res, points_per_pixel
        if self.enable_prune and self.training and not render_kp:
            n_points = self.pc.points.shape[0]
            # the first point for each pixel is visible
            visible_points = fragments.idx.long()[..., 0].reshape(-1)
            visible_points = visible_points[visible_points != -1]

            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

            # points with weights larger than prune_thresh are visible
            visible_points = fragments.idx.long().reshape(-1)[weights.reshape(-1) > self.prune_thresh]
            visible_points = visible_points[visible_points != -1]

            n_points = self.pc.points.shape[0]
            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

        return images

    # def face_parsing(self, img, label, device):
    #     n_classes = 19
    #     net = BiSeNet(n_classes=n_classes)
    #     net.to(device)
    #     save_pth = os.path.join(os.path.join(face_parsing_path, 'res', 'cp'), '79999_iter.pth')
    #     net.load_state_dict(torch.load(save_pth))
    #     net.eval()

    #     to_tensor = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #     ])

    #     label_mapping = {
    #         'skin': 1,
    #         'eyebrows': 2,       # 2,3
    #         'eyes': 4,           # 4,5 
    #         'ears': 7,           # 7,8
    #         'nose': 10,
    #         'mouth': 11,         # 11,12,13 (lips)
    #         'neck': 14,
    #         'necklace': 15,
    #         'cloth': 16,
    #         'hair': 17,
    #         'hat': 18,
    #     }

    #     with torch.no_grad():
    #         img = Image.fromarray(img)
    #         image = img.resize((512, 512), Image.BILINEAR)
    #         img = to_tensor(image)
    #         img = torch.unsqueeze(img, 0)
    #         img = img.to(device)
    #         out = net(img)[0]
    #         parsing = out.squeeze(0).cpu().numpy().argmax(0)

    #         condition = (parsing == label_mapping.get(label))
    #         locations = np.where(condition)
    #         mask_by_parsing = (condition).astype(np.uint8) * 255

    #         if locations == []:
    #             print('[WARN] No object detected...')
    #             return []
    #         else:
    #             return mask_by_parsing, torch.tensor(list(zip(locations[1], locations[0])))
            
    def canonical_mask_generator(self, input, what_to_render):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        shape_params = input["shape"]                               # NOTE BetaCancel
        batch_size = flame_pose.shape[0]
        # verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose, shape_params=shape_params)              # NOTE BetaCancel

        if self.ghostbone:
            # identity transformation for body
            # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
            transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
        cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        # NOTE custom #########################
        if self.optimize_latent_code or self.optimize_scene_latent_code:
            network_condition = dict()
        else:
            network_condition = None

        if self.optimize_latent_code:
            network_condition['latent'] = input["latent_code"] # [1, 32]
        
        if self.optimize_scene_latent_code:
            if self.test_target_finetuning and not self.training:
                if what_to_render == 'source':
                    network_condition['scene_latent'] = input['source_scene_latent_code'].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                elif what_to_render == 'target':
                    network_condition['scene_latent'] = input['target_scene_latent_code'].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
            else:
                network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]
            
        ######################################
        # if self.test_target_finetuning and not self.training and 'canonical_mask' not in input:
        
        # NOTE mask를 source human의 canonical space에서 찾아야한다. 가장 간단한 방법은 deform 되기 전에 것을 들고오면 된다. 
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        _, _, _, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        _, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        
        index_batch = index_batch.reshape(-1)
        gt_beta_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, :100]

        canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(p, network_condition)      # NOTE network_condition 추가

        flame_canonical_points, flame_canonical_rgb_points, _, _, _ = self.get_canonical_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                                            normals=canonical_normals,
                                                                                                            feature_vectors=feature_vector,
                                                                                                            pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                            betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                            transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                            cond=network_condition,
                                                                                                            shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                            gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가
        
        flame_canonical_points = flame_canonical_points.reshape(batch_size, n_points, 3)
        flame_canonical_rgb_points = flame_canonical_rgb_points.reshape(batch_size, n_points, 3)
        features = torch.cat([flame_canonical_rgb_points, torch.ones_like(flame_canonical_rgb_points[..., [0]])], dim=-1)           # NOTE [batch_size, num of PCD, num of RGB features (4)]
        flame_canonical_point_cloud = Pointclouds(points=flame_canonical_points, features=features)                                     # NOTE pytorch3d's pointcloud class.

        canonical_cameras = PerspectiveCameras(device=expression.device, R=R, T=torch.tensor([0, 0, 4]).unsqueeze(0), K=intrinsics)  
        focal_length = intrinsics[:, [0, 1], [0, 1]] # make sure the cameras focal length is logged too
        canonical_cameras.focal_length = focal_length
        canonical_cameras.principal_point = canonical_cameras.get_principal_point()

        images = self._render(flame_canonical_point_cloud, canonical_cameras)
        foreground_mask = images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)
        
        # Convert tensor to numpy and squeeze the batch dimension
        img_np = rgb_image.squeeze(0).detach().cpu().numpy()

        # Convert range from [0, 1] to [0, 255] if needed
        if img_np.max() <= 1:
            img_np = (img_np * 255).astype('uint8')

        # Swap RGB to BGR
        img_bgr = img_np[:, :, [2, 1, 0]]

        # Save the image
        cv2.imwrite('{}_output_image.png'.format(what_to_render), img_bgr)

        canonical_mask_filename = '{}_canonical_mask.png'.format(what_to_render)
        if not os.path.exists(canonical_mask_filename):
            mask_parsing, _ = self.face_parsing(img_bgr, input['target_category'], device=expression.device)
            cv2.imwrite(canonical_mask_filename, mask_parsing)
        else:
            mask_parsing = cv2.imread(canonical_mask_filename, cv2.IMREAD_GRAYSCALE)

        segment_mask, unsegment_mask, segmented_images = self._segment(flame_canonical_points, flame_canonical_point_cloud, canonical_cameras, mask=mask_parsing)

        foreground_mask = segmented_images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = segmented_images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = segmented_images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # Convert tensor to numpy and squeeze the batch dimension
        img_np = rgb_image.squeeze(0).detach().cpu().numpy()

        # Convert range from [0, 1] to [0, 255] if needed
        if img_np.max() <= 1:
            img_np = (img_np * 255).astype('uint8')

        # Swap RGB to BGR
        img_bgr = img_np[:, :, [2, 1, 0]]

        # Save the image
        cv2.imwrite('{}_segmented_images.png'.format(what_to_render), img_bgr)

        # del flame_canonical_points, flame_canonical_point_cloud, canonical_cameras, rgb_image, rgb_values, segmented_images, img_bgr
        # torch.cuda.empty_cache()
        return segment_mask, unsegment_mask

    def deformer_mask_generator(self, input, what_to_render='source'):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        shape_params = input["shape"]                               # NOTE BetaCancel
        batch_size = flame_pose.shape[0]
        # verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose, shape_params=shape_params)              # NOTE BetaCancel

        if self.ghostbone:
            # identity transformation for body
            # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
            transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
        cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        # NOTE custom #########################
        if self.optimize_latent_code or self.optimize_scene_latent_code:
            network_condition = dict()
        else:
            network_condition = None

        if self.optimize_latent_code:
            network_condition['latent'] = input["latent_code"] # [1, 32]
        
        if self.optimize_scene_latent_code:
            # if self.test_target_finetuning and not self.training:
            #     if what_to_render == 'source':
            #         network_condition['scene_latent'] = input['source_scene_latent_code'].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
            #     elif what_to_render == 'target':
            #         network_condition['scene_latent'] = input['target_scene_latent_code'].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
            # else:
            #     network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]
            # if self.test_target_finetuning and self.training:
            #     network_condition['scene_latent'] = input['scene_latent_code'].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
            network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]
            
        ######################################
        # NOTE mask를 source human의 canonical space에서 찾아야한다. 가장 간단한 방법은 deform 되기 전에 것을 들고오면 된다. 
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        _, _, _, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        _, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        
        index_batch = index_batch.reshape(-1)
        gt_beta_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, :100]

        canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(p, network_condition)      # NOTE network_condition 추가
        
        transformed_points, rgb_points, _, _, _ = self.get_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                normals=canonical_normals,
                                                                                feature_vectors=feature_vector,
                                                                                pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                cond=network_condition,
                                                                                shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가

        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        rgb_points = rgb_points.reshape(batch_size, n_points, 3)
        features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)        # NOTE [batch_size, num of PCD, num of RGB features (4)]

        transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

        segment_mask, segmented_images = self._segment(transformed_point_cloud, cameras, mask=input['mask_object'])

        foreground_mask = segmented_images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = segmented_images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = segmented_images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # Convert tensor to numpy and squeeze the batch dimension
        img_np = rgb_image.squeeze(0).detach().cpu().numpy()

        # Convert range from [0, 1] to [0, 255] if needed
        if img_np.max() <= 1:
            img_np = (img_np * 255).astype('uint8')

        # Swap RGB to BGR
        img_bgr = img_np[:, :, [2, 1, 0]]

        # Save the image
        cv2.imwrite('{}_segmented_images.png'.format(what_to_render), img_bgr)

        return segment_mask
    
    def forward(self, input):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        shape_params = input["shape"]                               # NOTE BetaCancel
        batch_size = flame_pose.shape[0]
        # verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose, shape_params=shape_params)              # NOTE BetaCancel

        if self.ghostbone:
            # identity transformation for body
            # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
            transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
        cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        # NOTE custom #########################
        if self.optimize_latent_code or self.optimize_scene_latent_code:
            network_condition = dict()
        else:
            network_condition = None

        if self.optimize_latent_code:
            network_condition['latent'] = input["latent_code"] # [1, 32]
        
        if self.optimize_scene_latent_code:
            # if self.test_target_finetuning and not self.training:
            #     segment_mask = input['segment_mask']
            #     expanded_target = input['target_scene_latent_code'].clone().expand(total_points, -1)
            #     expanded_source = input["source_scene_latent_code"].clone().expand(segment_mask.shape[0], -1)

            #     scene_latent_code = expanded_target.clone()
            #     scene_latent_code[segment_mask] = expanded_source
                
            #     network_condition['scene_latent'] = scene_latent_code
                
            #     unique_tensor = torch.unique(network_condition['scene_latent'], dim=0)
            #     assert unique_tensor.shape[0] != 1, 'source_scene_latent_code and target_scene_latent_code are same.'


            #     # expanded_target = input['target_scene_latent_code'].clone().expand(total_points, -1)
            #     # scene_latent_code = expanded_target.clone()
            #     # network_condition['scene_latent'] = scene_latent_code
            # else:
            network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]
            # network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]

        ######################################
        # NOTE shape blendshape를 FLAME에서 그대로 갖다쓰기 위해 수정한 코드
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)
        gt_beta_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, :100]

        canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(p, network_condition)      # NOTE network_condition 추가

        canonical_rendering = False
        if canonical_rendering:
            transformed_points, rgb_points, _, _, _ = self.get_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                    normals=canonical_normals,
                                                                                    feature_vectors=feature_vector,
                                                                                    pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                    betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                    transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                    cond=network_condition,
                                                                                    shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                    gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가

            transformed_points = transformed_points.reshape(batch_size, n_points, 3)
            rgb_points = rgb_points.reshape(batch_size, n_points, 3)
            features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)        # NOTE [batch_size, num of PCD, num of RGB features (4)]

            transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

            images = self._render(transformed_point_cloud, cameras)
            foreground_mask = images[..., 3].reshape(-1, 1)
            if not self.use_background:
                rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
            else:
                bkgd = torch.sigmoid(self.background * 100)
                rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
            rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

            # Convert tensor to numpy and squeeze the batch dimension
            img_np = rgb_image.squeeze(0).detach().cpu().numpy()

            # Convert range from [0, 1] to [0, 255] if needed
            if img_np.max() <= 1:
                img_np = (img_np * 255).astype('uint8')

            # Swap RGB to BGR
            img_bgr = img_np[:, :, [2, 1, 0]]

            # Save the image
            cv2.imwrite('output_image2.png', img_bgr)
            ######################################

        # canonical_rendering = False
        # if canonical_rendering:
        #     flame_canonical_points, flame_canonical_rgb_points, _, _, _ = self.get_canonical_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
        #                                                                                                         normals=canonical_normals,
        #                                                                                                         feature_vectors=feature_vector,
        #                                                                                                         pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
        #                                                                                                         betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
        #                                                                                                         transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
        #                                                                                                         cond=network_condition,
        #                                                                                                         shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
        #                                                                                                         gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가

        #     flame_canonical_points = flame_canonical_points.reshape(batch_size, n_points, 3)
        #     flame_canonical_rgb_points = flame_canonical_rgb_points.reshape(batch_size, n_points, 3)
        #     features = torch.cat([flame_canonical_rgb_points, torch.ones_like(flame_canonical_rgb_points[..., [0]])], dim=-1)           # NOTE [batch_size, num of PCD, num of RGB features (4)]
        #     flame_canonical_point_cloud = Pointclouds(points=flame_canonical_points, features=features)                                     # NOTE pytorch3d's pointcloud class.

        #     canonical_cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  
        #     focal_length = intrinsics[:, [0, 1], [0, 1]] # make sure the cameras focal length is logged too
        #     canonical_cameras.focal_length = focal_length
        #     canonical_cameras.principal_point = canonical_cameras.get_principal_point()

        #     images = self._render(flame_canonical_point_cloud, canonical_cameras)
        #     foreground_mask = images[..., 3].reshape(-1, 1)
        #     if not self.use_background:
        #         rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        #     else:
        #         bkgd = torch.sigmoid(self.background * 100)
        #         rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
        #     rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        #     # Convert tensor to numpy and squeeze the batch dimension
        #     img_np = rgb_image.squeeze(0).detach().cpu().numpy()

        #     # Convert range from [0, 1] to [0, 255] if needed
        #     if img_np.max() <= 1:
        #         img_np = (img_np * 255).astype('uint8')

        #     # Swap RGB to BGR
        #     img_bgr = img_np[:, :, [2, 1, 0]]

        #     # Save the image
        #     cv2.imwrite('output_image2.png', img_bgr)
        #     ######################################

        transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                                                     normals=canonical_normals,
                                                                                                                     feature_vectors=feature_vector,
                                                                                                                     pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                                     cond=network_condition,
                                                                                                                     shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가

        # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        # p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        # # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach(), cond=network_condition) # NOTE network_condition 추가
        # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가

        # NOTE transformed_points: x_d
        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        rgb_points = rgb_points.reshape(batch_size, n_points, 3)
        # point feature to rasterize and composite

        if 'middle_inference' in input:
            middle_inference = input['middle_inference']
            target_human_rgb_points = middle_inference['rgb_points']
            target_human_normals_points = middle_inference['normals_points']
            target_human_shading_points = middle_inference['shading_points']
            target_human_albedo_points = middle_inference['albedo_points']
            target_human_transformed_points = middle_inference['transformed_points']

        if not self.training and 'masked_point_cloud_indices' in input:
            input['masked_point_cloud_indices'] = (self._output['binary_segment'].squeeze() > 0.5).float()
            rgb_points_bak = rgb_points.clone()
            rgb_points = rgb_points[:, input['masked_point_cloud_indices'].bool(), :]
            if 'middle_inference' in input:
                rgb_points = torch.cat([rgb_points, target_human_rgb_points.to(rgb_points.device)], dim=1)

        features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)        # NOTE [batch_size, num of PCD, num of RGB features (4)]
        
        if self.normal and self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)
            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)                # NOTE [batch_size, num of PCD, num of normal features (3)]
        
        if self.training and self.binary_segmentation:
            # render segment image
            segment_begin_index = features.shape[-1]
            num_segment = self._output['binary_segment'].shape[-1]
            segments_points = self._output['binary_segment'].reshape(batch_size, n_points, num_segment)
            features = torch.cat([features, segments_points], dim=-1)                           # NOTE [batch_size, num of PCD, num of segment features (19)]

        if not self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)

            if 'masked_point_cloud_indices' in input:
                normals_points_bak = normals_points.clone()
                normals_points = normals_points[:, input['masked_point_cloud_indices'].bool(), :]
                if 'middle_inference' in input:
                    normals_points = torch.cat([normals_points, target_human_normals_points.to(normals_points.device)], dim=1)

            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

            shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1] + 3
            albedo_points = torch.clamp(albedo_points, 0., 1.)

            shading_points = shading_points.reshape(batch_size, n_points, 3)
            albedo_points = albedo_points.reshape(batch_size, n_points, 3)

            if 'masked_point_cloud_indices' in input:
                shading_points = shading_points[:, input['masked_point_cloud_indices'].bool(), :]
                albedo_points = albedo_points[:, input['masked_point_cloud_indices'].bool(), :]
                if 'middle_inference' in input:
                    shading_points = torch.cat([shading_points, target_human_shading_points.to(shading_points.device)], dim=1)
                    albedo_points = torch.cat([albedo_points, target_human_albedo_points.to(albedo_points.device)], dim=1)

            features = torch.cat([features, shading_points, albedo_points], dim=-1)   # NOTE shading: [3], albedo: [3]

        if not self.training and 'masked_point_cloud_indices' in input:
            transformed_points_bak = transformed_points.clone()
            transformed_points = transformed_points[:, input['masked_point_cloud_indices'].bool(), :]
        
            if 'middle_inference' in input:
                transformed_points = torch.cat([transformed_points, target_human_transformed_points.to(transformed_points.device)], dim=1)

        if 'blending_middle_inference' in input:
            middle_inference = {
                'rgb_points': rgb_points,
                'normals_points': normals_points,
                'shading_points': shading_points,
                'albedo_points': albedo_points,
                'transformed_points': transformed_points
            }
            return middle_inference
        
        transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

        if self.training and ('rank' in input) and (input['rank'] == 0) and (not self.target_training):
            segment_mask = self._segment(transformed_point_cloud, cameras, mask=input['mask_object'], render_debug=False)
            segment_mask = segment_mask.detach().cpu()
            if input['sub_dir'][0] in self.count_sub_dirs.keys():
                if self.count_sub_dirs[input['sub_dir'][0]] < self.num_views-1:
                    self.count_sub_dirs[input['sub_dir'][0]] += 1
                    self.voting_table[input['indices_tensor'], segment_mask, self.count_sub_dirs[input['sub_dir'][0]]] = 1
            else:
                self.count_sub_dirs[input['sub_dir'][0]] = 0
                self.voting_table[input['indices_tensor'], segment_mask, 0] = 1

        images = self._render(transformed_point_cloud, cameras)

        if not self.training:
            # render landmarks for easier camera format debugging
            landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
            transformed_verts = Pointclouds(points=landmarks2d, features=torch.ones_like(landmarks2d))
            rendered_landmarks = self._render(transformed_verts, cameras, render_kp=True)

        foreground_mask = images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)

        if not self.training and 'masked_point_cloud_indices' in input:
            rgb_points = rgb_points_bak
            transformed_points = transformed_points_bak
            normals_points = normals_points_bak
        # knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        # flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        # index_batch = index_batch.reshape(-1)

        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # training outputs
        output = {
            'img_res': self.img_res,
            'batch_size': batch_size,
            'predicted_mask': foreground_mask,  # mask loss
            'rgb_image': rgb_image,
            'canonical_points': pnts_c_flame,
            # for flame loss
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights,
        }

        if self.normal and self.training:
            output['normal_image'] = (images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask)).reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        if self.training and self.binary_segmentation:
            output['segment_image'] = images[..., segment_begin_index:segment_begin_index+num_segment]

        if not self.training:
            output_testing = {
                'normal_image': images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'shading_image': images[..., shading_begin_index:shading_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'albedo_image': images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'rendered_landmarks': rendered_landmarks.reshape(-1, 3),
                'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
                'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
                'deformed_verts': verts.reshape(-1, 3),
                'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
                'pnts_normal_deformed': normals_points.reshape(batch_size, n_points, 3),
                #'pnts_normal_canonical': canonical_normals,
            }
            # if self.segment:
            #     output_testing['segment_image'] = images[..., segment_begin_index:segment_begin_index+num_segment].reshape(-1, 10)

            if self.deformer_network.deform_c:
                output_testing['unconstrained_canonical_points'] = self.pc.points
            output.update(output_testing)
        output.update(self._output)

        if not self.training and 'masked_point_cloud_indices' in input:
            self._output['mask_hole'] = self._output['mask_hole'].reshape(-1).unsqueeze(0)

        if self.optimize_scene_latent_code and self.training:
            output['scene_latent_code'] = input["scene_latent_code"]

        return output


    def get_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None, shapes=None, gt_beta_shapedirs=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

        pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
        n_points = pnts_c.shape[0]                                                  # NOTE 400
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None, shapes=None, gt_beta_shapedirs=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
            pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
            # NOTE custom #########################
            condition = {}
            condition['scene_latent'] = scene_latent
            #######################################
            # shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)           
            shapedirs = shapedirs.expand(batch_size, -1, -1)
            posedirs = posedirs.expand(batch_size, -1, -1)
            lbs_weights = lbs_weights.expand(batch_size, -1)
            pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
            # beta_shapedirs = beta_shapedirs.expand(batch_size, -1, -1)                                                          # NOTE beta cancel
            beta_shapedirs = gt_beta_shapedirs.unsqueeze(0).expand(batch_size, -1, -1)
            pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, beta_shapedirs=beta_shapedirs, shapes=shapes) # FLAME-based deformed
            pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
            return pnts_d, pnts_d

        normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        shapes = shapes.reshape(batch_size, n_points, *shapes.shape[1:]).transpose(0, 1)                                             # NOTE beta cancel
        if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
            scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
            # pnts_c: [400, 3], betas: [400, 1, 50], transformations: [400, 1, 6, 4, 4], pose_feature: [400, 1, 36], scene_latent: [400, 1, 32]
        else:
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        grads_inv = grads_batch.inverse()
        normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)
        shading = self.rendering_network(normals_d, cond) # TODO 여기다가 condition을 추가하면 어떻게 될까?????
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
        return pnts_d, rgb_vals, albedo, shading, normals_d


    def get_canonical_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None, shapes=None, gt_beta_shapedirs=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

        pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
        n_points = pnts_c.shape[0]                                                  # NOTE 400
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None, shapes=None, gt_beta_shapedirs=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
            pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
            # NOTE custom #########################
            condition = {}
            condition['scene_latent'] = scene_latent
            #######################################
            # shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)           # NOTE pnts_c_flame: [1, 3]
            # shapedirs = shapedirs.expand(batch_size, -1, -1)
            # posedirs = posedirs.expand(batch_size, -1, -1)
            # lbs_weights = lbs_weights.expand(batch_size, -1)
            # pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
            # beta_shapedirs = gt_beta_shapedirs.unsqueeze(0).expand(batch_size, -1, -1)
            # pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, beta_shapedirs=beta_shapedirs, shapes=shapes) # FLAME-based deformed
            # pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
            # return pnts_d, pnts_d
            pnts_c_flame = pnts_c_flame.reshape(-1)
            return pnts_c_flame, pnts_c_flame

        normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        shapes = shapes.reshape(batch_size, n_points, *shapes.shape[1:]).transpose(0, 1)                                             # NOTE beta cancel
        if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
            scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
            # grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
            grads_batch, pnts_c = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
        else:
            # grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)
            grads_batch, pnts_c = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        # pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        pnts_c = pnts_c.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        # grads_inv = grads_batch.inverse()
        # normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)

        # shading = self.rendering_network(normals_d, cond) 
        shading = self.rendering_network(normals, cond) 
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
        # return pnts_d, rgb_vals, albedo, shading, normals_d
        return pnts_c, rgb_vals, albedo, shading, normals





class PEGASUSModel(nn.Module):
    # NOTE SceneLatentThreeStages모델을 기반으로 만들었음. beta cancel이 적용되어있고 target human도 함께 들어가도록 설계되어있다.
    # previous name: SceneLatentThreeStagesBlendingCUDAModel
    # deform_cc를 추가하였음. 반드시 켜져있어야함.
    # CUDA가 직접 할당됨. 이건 DDP랑 native pytorch에 쓰이겠금.
    def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background, checkpoint_path, latent_code_dim, pcd_init=None):
        super().__init__()
        shape_params = None
        self.optimize_latent_code = conf.get_bool('train.optimize_latent_code')
        self.optimize_scene_latent_code = conf.get_bool('train.optimize_scene_latent_code')

        # FLAME_lightning
        self.FLAMEServer = utils.get_class(conf.get_string('model.FLAME_class'))(conf=conf,
                                                                                flame_model_path='./flame/FLAME2020/generic_model.pkl', 
                                                                                lmk_embedding_path='./flame/FLAME2020/landmark_embedding.npy',
                                                                                n_shape=100,
                                                                                n_exp=50,
                                                                                # shape_params=shape_params,                            # NOTE BetaCancel
                                                                                canonical_expression=canonical_expression,
                                                                                canonical_pose=canonical_pose).cuda()                           # NOTE cuda 없앴음
        
        # NOTE original code
        # self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
        #     self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        # self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)

        self.prune_thresh = conf.get_float('model.prune_thresh', default=0.5)

        # NOTE custom #########################
        # scene latent를 위해 변형한 모델들이 들어감.
        # self.latent_code_dim = conf.get_int('model.latent_code_dim')
        # print('[DEBUG] latent_code_dim:', self.latent_code_dim)
        self.latent_code_dim = latent_code_dim
        # GeometryNetworkSceneLatent
        self.geometry_network = utils.get_class(conf.get_string('model.geometry_class'))(optimize_latent_code=self.optimize_latent_code,
                                                                                         optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                         latent_code_dim=self.latent_code_dim,
                                                                                         **conf.get_config('model.geometry_network'))
        # ForwardDeformerSceneLatentThreeStages
        self.deformer_network = utils.get_class(conf.get_string('model.deformer_class'))(FLAMEServer=self.FLAMEServer,
                                                                                        optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                        latent_code_dim=self.latent_code_dim,
                                                                                        **conf.get_config('model.deformer_network'))
        # RenderingNetworkSceneLatentThreeStages
        self.rendering_network = utils.get_class(conf.get_string('model.rendering_class'))(optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                            latent_code_dim=self.latent_code_dim,
                                                                                            **conf.get_config('model.rendering_network'))
        #######################################

        self.ghostbone = self.deformer_network.ghostbone

        # NOTE custom #########################
        # self.test_target_finetuning = conf.get_bool('test.target_finetuning')
        self.normal = True if conf.get_float('loss.normal_weight') > 0 and self.training else False
        self.target_training = conf.get_bool('train.target_training', default=False)
        # self.segment = True if conf.get_float('loss.segment_weight') > 0 else False
        if checkpoint_path is not None:
            print('[DEBUG] init point cloud from previous checkpoint')
            # n_init_point를 checkpoint으로부터 불러오기 위해..
            data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            n_init_points = data['state_dict']['model.pc.points'].shape[0]
            try:
                init_radius = data['state_dict']['model.pc.radius'].item()
            except:
                init_radius = data['state_dict']['model.radius'].item()
        elif pcd_init is not None:
            print('[DEBUG] init point cloud from meta learning')
            n_init_points = pcd_init['n_init_points']
            init_radius = pcd_init['init_radius']
        else:
            print('[DEBUG] init point cloud from scratch')
            n_init_points = 400
            init_radius = 0.5
            # init_radius = self.pc.radius_factor * (0.75 ** math.log2(n_points / 100))

        self.pc = utils.get_class(conf.get_string('model.pointcloud_class'))(n_init_points=400,
                                                                            init_radius=0.5,
                                                                            **conf.get_config('model.point_cloud')).cuda()    # NOTE .cuda() 없앴음
        #######################################

        n_points = self.pc.points.shape[0]
        self.img_res = img_res
        self.use_background = use_background
        # if self.use_background:
        #     init_background = torch.zeros(img_res[0] * img_res[1], 3).float().cuda()                   # NOTE .cuda() 없앰
        #     # self.background = nn.Parameter(init_background)                                   # NOTE singleGPU코드에서는 이렇게 작성했지만,
        #     self.register_parameter('background', nn.Parameter(init_background))                # NOTE 이렇게 수정해서 혹시나하는 버그를 방지해보고자 한다.
        # else:
        #     # self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()           # NOTE singleGPU코드에서는 이렇게 작성했지만,
        #     self.register_buffer('background', torch.ones(img_res[0] * img_res[1], 3).float())  # NOTE cuda 할당이 자동으로 되도록 수정해본다.
        if self.use_background:
            init_background = torch.zeros(img_res[0] * img_res[1], 3).float().cuda()
            self.background = nn.Parameter(init_background)
        else:
            self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()

        # # NOTE original code
        # self.raster_settings = PointsRasterizationSettings(image_size=img_res[0],
        #                                                    radius=init_radius,
        #                                                    points_per_pixel=10,
        #                                                    bin_size=0)              # NOTE warning이 뜨길래 bin_size=0 추가함.
        # # self.register_buffer('radius', torch.tensor(self.raster_settings.radius))
        
        # # keypoint rasterizer is only for debugging camera intrinsics
        # self.raster_settings_kp = PointsRasterizationSettings(image_size=self.img_res[0],
        #                                                       radius=0.007,
        #                                                       points_per_pixel=1)  
        self.raster_settings = PointsRasterizationSettings(
            image_size=img_res[0],
            radius=self.pc.radius_factor * (0.75 ** math.log2(n_points / 100)),
            points_per_pixel=10
        )
        # keypoint rasterizer is only for debugging camera intrinsics
        self.raster_settings_kp = PointsRasterizationSettings(
            image_size=self.img_res[0],
            radius=0.007,
            points_per_pixel=1
        )
        self.visible_points = torch.zeros(n_points).bool().cuda()
        self.compositor = AlphaCompositor().cuda()

        # # NOTE ablation #########################################
        self.enable_prune = conf.get_bool('train.enable_prune')

        # # self.visible_points = torch.zeros(n_points).bool().cuda()                             # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        # if self.enable_prune:
        #     if checkpoint_path is not None:
        #         # n_init_point를 checkpoint으로부터 불러오기 위해..
        #         data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        #         visible_points = data['state_dict']['model.visible_points']                         # NOTE 이거 안해주면 visible이 0이 되어서 훈련이 안됨.
        #     else:
        #         visible_points = torch.zeros(n_points).bool()
        #     self.register_buffer('visible_points', visible_points)                                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.
            
        # self.compositor = AlphaCompositor().cuda()                                                     # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        self.num_views = 100
        # NOTE 모든 view에 대해서 segment된 point indices를 다 저장한다.
        # self.voting_table = torch.zeros((len(conf.get_list('dataset.train.sub_dir')), n_points, self.num_views))
        # NOTE voting_table의 마지막 view의 index를 지정하기 위해, 각 sub_dir이 몇번 나왔는지 세기 위해서 만들었다.
        # self.count_sub_dirs = {}
        self.binary_segmentation = conf.get_bool('model.binary_segmentation', default=False)


    def _compute_canonical_normals_and_feature_vectors(self, p, condition):
        # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        # p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
        # randomly sample some points in the neighborhood within 0.25 distance

        # eikonal_points = torch.cat([p, p + (torch.rand(p.shape).cuda() - 0.5) * 0.5], dim=0)                          # NOTE original code, eikonal_points.shape: [6400, 3]
        eikonal_points = torch.cat([p, p + (torch.rand(p.shape, device=p.device) - 0.5) * 0.5], dim=0)                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        if self.optimize_scene_latent_code:
            condition['scene_latent_gradient'] = torch.cat([condition['scene_latent'], condition['scene_latent']], dim=0).detach()

        eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points.detach(), condition)
        n_points = self.pc.points.shape[0] # 400
        canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1) # 400, 3

        # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        # p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
        geometry_output = self.geometry_network(p, condition)  # not using SDF to regularize point location, 3200, 4
        sdf_values = geometry_output[:, 0]

        if not self.binary_segmentation:
            feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector
        else:
            feature_vector = torch.sigmoid(geometry_output[:, 1:-1] * 10)  # albedo vector
            binary_segment = geometry_output[:, -1:]         # dim = 1
        # feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector
        # segment_probability = geometry_output[:, -10:]  # segment probability                       # NOTE segment를 위해 추가했음.

        if self.training and hasattr(self, "_output"):
            self._output['sdf_values'] = sdf_values
            self._output['grad_thetas'] = grad_thetas
        if self.binary_segmentation:
            self._output['binary_segment'] = binary_segment
        if not self.training:
            self._output['pnts_albedo'] = feature_vector

        return canonical_normals, feature_vector # (400, 3), (400, 3) -> (400, 3) (3200, 3)

    def _segment(self, point_cloud, cameras, mask, kernel_size=0, render_debug=True, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)

        # mask = mask > 127.5
        mask = mask > 0.5
        if kernel_size > 0:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.erode(mask.cpu().numpy().astype(np.uint8), kernel, iterations=1)
        mask = torch.tensor(mask, device=point_cloud.device).bool()
        on_pixels = fragments.idx[0, mask.reshape(1, 512, 512)[0]].long()

        unique_on_pixels = torch.unique(on_pixels[on_pixels >= 0])      
        # segmented_point = point[0, unique_on_pixels].unsqueeze(0)           # NOTE point.shape: [1, 108724, 3]

        # total = torch.tensor(list(range(point.shape[1])), device=unique_on_pixels.device)
        # mask = ~torch.isin(total, unique_on_pixels)
        # unsegmented_mask = total[mask]
        # unsegmented_point = point[0, unsegmented_mask].unsqueeze(0)

        if render_debug:
            red_color = torch.tensor([1, 0, 0], device=point_cloud.device)  # RGB for red
            for indices in on_pixels:
                for idx in indices:
                    # Check for valid index (i.e., not -1)
                    if idx >= 0:
                        point_cloud.features_packed()[idx, :3] = red_color

            fragments = rasterizer(point_cloud)
            r = rasterizer.raster_settings.radius
            dists2 = fragments.dists.permute(0, 3, 1, 2)
            alphas = 1 - dists2 / (r * r)
            images, weights = self.compositor(
                fragments.idx.long().permute(0, 3, 1, 2),
                alphas,
                point_cloud.features_packed().permute(1, 0),
            )
            images = images.permute(0, 2, 3, 1)
            return unique_on_pixels, images
        else:
            return unique_on_pixels
    
    def _find_empty_pixels(self, point_cloud, cameras, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)
        r = rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        images, weights = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            point_cloud.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
        
        mask_hole = (fragments.idx.long()[..., 0].reshape(-1) == -1).reshape(self.img_res)
        return mask_hole

    def _render(self, point_cloud, cameras, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)
        r = rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        images, weights = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            point_cloud.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
        
        mask_hole = (fragments.idx.long()[..., 0].reshape(-1) == -1).reshape(self.img_res)
        if not render_kp:
            self._output['mask_hole'] = mask_hole

        # batch_size, img_res, img_res, points_per_pixel
        if self.enable_prune and self.training and not render_kp:
            n_points = self.pc.points.shape[0]
            # the first point for each pixel is visible
            visible_points = fragments.idx.long()[..., 0].reshape(-1)
            visible_points = visible_points[visible_points != -1]

            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

            # points with weights larger than prune_thresh are visible
            visible_points = fragments.idx.long().reshape(-1)[weights.reshape(-1) > self.prune_thresh]
            visible_points = visible_points[visible_points != -1]

            n_points = self.pc.points.shape[0]
            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

        return images

    # def face_parsing(self, img, label, device):
    #     n_classes = 19
    #     net = BiSeNet(n_classes=n_classes)
    #     net.to(device)
    #     save_pth = os.path.join(os.path.join(face_parsing_path, 'res', 'cp'), '79999_iter.pth')
    #     net.load_state_dict(torch.load(save_pth))
    #     net.eval()

    #     to_tensor = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #     ])

    #     label_mapping = {
    #         'skin': 1,
    #         'eyebrows': 2,       # 2,3
    #         'eyes': 4,           # 4,5 
    #         'ears': 7,           # 7,8
    #         'nose': 10,
    #         'mouth': 11,         # 11,12,13 (lips)
    #         'neck': 14,
    #         'necklace': 15,
    #         'cloth': 16,
    #         'hair': 17,
    #         'hat': 18,
    #     }

    #     with torch.no_grad():
    #         img = Image.fromarray(img)
    #         image = img.resize((512, 512), Image.BILINEAR)
    #         img = to_tensor(image)
    #         img = torch.unsqueeze(img, 0)
    #         img = img.to(device)
    #         out = net(img)[0]
    #         parsing = out.squeeze(0).cpu().numpy().argmax(0)

    #         condition = (parsing == label_mapping.get(label))
    #         locations = np.where(condition)
    #         mask_by_parsing = (condition).astype(np.uint8) * 255

    #         if locations == []:
    #             print('[WARN] No object detected...')
    #             return []
    #         else:
    #             return mask_by_parsing, torch.tensor(list(zip(locations[1], locations[0])))
            
    def canonical_mask_generator(self, input, what_to_render):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        shape_params = input["shape"]                               # NOTE BetaCancel
        batch_size = flame_pose.shape[0]
        # verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose, shape_params=shape_params)              # NOTE BetaCancel

        if self.ghostbone:
            # identity transformation for body
            # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
            transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
        cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        # NOTE custom #########################
        if self.optimize_latent_code or self.optimize_scene_latent_code:
            network_condition = dict()
        else:
            network_condition = None

        if self.optimize_latent_code:
            network_condition['latent'] = input["latent_code"] # [1, 32]
        
        if self.optimize_scene_latent_code:
            if self.test_target_finetuning and not self.training:
                if what_to_render == 'source':
                    network_condition['scene_latent'] = input['source_scene_latent_code'].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                elif what_to_render == 'target':
                    network_condition['scene_latent'] = input['target_scene_latent_code'].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
            else:
                network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]
            
        ######################################
        # if self.test_target_finetuning and not self.training and 'canonical_mask' not in input:
        
        # NOTE mask를 source human의 canonical space에서 찾아야한다. 가장 간단한 방법은 deform 되기 전에 것을 들고오면 된다. 
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        _, _, _, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        _, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        
        index_batch = index_batch.reshape(-1)
        gt_beta_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, :100]

        canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(p, network_condition)      # NOTE network_condition 추가

        flame_canonical_points, flame_canonical_rgb_points, _, _, _ = self.get_canonical_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                                            normals=canonical_normals,
                                                                                                            feature_vectors=feature_vector,
                                                                                                            pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                            betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                            transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                            cond=network_condition,
                                                                                                            shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                            gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가
        
        flame_canonical_points = flame_canonical_points.reshape(batch_size, n_points, 3)
        flame_canonical_rgb_points = flame_canonical_rgb_points.reshape(batch_size, n_points, 3)
        features = torch.cat([flame_canonical_rgb_points, torch.ones_like(flame_canonical_rgb_points[..., [0]])], dim=-1)           # NOTE [batch_size, num of PCD, num of RGB features (4)]
        flame_canonical_point_cloud = Pointclouds(points=flame_canonical_points, features=features)                                     # NOTE pytorch3d's pointcloud class.

        canonical_cameras = PerspectiveCameras(device=expression.device, R=R, T=torch.tensor([0, 0, 4]).unsqueeze(0), K=intrinsics)  
        focal_length = intrinsics[:, [0, 1], [0, 1]] # make sure the cameras focal length is logged too
        canonical_cameras.focal_length = focal_length
        canonical_cameras.principal_point = canonical_cameras.get_principal_point()

        images = self._render(flame_canonical_point_cloud, canonical_cameras)
        foreground_mask = images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)
        
        # Convert tensor to numpy and squeeze the batch dimension
        img_np = rgb_image.squeeze(0).detach().cpu().numpy()

        # Convert range from [0, 1] to [0, 255] if needed
        if img_np.max() <= 1:
            img_np = (img_np * 255).astype('uint8')

        # Swap RGB to BGR
        img_bgr = img_np[:, :, [2, 1, 0]]

        # Save the image
        cv2.imwrite('{}_output_image.png'.format(what_to_render), img_bgr)

        canonical_mask_filename = '{}_canonical_mask.png'.format(what_to_render)
        if not os.path.exists(canonical_mask_filename):
            mask_parsing, _ = self.face_parsing(img_bgr, input['target_category'], device=expression.device)
            cv2.imwrite(canonical_mask_filename, mask_parsing)
        else:
            mask_parsing = cv2.imread(canonical_mask_filename, cv2.IMREAD_GRAYSCALE)

        segment_mask, unsegment_mask, segmented_images = self._segment(flame_canonical_points, flame_canonical_point_cloud, canonical_cameras, mask=mask_parsing)

        foreground_mask = segmented_images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = segmented_images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = segmented_images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # Convert tensor to numpy and squeeze the batch dimension
        img_np = rgb_image.squeeze(0).detach().cpu().numpy()

        # Convert range from [0, 1] to [0, 255] if needed
        if img_np.max() <= 1:
            img_np = (img_np * 255).astype('uint8')

        # Swap RGB to BGR
        img_bgr = img_np[:, :, [2, 1, 0]]

        # Save the image
        cv2.imwrite('{}_segmented_images.png'.format(what_to_render), img_bgr)

        # del flame_canonical_points, flame_canonical_point_cloud, canonical_cameras, rgb_image, rgb_values, segmented_images, img_bgr
        # torch.cuda.empty_cache()
        return segment_mask, unsegment_mask

    def deformer_mask_generator(self, input, what_to_render='source'):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        shape_params = input["shape"]                               # NOTE BetaCancel
        batch_size = flame_pose.shape[0]
        # verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose, shape_params=shape_params)              # NOTE BetaCancel

        if self.ghostbone:
            # identity transformation for body
            # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
            transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
        cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        # NOTE custom #########################
        if self.optimize_latent_code or self.optimize_scene_latent_code:
            network_condition = dict()
        else:
            network_condition = None

        if self.optimize_latent_code:
            network_condition['latent'] = input["latent_code"] # [1, 32]
        
        if self.optimize_scene_latent_code:
            # if self.test_target_finetuning and not self.training:
            #     if what_to_render == 'source':
            #         network_condition['scene_latent'] = input['source_scene_latent_code'].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
            #     elif what_to_render == 'target':
            #         network_condition['scene_latent'] = input['target_scene_latent_code'].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
            # else:
            #     network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]
            # if self.test_target_finetuning and self.training:
            #     network_condition['scene_latent'] = input['scene_latent_code'].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
            network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]
            
        ######################################
        # NOTE mask를 source human의 canonical space에서 찾아야한다. 가장 간단한 방법은 deform 되기 전에 것을 들고오면 된다. 
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        _, _, _, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        _, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        
        index_batch = index_batch.reshape(-1)
        gt_beta_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, :100]

        canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(p, network_condition)      # NOTE network_condition 추가
        
        transformed_points, rgb_points, _, _, _ = self.get_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                normals=canonical_normals,
                                                                                feature_vectors=feature_vector,
                                                                                pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                cond=network_condition,
                                                                                shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가

        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        rgb_points = rgb_points.reshape(batch_size, n_points, 3)
        features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)        # NOTE [batch_size, num of PCD, num of RGB features (4)]

        transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

        segment_mask, segmented_images = self._segment(transformed_point_cloud, cameras, mask=input['mask_object'])

        foreground_mask = segmented_images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = segmented_images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = segmented_images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # Convert tensor to numpy and squeeze the batch dimension
        img_np = rgb_image.squeeze(0).detach().cpu().numpy()

        # Convert range from [0, 1] to [0, 255] if needed
        if img_np.max() <= 1:
            img_np = (img_np * 255).astype('uint8')

        # Swap RGB to BGR
        img_bgr = img_np[:, :, [2, 1, 0]]

        # Save the image
        cv2.imwrite('{}_segmented_images.png'.format(what_to_render), img_bgr)

        return segment_mask
    
    def densify_point_cloud(self, verts, n):
        # verts: [1, 5023, 3]
        # Calculate pair-wise distance between points
        # PyTorch doesn't have a built-in for pairwise distances, we have to do it manually
        # Expand verts to [5023, 1, 3] and [1, 5023, 3] to get a tensor of shape [5023, 5023, 3]
        # where we can subtract and find distances
        diff = verts.expand(verts.size(1), verts.size(1), 3) - verts.transpose(0, 1).expand(verts.size(1), verts.size(1), 3)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-9)  # [5023, 5023] Add a small value to prevent NaN gradients

        # Set diagonal to infinity to ignore self-distance
        dist.fill_diagonal_(float('inf'))

        # Find the indices of the closest points
        dist_sorted, idx = dist.sort(dim=1)

        # For each point, find the n closest points, then calculate the midpoints
        midpoints = []
        for i in range(1, n + 1):
            # Take the average between the point and its ith nearest neighbor
            midpoints.append((verts.squeeze() + verts[0, idx[:, i]]) / 2)

        # Concatenate the original vertices with the midpoints along the point dimension
        new_verts = torch.cat([verts.squeeze()] + midpoints, dim=0).unsqueeze(0)
        
        return new_verts

    def forward(self, input):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        shape_params = input["shape"]                               # NOTE BetaCancel
        batch_size = flame_pose.shape[0]
        # verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose, shape_params=shape_params)              # NOTE BetaCancel

        if self.ghostbone:
            # identity transformation for body
            # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
            transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
        cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        # NOTE custom #########################
        if self.optimize_latent_code or self.optimize_scene_latent_code:
            network_condition = dict()
        else:
            network_condition = None

        if self.optimize_latent_code:
            network_condition['latent'] = input["latent_code"] # [1, 32]
        
        if self.optimize_scene_latent_code:
            # if self.test_target_finetuning and not self.training:
            #     segment_mask = input['segment_mask']
            #     expanded_target = input['target_scene_latent_code'].clone().expand(total_points, -1)
            #     expanded_source = input["source_scene_latent_code"].clone().expand(segment_mask.shape[0], -1)

            #     scene_latent_code = expanded_target.clone()
            #     scene_latent_code[segment_mask] = expanded_source
                
            #     network_condition['scene_latent'] = scene_latent_code
                
            #     unique_tensor = torch.unique(network_condition['scene_latent'], dim=0)
            #     assert unique_tensor.shape[0] != 1, 'source_scene_latent_code and target_scene_latent_code are same.'


            #     # expanded_target = input['target_scene_latent_code'].clone().expand(total_points, -1)
            #     # scene_latent_code = expanded_target.clone()
            #     # network_condition['scene_latent'] = scene_latent_code
            # else:
            network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]
            # network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]

            if 'mask_identity' in input:    # 바뀌는 부분을 제외한 나머지 부분에 대한 latent code는 원래 부분에서 갖고오는 코드이다.
                mask_identity = input['mask_identity']
                
                scene_latent = network_condition['scene_latent'].clone()
                scene_latent_default = input["scene_latent_code_default"].clone()       # source latent codes
                scene_latent[~mask_identity] = scene_latent_default.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)[~mask_identity].clone()

                network_condition['scene_latent'] = scene_latent

        # NOTE shape blendshape를 FLAME에서 그대로 갖다쓰기 위해 수정한 코드
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가

        ######################################
        if 'middle_inference' in input:
            middle_inference = input['middle_inference']
            target_human_rgb_points = middle_inference['rgb_points']
            target_human_normals_points = middle_inference['normals_points']
            target_human_shading_points = middle_inference['shading_points']
            target_human_albedo_points = middle_inference['albedo_points']
            target_human_transformed_points = middle_inference['transformed_points']
            if 'category' in input and input['category'] == 'hair':
                target_human_flame_transformed_points = middle_inference['flame_transformed_points']            # NOTE target human의 머리에 해당하는 부분이다.
                target_human_landmarks3d = middle_inference['landmarks3d']
                # target_human_flame_canonical_points = middle_inference['flame_canonical_points'].squeeze(0).cuda()
                # _, index_nn, _ = knn_points(target_human_flame_canonical_points.unsqueeze(0), pnts_c_flame.unsqueeze(0), K=200, return_nn=True) # NOTE pnts_c_flame에 대해 target_human_flame_canonical_points에서 K개의 가장 가까운 이웃.
                # masked_point_cloud_indices_by_target_human_flame_canonical_points = torch.zeros(pnts_c_flame.shape[0]).cuda()
                # masked_point_cloud_indices_by_target_human_flame_canonical_points[torch.unique(index_nn)] = 1.0

                # nearest_neightbor_sh_db_points = nearest_neightbor_sh_db_points.squeeze()
                # pnts_c_flame = torch.cat([pnts_c_flame, nearest_neightbor_sh_db_points], dim=0)
                # n_points = pnts_c_flame.shape[0]
                # total_points = batch_size * n_points
                # network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                
        #######################################

        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)
        gt_beta_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, :100]        # NOTE FLAME의 beta blendshapes basis를 FLAME canonical space의 beta blendshaps basis로 바꿔주는 과정.

        canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(p, network_condition)      # NOTE network_condition 추가

        if 'generate_mask_identity' in input:
            mask_attributes = (torch.sigmoid(self._output['binary_segment']).squeeze() > 0.5).float().bool()
            mask_identity = (torch.sigmoid(self._output['binary_segment']).squeeze() < 0.5).float().bool()

            return mask_attributes # mask_identity
            # # shape blendshape를 FLAME에서 그대로 갖다쓰기 위해 수정한 코드
            # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
            # # 우선 bak_scene_latent를 백업해두고
            # bak_scene_latent = network_condition['scene_latent'].clone()
            # # default identity에 대해서 scene latent를 만들어준다.
            # scene_latent_default = input["scene_latent_code_default"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]
            # # 이제 교체를 해주고
            # network_condition['scene_latent'][mask_identity] = scene_latent_default[mask_identity]
            # # 통과시켜서 위치 좌표를 preserving해준다.
            # p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
            # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가

            # knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
            # flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
            # index_batch = index_batch.reshape(-1)
            # gt_beta_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, :100]        # NOTE FLAME의 beta blendshapes basis를 FLAME canonical space의 beta blendshaps basis로 바꿔주는 과정.

            # canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(p, network_condition)      # NOTE network_condition 추가


        generic_pcd_canonical_rendering = False
        specific_pcd_canonical_rendering = False
        specific_canonical_rendering = False
        flame_canonical_rendering = False
        deformed_rendering = False

        def save_image(image_tensor, filename):
            # Convert tensor to numpy and squeeze the batch dimension
            img_np = image_tensor.squeeze(0).detach().cpu().numpy()

            # Convert range from [0, 1] to [0, 255] if needed
            if img_np.max() <= 1:
                img_np = (img_np * 255).astype('uint8')

            # Swap RGB to BGR
            img_bgr = img_np[:, :, [2, 1, 0]]

            kernel = np.ones((3, 3), np.uint8)

            img_bgr = cv2.erode(img_bgr, kernel, iterations=1)            # NOTE 흰색 노이즈를 제거
            img_bgr = cv2.dilate(img_bgr, kernel, iterations=1)           # NOTE 구멍이나 간격을 메움.
            # Save the image
            cv2.imwrite('{}.png'.format(filename), img_bgr)

        if generic_pcd_canonical_rendering: 
            generic_canonical_points = self.pc.points.detach().reshape(batch_size, n_points, 3)
            sample_size = 100000
            sampled_indices = torch.randperm(generic_canonical_points.size(1))[:sample_size]
            generic_canonical_points = generic_canonical_points[:, sampled_indices, :]

            # dark blue color torch.ones_like(generic_canonical_points) * torch.tensor([0.529, 0.808, 0.922]).unsqueeze(0).unsqueeze(0).cuda()
            generic_canonical_rgb_points = torch.ones_like(generic_canonical_points) * torch.tensor([0.0, 0.0, 0.5]).unsqueeze(0).unsqueeze(0).cuda()
            features = torch.cat([generic_canonical_rgb_points, torch.ones_like(generic_canonical_rgb_points[..., [0]])], dim=-1)           # NOTE [batch_size, num of PCD, num of RGB features (4)]

            generic_canonical_point_cloud = Pointclouds(points=generic_canonical_points, features=features)                                     # NOTE pytorch3d's pointcloud class.

            canonical_cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  
            focal_length = intrinsics[:, [0, 1], [0, 1]] # make sure the cameras focal length is logged too
            canonical_cameras.focal_length = focal_length
            canonical_cameras.principal_point = canonical_cameras.get_principal_point()

            images = self._render(generic_canonical_point_cloud, canonical_cameras)
            foreground_mask = images[..., 3].reshape(-1, 1)
            if not self.use_background:
                rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
            else:
                bkgd = torch.sigmoid(self.background * 100)
                rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
            rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

            save_image(rgb_image, 'generic_pcd_canonical_space_rgb')

        if specific_pcd_canonical_rendering: 
            specific_canonical_points = p.reshape(batch_size, n_points, 3)
            sample_size = 100000
            sampled_indices = torch.randperm(specific_canonical_points.size(1))[:sample_size]
            specific_canonical_points = specific_canonical_points[:, sampled_indices, :]

            # dark blue color torch.ones_like(generic_canonical_points) * torch.tensor([0.529, 0.808, 0.922]).unsqueeze(0).unsqueeze(0).cuda()
            specific_canonical_rgb_points = torch.ones_like(specific_canonical_points) * torch.tensor([0.0, 0.0, 0.5]).unsqueeze(0).unsqueeze(0).cuda()
            features = torch.cat([specific_canonical_rgb_points, torch.ones_like(specific_canonical_rgb_points[..., [0]])], dim=-1)           # NOTE [batch_size, num of PCD, num of RGB features (4)]

            specific_canonical_point_cloud = Pointclouds(points=specific_canonical_points, features=features)                                     # NOTE pytorch3d's pointcloud class.

            canonical_cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  
            focal_length = intrinsics[:, [0, 1], [0, 1]] # make sure the cameras focal length is logged too
            canonical_cameras.focal_length = focal_length
            canonical_cameras.principal_point = canonical_cameras.get_principal_point()

            images = self._render(specific_canonical_point_cloud, canonical_cameras)
            foreground_mask = images[..., 3].reshape(-1, 1)
            if not self.use_background:
                rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
            else:
                bkgd = torch.sigmoid(self.background * 100)
                rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
            rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

            save_image(rgb_image, 'specific_pcd_canonical_space_rgb')

        if specific_canonical_rendering:     # 
            specific_canonical_points, specific_canonical_rgb_points, specific_canonical_albedo_points, _, specific_canonical_normal = self.get_specific_canonical_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                                                normals=canonical_normals,
                                                                                                                feature_vectors=feature_vector,
                                                                                                                pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                                cond=network_condition,
                                                                                                                shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가

            mask_hair = (torch.sigmoid(self._output['binary_segment']).squeeze() > 0.5).float().bool()

            specific_canonical_points = specific_canonical_points.reshape(batch_size, n_points, 3)
            specific_canonical_rgb_points = specific_canonical_rgb_points.reshape(batch_size, n_points, 3)

            specific_canonical_points = specific_canonical_points[:, mask_hair, :]
            specific_canonical_rgb_points = specific_canonical_rgb_points[:, mask_hair, :]

            features = torch.cat([specific_canonical_rgb_points, torch.ones_like(specific_canonical_rgb_points[..., [0]])], dim=-1)           # NOTE [batch_size, num of PCD, num of RGB features (4)]

            normal_begin_index = features.shape[-1]
            specific_canonical_normal = specific_canonical_normal.reshape(batch_size, n_points, 3)
            specific_canonical_normal = specific_canonical_normal[:, mask_hair, :]

            features = torch.cat([features, specific_canonical_normal * 0.5 + 0.5], dim=-1)

            # shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1]
            specific_canonical_albedo_points = torch.clamp(specific_canonical_albedo_points, 0., 1.)

            # shading_points = shading_points.reshape(batch_size, n_points, 3)
            specific_canonical_albedo_points = specific_canonical_albedo_points.reshape(batch_size, n_points, 3)
            specific_canonical_albedo_points = specific_canonical_albedo_points[:, mask_hair, :]

            features = torch.cat([features, specific_canonical_albedo_points], dim=-1)   # NOTE shading: [3], albedo: [3]

            specific_canonical_point_cloud = Pointclouds(points=specific_canonical_points, features=features)                                     # NOTE pytorch3d's pointcloud class.

            canonical_cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  
            focal_length = intrinsics[:, [0, 1], [0, 1]] # make sure the cameras focal length is logged too
            canonical_cameras.focal_length = focal_length
            canonical_cameras.principal_point = canonical_cameras.get_principal_point()

            images = self._render(specific_canonical_point_cloud, canonical_cameras)
            foreground_mask = images[..., 3].reshape(-1, 1)
            if not self.use_background:
                rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
            else:
                bkgd = torch.sigmoid(self.background * 100)
                rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
            rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)
            normal_image = (images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask)).reshape(batch_size, self.img_res[0], self.img_res[1], 3)
            albedo_image = (images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask)).reshape(batch_size, self.img_res[0], self.img_res[1], 3)

            save_image(rgb_image, 'specific_canonical_space_rgb')
            save_image(normal_image, 'specific_canonical_space_normal')
            save_image(albedo_image, 'specific_canonical_space_albedo')


        if flame_canonical_rendering:     # get_generic_canonical_rbg_value_functorch
            flame_canonical_points, flame_canonical_rgb_points, flame_canonical_albedo_points, _, flame_canonical_normal = self.get_flame_canonical_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                                                normals=canonical_normals,
                                                                                                                feature_vectors=feature_vector,
                                                                                                                pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                                cond=network_condition,
                                                                                                                shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가

            flame_canonical_points = flame_canonical_points.reshape(batch_size, n_points, 3)
            flame_canonical_rgb_points = flame_canonical_rgb_points.reshape(batch_size, n_points, 3)

            mask_hair = (torch.sigmoid(self._output['binary_segment']).squeeze() > 0.5).float().bool()
            flame_canonical_points = flame_canonical_points[:, mask_hair, :]
            flame_canonical_rgb_points = flame_canonical_rgb_points[:, mask_hair, :]

            features = torch.cat([flame_canonical_rgb_points, torch.ones_like(flame_canonical_rgb_points[..., [0]])], dim=-1)           # NOTE [batch_size, num of PCD, num of RGB features (4)]

            normal_begin_index = features.shape[-1]
            flame_canonical_normal = flame_canonical_normal.reshape(batch_size, n_points, 3)

            flame_canonical_normal = flame_canonical_normal[:, mask_hair, :]

            features = torch.cat([features, flame_canonical_normal * 0.5 + 0.5], dim=-1)

            # shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1]
            flame_canonical_albedo_points = torch.clamp(flame_canonical_albedo_points, 0., 1.)

            # shading_points = shading_points.reshape(batch_size, n_points, 3)
            flame_canonical_albedo_points = flame_canonical_albedo_points.reshape(batch_size, n_points, 3)

            flame_canonical_albedo_points = flame_canonical_albedo_points[:, mask_hair, :]

            features = torch.cat([features, flame_canonical_albedo_points], dim=-1)   # NOTE shading: [3], albedo: [3]


            flame_canonical_point_cloud = Pointclouds(points=flame_canonical_points, features=features)                                     # NOTE pytorch3d's pointcloud class.

            canonical_cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  
            focal_length = intrinsics[:, [0, 1], [0, 1]] # make sure the cameras focal length is logged too
            canonical_cameras.focal_length = focal_length
            canonical_cameras.principal_point = canonical_cameras.get_principal_point()

            images = self._render(flame_canonical_point_cloud, canonical_cameras)
            foreground_mask = images[..., 3].reshape(-1, 1)
            if not self.use_background:
                rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
            else:
                bkgd = torch.sigmoid(self.background * 100)
                rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
            rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)
            normal_image = (images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask)).reshape(batch_size, self.img_res[0], self.img_res[1], 3)
            albedo_image = (images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask)).reshape(batch_size, self.img_res[0], self.img_res[1], 3)

            save_image(rgb_image, 'flame_canonical_space_rgb')
            save_image(normal_image, 'flame_canonical_space_normal')
            save_image(albedo_image, 'flame_canonical_space_albedo')

        if deformed_rendering: # and 'target_human_values' not in input and not input['chamfer_loss']: # NOTE deformed space에서 머리카락만 따는 방법.
            transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                                                normals=canonical_normals,
                                                                                                                feature_vectors=feature_vector,
                                                                                                                pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                                cond=network_condition,
                                                                                                                shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가

            transformed_points = transformed_points.reshape(batch_size, n_points, 3)
            rgb_points = rgb_points.reshape(batch_size, n_points, 3)

            mask_hair = (torch.sigmoid(self._output['binary_segment']).squeeze() > 0.5).float().bool()
            transformed_points = transformed_points[:, mask_hair, :]
            rgb_points = rgb_points[:, mask_hair, :]

            features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)           # NOTE [batch_size, num of PCD, num of RGB features (4)]

            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)

            normals_points = normals_points[:, mask_hair, :]

            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

            # shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1]
            albedo_points = torch.clamp(albedo_points, 0., 1.)

            shading_points = shading_points.reshape(batch_size, n_points, 3)
            albedo_points = albedo_points.reshape(batch_size, n_points, 3)

            albedo_points = albedo_points[:, mask_hair, :]

            features = torch.cat([features, albedo_points], dim=-1)   # NOTE shading: [3], albedo: [3]


            flame_canonical_point_cloud = Pointclouds(points=transformed_points, features=features)                                     # NOTE pytorch3d's pointcloud class.

            canonical_cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  
            focal_length = intrinsics[:, [0, 1], [0, 1]] # make sure the cameras focal length is logged too
            canonical_cameras.focal_length = focal_length
            canonical_cameras.principal_point = canonical_cameras.get_principal_point()

            images = self._render(flame_canonical_point_cloud, canonical_cameras)
            foreground_mask = images[..., 3].reshape(-1, 1)
            if not self.use_background:
                rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
            else:
                bkgd = torch.sigmoid(self.background * 100)
                rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
            rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)
            normal_image = (images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask)).reshape(batch_size, self.img_res[0], self.img_res[1], 3)
            albedo_image = (images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask)).reshape(batch_size, self.img_res[0], self.img_res[1], 3)

            save_image(rgb_image, '{}_deformed_canonical_space_rgb'.format(input['img_name'].item()))
            save_image(normal_image, 'deformed_canonical_space_normal')
            save_image(albedo_image, 'deformed_canonical_space_albedo')

            return None


        transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                                                     normals=canonical_normals,
                                                                                                                     feature_vectors=feature_vector,
                                                                                                                     pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                                     cond=network_condition,
                                                                                                                     shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     gt_beta_shapedirs=gt_beta_shapedirs,
                                                                                                                     rotation=batch_rodrigues(input['rotation']) if 'rotation' in input and input['rotation'] is not None else None,
                                                                                                                     translation=input['translation'] if 'translation' in input and input['translation'] is not None else None)

        # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        # p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        # # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach(), cond=network_condition) # NOTE network_condition 추가
        # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가

        # NOTE deformed space에서 translation시켜주기.
        # NOTE transformed_points: x_d
        # if 'rotation' in input:
        #     # NOTE input['rotation] is the angle-axis 3 dim.
        #     # NOTE transformed_points: [174000, 3]
        #     rotation_matrix = batch_rodrigues(input['rotation'])    # NOTE [1, 3] -> [1, 3, 3]
        #     transformed_points = torch.matmul(transformed_points, rotation_matrix.squeeze(0).cuda()) # NOTE [174000, 3] * [3, 3] = [174000, 3]

        # if 'translation' in input:
        #     # NOTE input['translation]: [1, 3], transformed_points: [174000, 3]
        #     transformed_points = transformed_points + input['translation'].expand(total_points, -1)

        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        rgb_points = rgb_points.reshape(batch_size, n_points, 3)
        # point feature to rasterize and composite

        hair_mask_rendering = False
        if hair_mask_rendering:
            mask_hair = (torch.sigmoid(self._output['binary_segment']).squeeze() > 0.5).float().bool()
            transformed_points = transformed_points[:, mask_hair, :]
            rgb_points = rgb_points[:, mask_hair, :]

        # if not self.training and 'masked_point_cloud_indices' in input: 
        #     # NOTE target_human의 뒤통수가 잘린 머리부분(FLAME vertices인걸로 추정..)과 sh+db에 대해 knn을 구한다. 
        #     _, index_nn, _ = knn_points(target_human_flame_transformed_points.cuda(), transformed_points, K=1000, return_nn=True) # NOTE pnts_c_flame에 대해 target_human_flame_canonical_points에서 K개의 가장 가까운 이웃.
        #     masked_point_cloud_indices_by_target_human = torch.zeros(transformed_points.shape[1]).cuda()
        #     masked_point_cloud_indices_by_target_human[torch.unique(index_nn)] = 1.0                        # NOTE 해당하는 부분에 대해 mask 생성

        #     mask_sh_db_hair = (torch.sigmoid(self._output['binary_segment']).squeeze() > 0.5).float()   # NOTE sh+db에서 hair만 있는 부분.
        #     mask_additional = masked_point_cloud_indices_by_target_human.bool() & ~mask_sh_db_hair.bool()

        #     input['masked_point_cloud_indices'] = torch.logical_or(mask_sh_db_hair, masked_point_cloud_indices_by_target_human).float()     # NOTE sh+db의 hair부분과 sh+db에서 추가할 부분만 갖고온다.

        #     rgb_points_bak = rgb_points.clone()

        #     rgb_points = torch.cat([rgb_points[:, mask_sh_db_hair.bool(), :], rgb_points[:, mask_additional.bool(), :]], dim=1)             # NOTE 추가하는 부분을 뒤에 붙인다.
            
        #     # rgb_points = rgb_points[:, input['masked_point_cloud_indices'].bool(), :]
        #     if 'middle_inference' in input:
        #         rgb_points = torch.cat([rgb_points, target_human_rgb_points.to(rgb_points.device)], dim=1)

        if not self.training and 'masked_point_cloud_indices' in input: 
            # NOTE target_human의 뒤통수가 잘린 머리부분(FLAME vertices인걸로 추정..)과 sh+db에 대해 knn을 구한다. 
            _, index_nn, _ = knn_points(target_human_flame_transformed_points.cuda(), transformed_points, K=1500, return_nn=True) # NOTE pnts_c_flame에 대해 target_human_flame_canonical_points에서 K개의 가장 가까운 이웃.
            masked_point_cloud_indices_by_target_human = torch.zeros(transformed_points.shape[1]).cuda()
            masked_point_cloud_indices_by_target_human[torch.unique(index_nn)] = 1.0                        # NOTE 해당하는 부분에 대해 mask 생성
            # NOTE supplmentary의 (6)에 해당함.

            mask_sh_db_hair = (torch.sigmoid(self._output['binary_segment']).squeeze() > 0.5).float()   # NOTE sh+db에서 hair만 있는 부분.
            mask_additional = masked_point_cloud_indices_by_target_human.bool() & ~mask_sh_db_hair.bool()

            # transformed_points[:, mask_additional.bool(), :]    # NOTE sh+db의 챙 부분
            # target_human_transformed_points # NOTE th의 face 부분

            dist, index_nn, _ = knn_points(transformed_points[:, mask_additional.bool(), :], target_human_transformed_points.cuda(), K=1, return_nn=True)      # NOTE point끼리 비교.

            # target_human_rgb_points # target face의 rgb들
            # target_human_rgb_points[:, index_nn[:, :, 0], :] # target face의 챙 부분에 대한 RGB들.
            rgb_points[:, mask_additional.bool(), :] = target_human_rgb_points[:, index_nn[:, :, 0].squeeze(0), :].cuda()          # NOTE target face의 챙 부분의 RGB를 원래 챙에다가 집어넣어준다. 

            input['masked_point_cloud_indices'] = torch.logical_or(mask_sh_db_hair, masked_point_cloud_indices_by_target_human).float()     # NOTE sh+db의 hair부분과 sh+db에서 추가할 부분만 갖고온다.

            rgb_points_bak = rgb_points.clone()

            # rgb_points = torch.cat([rgb_points[:, mask_sh_db_hair.bool(), :], rgb_points[:, mask_additional.bool(), :]], dim=1)             # NOTE 추가하는 부분을 뒤에 붙인다.
            
            rgb_points = rgb_points[:, input['masked_point_cloud_indices'].bool(), :]
            if 'middle_inference' in input:
                rgb_points = torch.cat([rgb_points, target_human_rgb_points.to(rgb_points.device)], dim=1)

        # if not self.training and 'masked_point_cloud_indices' in input:
        #     # _, index_nn, _ = knn_points(target_human_flame_transformed_points.cuda(), transformed_points, K=1000, return_nn=True) # NOTE pnts_c_flame에 대해 target_human_flame_canonical_points에서 K개의 가장 가까운 이웃.
        #     # masked_point_cloud_indices_by_target_human = torch.zeros(transformed_points.shape[1]).cuda()
        #     # masked_point_cloud_indices_by_target_human[torch.unique(index_nn)] = 1.0

        #     mask_sh_db_hair = (torch.sigmoid(self._output['binary_segment']).squeeze() > 0.5).float()   # NOTE sh+db에서 hair만 있는 부분.
        #     # mask_additional = masked_point_cloud_indices_by_target_human.bool() & ~mask_sh_db_hair.bool()
        #     mask_sh_db_face = (torch.sigmoid(self._output['binary_segment']).squeeze() <= 0.5).float() # NOTE sh+db에서 face만 있는 부분.

        #     input['masked_point_cloud_indices'] = mask_sh_db_hair # torch.logical_or(, masked_point_cloud_indices_by_target_human).float()

        #     rgb_points_bak = rgb_points.clone()

        #     rgb_points = rgb_points[:, input['masked_point_cloud_indices'].bool(), :]

        #     if 'middle_inference' in input:
        #         rgb_points = torch.cat([rgb_points, target_human_rgb_points.to(rgb_points.device)], dim=1)

        features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)        # NOTE [batch_size, num of PCD, num of RGB features (4)]
        
        if self.normal and self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)
            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)                # NOTE [batch_size, num of PCD, num of normal features (3)]
        
        if self.training and self.binary_segmentation:
            # render segment image
            segment_begin_index = features.shape[-1]
            num_segment = self._output['binary_segment'].shape[-1]
            segments_points = self._output['binary_segment'].reshape(batch_size, n_points, num_segment)
            features = torch.cat([features, segments_points], dim=-1)                           # NOTE [batch_size, num of PCD, num of segment features (19)]

        if not self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)

            if 'masked_point_cloud_indices' in input:
                normals_points_bak = normals_points.clone()

                normals_points[:, mask_additional.bool(), :] = target_human_normals_points[:, index_nn[:, :, 0].squeeze(0), :].cuda()

                normals_points = normals_points[:, input['masked_point_cloud_indices'].bool(), :]
                if 'middle_inference' in input:
                    normals_points = torch.cat([normals_points, target_human_normals_points.to(normals_points.device)], dim=1)

            if hair_mask_rendering:
                normals_points = normals_points[:, mask_hair, :]

            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

            shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1] + 3
            albedo_points = torch.clamp(albedo_points, 0., 1.)

            shading_points = shading_points.reshape(batch_size, n_points, 3)
            albedo_points = albedo_points.reshape(batch_size, n_points, 3)

            if hair_mask_rendering:
                shading_points = shading_points[:, mask_hair, :]
                albedo_points = albedo_points[:, mask_hair, :]

            if 'masked_point_cloud_indices' in input:
                shading_points[:, mask_additional.bool(), :] = target_human_shading_points[:, index_nn[:, :, 0].squeeze(0), :].cuda()
                albedo_points[:, mask_additional.bool(), :] = target_human_albedo_points[:, index_nn[:, :, 0].squeeze(0), :].cuda()

                shading_points = shading_points[:, input['masked_point_cloud_indices'].bool(), :]
                albedo_points = albedo_points[:, input['masked_point_cloud_indices'].bool(), :]
                if 'middle_inference' in input:
                    shading_points = torch.cat([shading_points, target_human_shading_points.to(shading_points.device)], dim=1)
                    albedo_points = torch.cat([albedo_points, target_human_albedo_points.to(albedo_points.device)], dim=1)

            features = torch.cat([features, shading_points, albedo_points], dim=-1)   # NOTE shading: [3], albedo: [3]

        if not self.training and 'masked_point_cloud_indices' in input:
            transformed_points_bak = transformed_points.clone()

            ######################
            # NOTE sh+db에서 mask_additional에 해당하는 부분의 point를 움직일 것이다.
            # pcd_sh_db_hair = transformed_points[:, mask_sh_db_hair.bool(), :]       # NOTE sh+db에서 hair만 해당하는 부분. torch.Size([1, 105911, 3])
            # pcd_empty = transformed_points[:, mask_additional.bool(), :]            # NOTE 빈 공백에 해당하는 부분. torch.Size([1, 1943, 3])
            # pcd_th_face = target_human_transformed_points.cuda()                    # NOTE th의 face에 해당하는 부분.
            # # NOTE 빈 공백에 해당하는 부분을 th의 face에 해당하는 부분으로 거리에 반비례해서 움직인다.
            # # Step 1: Compute KNN
            # knn_result = knn_points(pcd_empty, pcd_th_face, K=1) # pcd_th_face -> pcd_sh_db_hair으로 바꾸었다.

            # # Step 2: Extract distances and indices of the nearest points
            # distances = knn_result.dists[..., 0]  # Shape [1, 1382]
            # nearest_indices = knn_result.idx[..., 0]  # Shape [1, 1382]

            # # Step 3: Normalize distances to use them as weights for interpolation
            # max_distance = distances.max()
            # weights = 1 - (distances / max_distance)  # Closer points get higher weights

            # # Step 4: Gather the closest points from pcd_th_face
            # closest_points = torch.gather(pcd_th_face, 1, nearest_indices.unsqueeze(-1).expand(-1, -1, 3))

            # # Step 5: Interpolate new positions for points in pcd_empty
            # pcd_empty_deformed = weights.unsqueeze(-1) * closest_points + (1 - weights.unsqueeze(-1)) * pcd_empty

            # transformed_points = torch.cat([pcd_sh_db_hair, pcd_empty_deformed], dim=1)
            ######################
            
            if 'chamfer_loss' in input and input['chamfer_loss']:
                # chamferDist = ChamferDistance()
                landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)                       # NOTE 이건 FLAME에 따른 거니까 일정할거고
                
                mask_face_part_sh_db = (torch.sigmoid(self._output['binary_segment']).squeeze() < 0.5).float()      # NOTE 얘도 일정할거같긴한데.. 어차피 face만 잡아주니까.

                sh_db_face_part = transformed_points[:, mask_face_part_sh_db.bool(), :]
                # NOTE 추측이긴 하지만 만약 얘가 keypoint를 제대로 잡아주지 못하면 말짱 도루묵이 된다. 그래서 처음 잡아준걸 기준으로 하는거다. 

                if 'index_nn' in input and input['index_nn'] is None:
                    _, index_nn, _ = knn_points(landmarks3d.cuda(), sh_db_face_part, K=1, return_nn=True) 
                    # sh_db_landmarks3d = nn_points[:, :, 0, :]       # NOTE sh_db_landmarks3d.shape: [1, 68, 3], sh_db_face_part[:, index_nn.squeeze(), :] == nn_points[:, :, 0, :]
                else:
                    index_nn = input['index_nn']
                
                # if input['epoch'] > 30:
                #     _, index_nn, _ = knn_points(landmarks3d.cuda(), sh_db_face_part, K=1, return_nn=True) 

                sh_db_landmarks3d = sh_db_face_part[:, index_nn.squeeze(), :]

                # cut landmarks, 1,2,3,4,5: 바라보는 방향에서 머리 아래쪽, 6,7,8,9,10: 턱, 11,12,13,14,15: 바라보는 방향에서 오른쪽 뺨, 16,17,18,19,20: 왼쪽 눈썹과 오른쪽 뺨, 21,22,23,24,25: 오른쪽 눈썹
                # 26,27,28,29,30: 코, 31,32,33,34,35: 코 아래 인중, 36,37,38,39,40: 왼쪽 눈, 41,42,43,44,45: 오른쪽 눈, 46,47,48,49,50: 오른쪽 눈과 윗입술, 51,52,53,54,55: 오른쪽 윗입술과 아랫입술, 56,57,58,59,60: 아랫입술, 
                # 61,62,63,64,65,66,67:안쪽 입술
                # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]: 윤곽선
                mask_landmarks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] # paper: [7,8,9,16,17,18,19,20,21,22,23,24,25,56,57,58,59,60] # 이거 이제 버림.
                sh_db_landmarks3d = sh_db_landmarks3d[:, mask_landmarks, :]
                target_human_landmarks3d = target_human_landmarks3d[:, mask_landmarks, :]

                if True:        # for debugging
                    transformed_points_bak = transformed_points     # backup to restore

                    transformed_target_verts = Pointclouds(points=target_human_landmarks3d.cuda(), features=torch.ones_like(target_human_landmarks3d.cuda()))
                    rendered_landmarks_target = self._render(transformed_target_verts, cameras, render_kp=True)

                    transformed_verts = Pointclouds(points=sh_db_landmarks3d, features=torch.ones_like(sh_db_landmarks3d))
                    rendered_landmarks_sh_db = self._render(transformed_verts, cameras, render_kp=True)         # NOTE landmark rendering. result: 1, 512, 512, 3

                    transformed_points = transformed_points[:, input['masked_point_cloud_indices'].bool(), :] # NOTE torch.Size([1, 107854, 3])
                    transformed_points = torch.cat([transformed_points, target_human_transformed_points.cuda()], dim=1)  

                    flame_canonical_point_cloud = Pointclouds(points=transformed_points, features=features)                                     # NOTE pytorch3d's pointcloud class.

                    canonical_cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  
                    focal_length = intrinsics[:, [0, 1], [0, 1]] # make sure the cameras focal length is logged too
                    canonical_cameras.focal_length = focal_length
                    canonical_cameras.principal_point = canonical_cameras.get_principal_point()

                    images = self._render(flame_canonical_point_cloud, canonical_cameras)
                    foreground_mask = images[..., 3].reshape(-1, 1)
                    if not self.use_background:
                        rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
                    else:
                        bkgd = torch.sigmoid(self.background * 100)
                        rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
                    rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

                    rgb_image = rgb_image * (1 - rendered_landmarks_target) + rendered_landmarks_target * torch.tensor([0, 0, 1]).cuda()
                    rgb_image = rgb_image * (1 - rendered_landmarks_sh_db) + rendered_landmarks_sh_db * torch.tensor([1, 0, 0]).cuda()
                    
                    save_image(rgb_image, 'deformed_canonical_space_rgb')
                    transformed_points = transformed_points_bak


                # NOTE index_nn.shape: [1, 68, 1], nn_points.shape: [1, 68, 1, 3]
                # dist_forward = chamferDist(target_human_landmarks3d.cuda(), nn_points[:, :, 0, :])

                # NOTE canonical flame에서 비교해보려고 했다.
                # _, landmarks3d = self.FLAMEServer.find_landmarks(knn_v, full_pose=self.FLAMEServer.canonical_pose)                       # NOTE 이건 FLAME에 따른 거니까 일정할거고
                # # mask_face_part_sh_db = (torch.sigmoid(self._output['binary_segment']).squeeze() < 0.5).float()      # NOTE 얘도 일정할거같긴한데.. 어차피 face만 잡아주니까.
                # _, index_nn, nn_points = knn_points(landmarks3d.cuda(), pnts_c_flame.unsqueeze(0), K=1, return_nn=True) 

                

                dist_forward = torch.nn.MSELoss()(target_human_landmarks3d.squeeze(0).cuda(), sh_db_landmarks3d.squeeze(0)) + dist.squeeze().mean()
                return dist_forward, index_nn


            # # NOTE deformed말고 canonical space에서 따보자.
            # if 'chamfer_loss' in input and input['chamfer_loss']:
            #     # # chamferDist = ChamferDistance()
            #     # landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)                       # NOTE 이건 FLAME에 따른 거니까 일정할거고
            #     # mask_face_part_sh_db = (torch.sigmoid(self._output['binary_segment']).squeeze() < 0.5).float()      # NOTE 얘도 일정할거같긴한데.. 어차피 face만 잡아주니까.
            #     # sh_db_face_part = transformed_points[:, mask_face_part_sh_db.bool(), :]
            #     # # NOTE 추측이긴 하지만 만약 얘가 keypoint를 제대로 잡아주지 못하면 말짱 도루묵이 된다. 그래서 처음 잡아준걸 기준으로 하는거다. 
                

            #     _, landmarks3d = self.FLAMEServer.find_landmarks(knn_v, full_pose=self.FLAMEServer.canonical_pose)                       # NOTE 이건 FLAME에 따른 거니까 일정할거고
            #     mask_face_part_sh_db = (torch.sigmoid(self._output['binary_segment']).squeeze() < 0.5).float()      # NOTE 얘도 일정할거같긴한데.. 어차피 face만 잡아주니까.
            #     sh_db_face_part = pnts_c_flame.unsqueeze(0)[:, mask_face_part_sh_db.bool(), :]
                

            #     if 'index_nn' in input and input['index_nn'] is None:
            #         _, index_nn, _ = knn_points(landmarks3d.cuda(), sh_db_face_part, K=1, return_nn=True) 
            #         # sh_db_landmarks3d = nn_points[:, :, 0, :]       # NOTE sh_db_landmarks3d.shape: [1, 68, 3], sh_db_face_part[:, index_nn.squeeze(), :] == nn_points[:, :, 0, :]
            #     else:
            #         index_nn = input['index_nn']
                
            #     # if input['epoch'] > 30:
            #     #     _, index_nn, _ = knn_points(landmarks3d.cuda(), sh_db_face_part, K=1, return_nn=True) 

            #     sh_db_landmarks3d = sh_db_face_part[:, index_nn.squeeze(), :]

            #     if True:        # for debugging
            #         transformed_points_bak = transformed_points     # backup to restore

            #         transformed_target_verts = Pointclouds(points=target_human_landmarks3d.cuda(), features=torch.ones_like(target_human_landmarks3d.cuda()))
            #         rendered_landmarks_target = self._render(transformed_target_verts, cameras, render_kp=True)

            #         transformed_verts = Pointclouds(points=sh_db_landmarks3d, features=torch.ones_like(sh_db_landmarks3d))
            #         rendered_landmarks_sh_db = self._render(transformed_verts, cameras, render_kp=True)         # NOTE landmark rendering. result: 1, 512, 512, 3

            #         transformed_points = transformed_points[:, input['masked_point_cloud_indices'].bool(), :] # NOTE torch.Size([1, 107854, 3])
            #         transformed_points = torch.cat([transformed_points, target_human_transformed_points.cuda()], dim=1)  

            #         flame_canonical_point_cloud = Pointclouds(points=transformed_points, features=features)                                     # NOTE pytorch3d's pointcloud class.

            #         canonical_cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  
            #         focal_length = intrinsics[:, [0, 1], [0, 1]] # make sure the cameras focal length is logged too
            #         canonical_cameras.focal_length = focal_length
            #         canonical_cameras.principal_point = canonical_cameras.get_principal_point()

            #         images = self._render(flame_canonical_point_cloud, canonical_cameras)
            #         foreground_mask = images[..., 3].reshape(-1, 1)
            #         if not self.use_background:
            #             rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
            #         else:
            #             bkgd = torch.sigmoid(self.background * 100)
            #             rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
            #         rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

            #         rgb_image = rgb_image * (1 - rendered_landmarks_target) + rendered_landmarks_target * torch.tensor([0, 0, 1]).cuda()
            #         rgb_image = rgb_image * (1 - rendered_landmarks_sh_db) + rendered_landmarks_sh_db * torch.tensor([1, 0, 0]).cuda()
                    
            #         save_image(rgb_image, 'deformed_canonical_space_rgb')
            #         transformed_points = transformed_points_bak


            #     # NOTE index_nn.shape: [1, 68, 1], nn_points.shape: [1, 68, 1, 3]
            #     # dist_forward = chamferDist(target_human_landmarks3d.cuda(), nn_points[:, :, 0, :])

            #     # NOTE canonical flame에서 비교해보려고 했다.
            #     # _, landmarks3d = self.FLAMEServer.find_landmarks(knn_v, full_pose=self.FLAMEServer.canonical_pose)                       # NOTE 이건 FLAME에 따른 거니까 일정할거고
            #     # # mask_face_part_sh_db = (torch.sigmoid(self._output['binary_segment']).squeeze() < 0.5).float()      # NOTE 얘도 일정할거같긴한데.. 어차피 face만 잡아주니까.
            #     # _, index_nn, nn_points = knn_points(landmarks3d.cuda(), pnts_c_flame.unsqueeze(0), K=1, return_nn=True) 

            #     dist_forward = torch.nn.MSELoss()(target_human_landmarks3d.squeeze(0).cuda(), sh_db_landmarks3d.squeeze(0))
            #     return dist_forward, index_nn
            
            transformed_points = transformed_points[:, input['masked_point_cloud_indices'].bool(), :] # NOTE torch.Size([1, 107854, 3])

            if 'middle_inference' in input:
                # delta_transformed_points = middle_inference['transformed_points'].mean(dim=1).cuda() - transformed_points.mean(dim=1)
                # target_human_transformed_points = target_human_transformed_points.cuda() - delta_transformed_points
                transformed_points = torch.cat([transformed_points, target_human_transformed_points.cuda()], dim=1)                 # NOTE sh+db의 hair와 공백과 th의 얼굴을 합쳐준다. transformed_points_additional

        if 'target_human_values' in input:
            masked_point_cloud_indices = (torch.sigmoid(self._output['binary_segment']).squeeze() < 0.5).float()                # NOTE th의 face만 들고오기.
            middle_inference = {
                'rgb_points': rgb_points[:, masked_point_cloud_indices.bool(), :],
                'normals_points': normals_points[:, masked_point_cloud_indices.bool(), :],              # .unsqueeze(0)
                'shading_points': shading_points[:, masked_point_cloud_indices.bool(), :],              # .unsqueeze(0)
                'albedo_points': albedo_points[:, masked_point_cloud_indices.bool(), :],                # .unsqueeze(0)
                'transformed_points': transformed_points[:, masked_point_cloud_indices.bool(), :],
                'pcd_center': self.pc.points.detach().mean(dim=0)
            }
            # NOTE target human의 hair도 들고와서 그 point를 들고온다.
            if 'category' in input and input['category'] == 'hair':
                # NOTE chamfer distance로 해결을 해보기 위해 추가한다.
                
                # NOTE deformed space에서의 points
                if True:
                    landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
                    # Find the nearest vertices between the landmarks3d and transformed_points[:, masked_point_cloud_indices.bool(), :]
                    _, index_nn, nn_points = knn_points(landmarks3d.cuda(), transformed_points[:, masked_point_cloud_indices.bool(), :], K=1, return_nn=True) # NOTE pnts_c_flame에 대해 target_human_flame_canonical_points에서 K개의 가장 가까운 이웃.

                    middle_inference['landmarks3d'] = nn_points[:, :, 0, :] # target human의 얼굴 상에서 landmark를 찍은 거라고 보면 된다.
                    middle_inference['landmarks2d'] = landmarks2d

                # NOTE canonical space에서의 inputs
                # _, landmarks3d = self.FLAMEServer.find_landmarks(knn_v, full_pose=self.FLAMEServer.canonical_pose)      # FLAME의 canonical verts에서 나오느 landmarks3d와 pnts_c_flame과의 knn을 구해준다.
                # # Find the nearest vertices between the landmarks3d and transformed_points[:, masked_point_cloud_indices.bool(), :]
                # _, index_nn, nn_points = knn_points(landmarks3d.cuda(), pnts_c_flame.unsqueeze(0), K=1, return_nn=True) # NOTE pnts_c_flame에 대해 target_human_flame_canonical_points에서 K개의 가장 가까운 이웃.

                # middle_inference['landmarks3d'] = nn_points[:, :, 0, :] # target human의 얼굴 상에서 landmark를 찍은 거라고 보면 된다.


                masked_point_cloud_indices = (torch.sigmoid(self._output['binary_segment']).squeeze() > 0.4).float()                    # NOTE target human의 hair에 해당한다. 
                selected_indices = torch.from_numpy(np.load('../utility/selected_indices.npy'))                                         
                selected_mask = ~torch.isin(index_batch, selected_indices.cuda())                                                       # NOTE selected_indices를 제외 (즉, 윗통수만 제외한다.)
                final_vertices = torch.unique(verts[:, index_batch[selected_mask], :][:, masked_point_cloud_indices[selected_mask].bool(), :], dim=1)
                # flame_transformed_points = self.densify_point_cloud(final_vertices, final_vertices.shape[1]-5)    
                # NOTE 수식5에 해당함.
                middle_inference['flame_transformed_points'] = final_vertices # flame_transformed_points

                # canonical_verts = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
                # final_canonical_verts = torch.unique(canonical_verts[:, index_batch[selected_mask], :][:, masked_point_cloud_indices[selected_mask].bool(), :], dim=1)
                # flame_canonical_points = self.densify_point_cloud(final_canonical_verts, final_canonical_verts.shape[1]-5)  # 255-5
                # middle_inference['flame_canonical_points'] = flame_canonical_points

            if False:        # NOTE for debugging rendering. TH가 제대로 불러온게 맞는지 확인할 수 있는 방법.
                features = torch.cat([middle_inference['rgb_points'], torch.ones_like(middle_inference['rgb_points'][..., [0]])], dim=-1)
                transformed_point_cloud = Pointclouds(points=middle_inference['transformed_points'], features=features)                                     # NOTE pytorch3d's pointcloud class.

                images = self._render(transformed_point_cloud, cameras)
                foreground_mask = images[..., 3].reshape(-1, 1)
                if not self.use_background:
                    rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
                else:
                    bkgd = torch.sigmoid(self.background * 100)
                    rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
                rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

                save_image(rgb_image, 'debug_th_rgb')

            # NOTE FLAME을 들고오는 방법.
            # if 'category' in input and input['category'] == 'hair':
            #     masked_point_cloud_indices = (torch.sigmoid(self._output['binary_segment']).squeeze() > 0.5).float()

            #     flame_transformed_points = self.densify_point_cloud(torch.unique(verts[:, index_batch, :][:, masked_point_cloud_indices.bool(), :], dim=1), 20)       
            #     flame_rgb_points = torch.zeros_like(flame_transformed_points)
            #     flame_normals_points = torch.zeros_like(flame_transformed_points)
            #     flame_shading_points = torch.zeros_like(flame_transformed_points)
            #     flame_albedo_points = torch.zeros_like(flame_transformed_points)

            #     middle_inference['transformed_points'] = torch.cat([middle_inference['transformed_points'], flame_transformed_points], dim=1)
            #     middle_inference['rgb_points'] = torch.cat([middle_inference['rgb_points'], flame_rgb_points], dim=1)
            #     middle_inference['normals_points'] = torch.cat([middle_inference['normals_points'], flame_normals_points], dim=1)
            #     middle_inference['shading_points'] = torch.cat([middle_inference['shading_points'], flame_shading_points], dim=1)
            #     middle_inference['albedo_points'] = torch.cat([middle_inference['albedo_points'], flame_albedo_points], dim=1)

            return middle_inference
        
        transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

        if False and 'masked_point_cloud_indices' in input:     # NOTE FLAME points를 그리기위해 쓰이는 코드이다.
            mask_hole_empty_pixels = self._find_empty_pixels(transformed_point_cloud, cameras)

            target_flame_rgb_points = torch.zeros_like(target_human_flame_transformed_points).cuda()
            target_flame_features = torch.cat([target_flame_rgb_points, torch.ones_like(target_flame_rgb_points[..., [0]])], dim=-1)
            target_flame_transformed_point_cloud = Pointclouds(points=target_human_flame_transformed_points.cuda(), features=target_flame_features)
            segment_mask = self._segment(target_flame_transformed_point_cloud, cameras, mask=mask_hole_empty_pixels, render_debug=False)
            segment_mask = segment_mask.detach().cpu()
            target_human_flame_transformed_points = target_human_flame_transformed_points[:, segment_mask, :]
            # foreground_mask = segmented_images[..., 3].reshape(-1, 1)
            # if not self.use_background:
            #     rgb_values = segmented_images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
            # else:
            #     bkgd = torch.sigmoid(self.background * 100)
            #     rgb_values = segmented_images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
            # rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

            # # Convert tensor to numpy and squeeze the batch dimension
            # img_np = rgb_image.squeeze(0).detach().cpu().numpy()

            # # Convert range from [0, 1] to [0, 255] if needed
            # if img_np.max() <= 1:
            #     img_np = (img_np * 255).astype('uint8')

            # # Swap RGB to BGR
            # img_bgr = img_np[:, :, [2, 1, 0]]

            # # Save the image
            # cv2.imwrite('segmented_images.png', img_bgr)

            transformed_points = torch.cat([transformed_points, target_human_flame_transformed_points.cuda()], dim=1)

            target_human_flame_rgb_points = torch.zeros_like(target_human_flame_transformed_points)
            rgb_points = torch.cat([rgb_points, target_human_flame_rgb_points.to(rgb_points.device)], dim=1)

            features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1) 

            normal_begin_index = features.shape[-1]
            target_human_flame_normals_points = torch.zeros_like(target_human_flame_transformed_points)
            normals_points = torch.cat([normals_points, target_human_flame_normals_points.to(normals_points.device)], dim=1)
            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

            shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1] + 3

            target_human_flame_shading_points = torch.zeros_like(target_human_flame_transformed_points).cuda()
            target_human_flame_albedo_points = torch.zeros_like(target_human_flame_transformed_points).cuda()

            shading_points = torch.cat([shading_points, target_human_flame_shading_points], dim=1)
            albedo_points = torch.cat([albedo_points, target_human_flame_albedo_points], dim=1)

            features = torch.cat([features, shading_points, albedo_points], dim=-1)   # NOTE shading: [3], albedo: [3]

            transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.
            # if input['sub_dir'][0] in self.count_sub_dirs.keys():
            #     if self.count_sub_dirs[input['sub_dir'][0]] < self.num_views-1:
            #         self.count_sub_dirs[input['sub_dir'][0]] += 1
            #         self.voting_table[input['indices_tensor'], segment_mask, self.count_sub_dirs[input['sub_dir'][0]]] = 1
            # else:
            #     self.count_sub_dirs[input['sub_dir'][0]] = 0
            #     self.voting_table[input['indices_tensor'], segment_mask, 0] = 1

        # if self.training and ('rank' in input) and (input['rank'] == 0) and (not self.target_training):
        #     segment_mask = self._segment(transformed_point_cloud, cameras, mask=input['mask_object'], render_debug=False)
        #     segment_mask = segment_mask.detach().cpu()
        #     if input['sub_dir'][0] in self.count_sub_dirs.keys():
        #         if self.count_sub_dirs[input['sub_dir'][0]] < self.num_views-1:
        #             self.count_sub_dirs[input['sub_dir'][0]] += 1
        #             self.voting_table[input['indices_tensor'], segment_mask, self.count_sub_dirs[input['sub_dir'][0]]] = 1
        #     else:
        #         self.count_sub_dirs[input['sub_dir'][0]] = 0
        #         self.voting_table[input['indices_tensor'], segment_mask, 0] = 1

        images = self._render(transformed_point_cloud, cameras)

        if not self.training:
            # render landmarks for easier camera format debugging
            landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
            transformed_verts = Pointclouds(points=landmarks2d, features=torch.ones_like(landmarks2d))
            rendered_landmarks = self._render(transformed_verts, cameras, render_kp=True)

        foreground_mask = images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
        
        
        if not self.training and 'masked_point_cloud_indices' in input:
            rgb_points = rgb_points_bak
            transformed_points = transformed_points_bak
            normals_points = normals_points_bak
        # knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        # flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        # index_batch = index_batch.reshape(-1)

        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # training outputs
        output = {
            'img_res': self.img_res,
            'batch_size': batch_size,
            'predicted_mask': foreground_mask,  # mask loss
            'rgb_image': rgb_image,
            'canonical_points': pnts_c_flame,
            # for flame loss
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights,
            'pcd_center': self.pc.points.mean(dim=0),
        }

        if self.normal and self.training:
            output['normal_image'] = (images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask)).reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        if self.training and self.binary_segmentation:
            output['segment_image'] = images[..., segment_begin_index:segment_begin_index+num_segment]

        if not self.training:
            output_testing = {
                'normal_image': images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'shading_image': images[..., shading_begin_index:shading_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'albedo_image': images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'rendered_landmarks': rendered_landmarks.reshape(-1, 3),
                'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
                'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
                'deformed_verts': verts.reshape(-1, 3),
                'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
                'pnts_normal_deformed': normals_points.reshape(batch_size, n_points, 3),
                #'pnts_normal_canonical': canonical_normals,
            }
            # if self.segment:
            #     output_testing['segment_image'] = images[..., segment_begin_index:segment_begin_index+num_segment].reshape(-1, 10)

            if self.deformer_network.deform_c:
                output_testing['unconstrained_canonical_points'] = self.pc.points
            output.update(output_testing)
        output.update(self._output)

        if not self.training and 'masked_point_cloud_indices' in input:
            self._output['mask_hole'] = self._output['mask_hole'].reshape(-1).unsqueeze(0)

        if self.optimize_scene_latent_code and self.training:
            output['scene_latent_code'] = input["scene_latent_code"]

        return output


    def get_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None, shapes=None, gt_beta_shapedirs=None, rotation=None, translation=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

        pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
        n_points = pnts_c.shape[0]                                                  # NOTE 400
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None, shapes=None, gt_beta_shapedirs=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
            pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
            # NOTE custom #########################
            condition = {}
            condition['scene_latent'] = scene_latent
            #######################################
            # shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)           
            shapedirs = shapedirs.expand(batch_size, -1, -1)
            posedirs = posedirs.expand(batch_size, -1, -1)
            lbs_weights = lbs_weights.expand(batch_size, -1)
            pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
            if rotation is not None:
                pnts_c_flame = torch.matmul(pnts_c_flame, rotation.squeeze(0).cuda())
            if translation is not None:
                pnts_c_flame  = pnts_c_flame + translation
            # beta_shapedirs = beta_shapedirs.expand(batch_size, -1, -1)                                                          # NOTE beta cancel
            beta_shapedirs = gt_beta_shapedirs.unsqueeze(0).expand(batch_size, -1, -1)
            pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, beta_shapedirs=beta_shapedirs, shapes=shapes) # FLAME-based deformed
            pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
            return pnts_d, pnts_d

        normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        shapes = shapes.reshape(batch_size, n_points, *shapes.shape[1:]).transpose(0, 1)                                             # NOTE beta cancel
        if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
            scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)      # NOTE pnts_c로 미분.
            # pnts_c: [400, 3], betas: [400, 1, 50], transformations: [400, 1, 6, 4, 4], pose_feature: [400, 1, 36], scene_latent: [400, 1, 32]
        else:
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        grads_inv = grads_batch.inverse()
        normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)
        shading = self.rendering_network(normals_d, cond) # TODO 여기다가 condition을 추가하면 어떻게 될까?????
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
        return pnts_d, rgb_vals, albedo, shading, normals_d


    def get_flame_canonical_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None, shapes=None, gt_beta_shapedirs=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

        pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
        n_points = pnts_c.shape[0]                                                  # NOTE 400
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None, shapes=None, gt_beta_shapedirs=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
            pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
            # NOTE custom #########################
            condition = {}
            condition['scene_latent'] = scene_latent
            #######################################
            # shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)           # NOTE pnts_c_flame: [1, 3]
            # shapedirs = shapedirs.expand(batch_size, -1, -1)
            # posedirs = posedirs.expand(batch_size, -1, -1)
            # lbs_weights = lbs_weights.expand(batch_size, -1)
            # pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
            # beta_shapedirs = gt_beta_shapedirs.unsqueeze(0).expand(batch_size, -1, -1)
            # pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, beta_shapedirs=beta_shapedirs, shapes=shapes) # FLAME-based deformed
            # pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
            # return pnts_d, pnts_d
            pnts_c_flame = pnts_c_flame.reshape(-1)
            return pnts_c_flame, pnts_c_flame

        normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        shapes = shapes.reshape(batch_size, n_points, *shapes.shape[1:]).transpose(0, 1)                                             # NOTE beta cancel
        if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
            scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
            # grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
            grads_batch, pnts_c = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
        else:
            # grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)
            grads_batch, pnts_c = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        # pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        pnts_c = pnts_c.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        grads_inv = grads_batch.inverse()
        # normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        normals = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)

        # shading = self.rendering_network(normals_d, cond) 
        shading = self.rendering_network(normals, cond) 
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
        # return pnts_d, rgb_vals, albedo, shading, normals_d
        return pnts_c, rgb_vals, albedo, shading, normals
    
    def get_specific_canonical_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None, shapes=None, gt_beta_shapedirs=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

        pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
        n_points = pnts_c.shape[0]                                                  # NOTE 400
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None, shapes=None, gt_beta_shapedirs=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
            pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
            # NOTE custom #########################
            condition = {}
            condition['scene_latent'] = scene_latent
            #######################################
            # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)           # NOTE pnts_c_flame: [1, 3]
            pnts_c = pnts_c.reshape(-1)
            return pnts_c, pnts_c

        normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        shapes = shapes.reshape(batch_size, n_points, *shapes.shape[1:]).transpose(0, 1)                                             # NOTE beta cancel
        if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
            scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
            # grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
            grads_batch, pnts_c = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
        else:
            # grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)
            grads_batch, pnts_c = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        # pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        pnts_c = pnts_c.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        grads_inv = grads_batch.inverse()
        # normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        normals = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)

        # shading = self.rendering_network(normals_d, cond) 
        shading = self.rendering_network(normals, cond) 
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
        # return pnts_d, rgb_vals, albedo, shading, normals_d
        return pnts_c, rgb_vals, albedo, shading, normals





class Ablation1Model(nn.Module):
    # NOTE cano canonical offset, shape deformation이 삭제되어 있음.
    def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background, checkpoint_path, latent_code_dim, pcd_init=None):
        super().__init__()
        # shape_params = None
        self.optimize_latent_code = conf.get_bool('train.optimize_latent_code')
        self.optimize_scene_latent_code = conf.get_bool('train.optimize_scene_latent_code')

        # FLAME_lightning
        self.FLAMEServer = utils.get_class(conf.get_string('model.FLAME_class'))(flame_model_path='./flame/FLAME2020/generic_model.pkl', 
                                                                                lmk_embedding_path='./flame/FLAME2020/landmark_embedding.npy',
                                                                                n_shape=100,
                                                                                n_exp=50,
                                                                                shape_params=shape_params,                                      # NOTE BetaCancel
                                                                                canonical_expression=canonical_expression,
                                                                                canonical_pose=canonical_pose).cuda()                           # NOTE cuda 없앴음

        self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
            self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)
        
        self.prune_thresh = conf.get_float('model.prune_thresh', default=0.5)

        # NOTE custom #########################
        # scene latent를 위해 변형한 모델들이 들어감.
        # self.latent_code_dim = conf.get_int('model.latent_code_dim')
        # print('[DEBUG] latent_code_dim:', self.latent_code_dim)
        self.latent_code_dim = latent_code_dim
        # GeometryNetworkSceneLatent
        self.geometry_network = utils.get_class(conf.get_string('model.geometry_class'))(optimize_latent_code=self.optimize_latent_code,
                                                                                         optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                         latent_code_dim=self.latent_code_dim,
                                                                                         **conf.get_config('model.geometry_network'))
        # ForwardDeformerSceneLatentThreeStages
        self.deformer_network = utils.get_class(conf.get_string('model.deformer_class'))(FLAMEServer=self.FLAMEServer,
                                                                                        optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                        latent_code_dim=self.latent_code_dim,
                                                                                        **conf.get_config('model.deformer_network'))
        # RenderingNetworkSceneLatentThreeStages
        self.rendering_network = utils.get_class(conf.get_string('model.rendering_class'))(optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                            latent_code_dim=self.latent_code_dim,
                                                                                            **conf.get_config('model.rendering_network'))
        #######################################

        self.ghostbone = self.deformer_network.ghostbone
        if self.ghostbone:
            self.FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().cuda(), self.FLAMEServer.canonical_transformations], 1)

        # NOTE custom #########################
        # self.test_target_finetuning = conf.get_bool('test.target_finetuning')
        self.normal = True if conf.get_float('loss.normal_weight') > 0 and self.training else False
        self.target_training = conf.get_bool('train.target_training', default=False)
        # self.segment = True if conf.get_float('loss.segment_weight') > 0 else False
        if checkpoint_path is not None:
            print('[DEBUG] init point cloud from previous checkpoint')
            # n_init_point를 checkpoint으로부터 불러오기 위해..
            data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            n_init_points = data['state_dict']['model.pc.points'].shape[0]
            try:
                init_radius = data['state_dict']['model.pc.radius'].item()
            except:
                init_radius = data['state_dict']['model.radius'].item()
        elif pcd_init is not None:
            print('[DEBUG] init point cloud from meta learning')
            n_init_points = pcd_init['n_init_points']
            init_radius = pcd_init['init_radius']
        else:
            print('[DEBUG] init point cloud from scratch')
            n_init_points = 400
            init_radius = 0.5
            # init_radius = self.pc.radius_factor * (0.75 ** math.log2(n_points / 100))

        self.pc = utils.get_class(conf.get_string('model.pointcloud_class'))(n_init_points=400,
                                                                            init_radius=0.5,
                                                                            **conf.get_config('model.point_cloud')).cuda()    # NOTE .cuda() 없앴음
        #######################################

        n_points = self.pc.points.shape[0]
        self.img_res = img_res
        self.use_background = use_background
        # if self.use_background:
        #     init_background = torch.zeros(img_res[0] * img_res[1], 3).float().cuda()                   # NOTE .cuda() 없앰
        #     # self.background = nn.Parameter(init_background)                                   # NOTE singleGPU코드에서는 이렇게 작성했지만,
        #     self.register_parameter('background', nn.Parameter(init_background))                # NOTE 이렇게 수정해서 혹시나하는 버그를 방지해보고자 한다.
        # else:
        #     # self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()           # NOTE singleGPU코드에서는 이렇게 작성했지만,
        #     self.register_buffer('background', torch.ones(img_res[0] * img_res[1], 3).float())  # NOTE cuda 할당이 자동으로 되도록 수정해본다.
        if self.use_background:
            init_background = torch.zeros(img_res[0] * img_res[1], 3).float().cuda()
            self.background = nn.Parameter(init_background)
        else:
            self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()

        # # NOTE original code
        # self.raster_settings = PointsRasterizationSettings(image_size=img_res[0],
        #                                                    radius=init_radius,
        #                                                    points_per_pixel=10,
        #                                                    bin_size=0)              # NOTE warning이 뜨길래 bin_size=0 추가함.
        # # self.register_buffer('radius', torch.tensor(self.raster_settings.radius))
        
        # # keypoint rasterizer is only for debugging camera intrinsics
        # self.raster_settings_kp = PointsRasterizationSettings(image_size=self.img_res[0],
        #                                                       radius=0.007,
        #                                                       points_per_pixel=1)  
        self.raster_settings = PointsRasterizationSettings(
            image_size=img_res[0],
            radius=self.pc.radius_factor * (0.75 ** math.log2(n_points / 100)),
            points_per_pixel=10
        )
        # keypoint rasterizer is only for debugging camera intrinsics
        self.raster_settings_kp = PointsRasterizationSettings(
            image_size=self.img_res[0],
            radius=0.007,
            points_per_pixel=1
        )
        self.visible_points = torch.zeros(n_points).bool().cuda()
        self.compositor = AlphaCompositor().cuda()

        # # NOTE ablation #########################################
        self.enable_prune = conf.get_bool('train.enable_prune')

        # # self.visible_points = torch.zeros(n_points).bool().cuda()                             # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        # if self.enable_prune:
        #     if checkpoint_path is not None:
        #         # n_init_point를 checkpoint으로부터 불러오기 위해..
        #         data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        #         visible_points = data['state_dict']['model.visible_points']                         # NOTE 이거 안해주면 visible이 0이 되어서 훈련이 안됨.
        #     else:
        #         visible_points = torch.zeros(n_points).bool()
        #     self.register_buffer('visible_points', visible_points)                                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.
            
        # self.compositor = AlphaCompositor().cuda()                                                     # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        self.num_views = 100
        # NOTE 모든 view에 대해서 segment된 point indices를 다 저장한다.
        # self.voting_table = torch.zeros((len(conf.get_list('dataset.train.sub_dir')), n_points, self.num_views))
        # NOTE voting_table의 마지막 view의 index를 지정하기 위해, 각 sub_dir이 몇번 나왔는지 세기 위해서 만들었다.
        # self.count_sub_dirs = {}
        self.binary_segmentation = conf.get_bool('model.binary_segmentation', default=False)


    def _compute_canonical_normals_and_feature_vectors(self, p, condition):
        # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        # p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
        # randomly sample some points in the neighborhood within 0.25 distance

        # eikonal_points = torch.cat([p, p + (torch.rand(p.shape).cuda() - 0.5) * 0.5], dim=0)                          # NOTE original code, eikonal_points.shape: [6400, 3]
        eikonal_points = torch.cat([p, p + (torch.rand(p.shape, device=p.device) - 0.5) * 0.5], dim=0)                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        if self.optimize_scene_latent_code:
            condition['scene_latent_gradient'] = torch.cat([condition['scene_latent'], condition['scene_latent']], dim=0).detach()

        eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points.detach(), condition)
        n_points = self.pc.points.shape[0] # 400
        canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1) # 400, 3

        # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        # p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
        geometry_output = self.geometry_network(p, condition)  # not using SDF to regularize point location, 3200, 4
        sdf_values = geometry_output[:, 0]

        if not self.binary_segmentation:
            feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector
        else:
            feature_vector = torch.sigmoid(geometry_output[:, 1:-1] * 10)  # albedo vector
            binary_segment = geometry_output[:, -1:]         # dim = 1
        # feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector
        # segment_probability = geometry_output[:, -10:]  # segment probability                       # NOTE segment를 위해 추가했음.

        if self.training and hasattr(self, "_output"):
            self._output['sdf_values'] = sdf_values
            self._output['grad_thetas'] = grad_thetas
        if self.binary_segmentation:
            self._output['binary_segment'] = binary_segment
        if not self.training:
            self._output['pnts_albedo'] = feature_vector

        return canonical_normals, feature_vector # (400, 3), (400, 3) -> (400, 3) (3200, 3)

    def _segment(self, point_cloud, cameras, mask, kernel_size=0, render_debug=True, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)

        # mask = mask > 127.5
        mask = mask > 0.5
        if kernel_size > 0:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.erode(mask.cpu().numpy().astype(np.uint8), kernel, iterations=1)
        mask = torch.tensor(mask, device=point_cloud.device).bool()
        on_pixels = fragments.idx[0, mask.reshape(1, 512, 512)[0]].long()

        unique_on_pixels = torch.unique(on_pixels[on_pixels >= 0])      
        # segmented_point = point[0, unique_on_pixels].unsqueeze(0)           # NOTE point.shape: [1, 108724, 3]

        # total = torch.tensor(list(range(point.shape[1])), device=unique_on_pixels.device)
        # mask = ~torch.isin(total, unique_on_pixels)
        # unsegmented_mask = total[mask]
        # unsegmented_point = point[0, unsegmented_mask].unsqueeze(0)

        if render_debug:
            red_color = torch.tensor([1, 0, 0], device=point_cloud.device)  # RGB for red
            for indices in on_pixels:
                for idx in indices:
                    # Check for valid index (i.e., not -1)
                    if idx >= 0:
                        point_cloud.features_packed()[idx, :3] = red_color

            fragments = rasterizer(point_cloud)
            r = rasterizer.raster_settings.radius
            dists2 = fragments.dists.permute(0, 3, 1, 2)
            alphas = 1 - dists2 / (r * r)
            images, weights = self.compositor(
                fragments.idx.long().permute(0, 3, 1, 2),
                alphas,
                point_cloud.features_packed().permute(1, 0),
            )
            images = images.permute(0, 2, 3, 1)
            return unique_on_pixels, images
        else:
            return unique_on_pixels
    
    def _find_empty_pixels(self, point_cloud, cameras, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)
        r = rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        images, weights = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            point_cloud.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
        
        mask_hole = (fragments.idx.long()[..., 0].reshape(-1) == -1).reshape(self.img_res)
        return mask_hole

    def _render(self, point_cloud, cameras, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)
        r = rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        images, weights = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            point_cloud.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
        
        mask_hole = (fragments.idx.long()[..., 0].reshape(-1) == -1).reshape(self.img_res)
        if not render_kp:
            self._output['mask_hole'] = mask_hole

        # batch_size, img_res, img_res, points_per_pixel
        if self.enable_prune and self.training and not render_kp:
            n_points = self.pc.points.shape[0]
            # the first point for each pixel is visible
            visible_points = fragments.idx.long()[..., 0].reshape(-1)
            visible_points = visible_points[visible_points != -1]

            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

            # points with weights larger than prune_thresh are visible
            visible_points = fragments.idx.long().reshape(-1)[weights.reshape(-1) > self.prune_thresh]
            visible_points = visible_points[visible_points != -1]

            n_points = self.pc.points.shape[0]
            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

        return images

    def densify_point_cloud(self, verts, n):
        # verts: [1, 5023, 3]
        # Calculate pair-wise distance between points
        # PyTorch doesn't have a built-in for pairwise distances, we have to do it manually
        # Expand verts to [5023, 1, 3] and [1, 5023, 3] to get a tensor of shape [5023, 5023, 3]
        # where we can subtract and find distances
        diff = verts.expand(verts.size(1), verts.size(1), 3) - verts.transpose(0, 1).expand(verts.size(1), verts.size(1), 3)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-9)  # [5023, 5023] Add a small value to prevent NaN gradients

        # Set diagonal to infinity to ignore self-distance
        dist.fill_diagonal_(float('inf'))

        # Find the indices of the closest points
        dist_sorted, idx = dist.sort(dim=1)

        # For each point, find the n closest points, then calculate the midpoints
        midpoints = []
        for i in range(1, n + 1):
            # Take the average between the point and its ith nearest neighbor
            midpoints.append((verts.squeeze() + verts[0, idx[:, i]]) / 2)

        # Concatenate the original vertices with the midpoints along the point dimension
        new_verts = torch.cat([verts.squeeze()] + midpoints, dim=0).unsqueeze(0)
        
        return new_verts

    def forward(self, input):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        # shape_params = input["shape"]                               # NOTE BetaCancel
        batch_size = flame_pose.shape[0]
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)
        # verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose, shape_params=shape_params)              # NOTE BetaCancel

        if self.ghostbone:
            # identity transformation for body
            # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
            transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
        cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        # NOTE custom #########################
        if self.optimize_latent_code or self.optimize_scene_latent_code:
            network_condition = dict()
        else:
            network_condition = None

        if self.optimize_latent_code:
            network_condition['latent'] = input["latent_code"] # [1, 32]
        
        if self.optimize_scene_latent_code:
            # if self.test_target_finetuning and not self.training:
            #     segment_mask = input['segment_mask']
            #     expanded_target = input['target_scene_latent_code'].clone().expand(total_points, -1)
            #     expanded_source = input["source_scene_latent_code"].clone().expand(segment_mask.shape[0], -1)

            #     scene_latent_code = expanded_target.clone()
            #     scene_latent_code[segment_mask] = expanded_source
                
            #     network_condition['scene_latent'] = scene_latent_code
                
            #     unique_tensor = torch.unique(network_condition['scene_latent'], dim=0)
            #     assert unique_tensor.shape[0] != 1, 'source_scene_latent_code and target_scene_latent_code are same.'


            #     # expanded_target = input['target_scene_latent_code'].clone().expand(total_points, -1)
            #     # scene_latent_code = expanded_target.clone()
            #     # network_condition['scene_latent'] = scene_latent_code
            # else:
            network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]
            # network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]

        ######################################
        # NOTE shape blendshape를 FLAME에서 그대로 갖다쓰기 위해 수정한 코드
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)
        # gt_beta_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, :100]

        canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(p, network_condition)      # NOTE network_condition 추가

        transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                                                     normals=canonical_normals,
                                                                                                                     feature_vectors=feature_vector,
                                                                                                                     pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                                     cond=network_condition) # NOTE network_condition 추가

        # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        # p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        # # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach(), cond=network_condition) # NOTE network_condition 추가
        # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가

        # NOTE transformed_points: x_d
        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        rgb_points = rgb_points.reshape(batch_size, n_points, 3)
        # point feature to rasterize and composite

        if 'middle_inference' in input:
            middle_inference = input['middle_inference']
            target_human_rgb_points = middle_inference['rgb_points']
            target_human_normals_points = middle_inference['normals_points']
            target_human_shading_points = middle_inference['shading_points']
            target_human_albedo_points = middle_inference['albedo_points']
            target_human_transformed_points = middle_inference['transformed_points']
            if 'category' in input and input['category'] == 'hair':
                target_human_flame_transformed_points = middle_inference['flame_transformed_points']

        if not self.training and 'masked_point_cloud_indices' in input:
            input['masked_point_cloud_indices'] = (torch.sigmoid(self._output['binary_segment']).squeeze() > 0.5).float()
            rgb_points_bak = rgb_points.clone()
            rgb_points = rgb_points[:, input['masked_point_cloud_indices'].bool(), :]
            if 'middle_inference' in input:
                rgb_points = torch.cat([rgb_points, target_human_rgb_points.to(rgb_points.device)], dim=1)

        features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)        # NOTE [batch_size, num of PCD, num of RGB features (4)]
        
        if self.normal and self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)
            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)                # NOTE [batch_size, num of PCD, num of normal features (3)]
        
        if self.training and self.binary_segmentation:
            # render segment image
            segment_begin_index = features.shape[-1]
            num_segment = self._output['binary_segment'].shape[-1]
            segments_points = self._output['binary_segment'].reshape(batch_size, n_points, num_segment)
            features = torch.cat([features, segments_points], dim=-1)                           # NOTE [batch_size, num of PCD, num of segment features (19)]

        if not self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)

            if 'masked_point_cloud_indices' in input:
                normals_points_bak = normals_points.clone()
                normals_points = normals_points[:, input['masked_point_cloud_indices'].bool(), :]
                if 'middle_inference' in input:
                    normals_points = torch.cat([normals_points, target_human_normals_points.to(normals_points.device)], dim=1)

            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

            shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1] + 3
            albedo_points = torch.clamp(albedo_points, 0., 1.)

            shading_points = shading_points.reshape(batch_size, n_points, 3)
            albedo_points = albedo_points.reshape(batch_size, n_points, 3)

            if 'masked_point_cloud_indices' in input:
                shading_points = shading_points[:, input['masked_point_cloud_indices'].bool(), :]
                albedo_points = albedo_points[:, input['masked_point_cloud_indices'].bool(), :]
                if 'middle_inference' in input:
                    shading_points = torch.cat([shading_points, target_human_shading_points.to(shading_points.device)], dim=1)
                    albedo_points = torch.cat([albedo_points, target_human_albedo_points.to(albedo_points.device)], dim=1)

            features = torch.cat([features, shading_points, albedo_points], dim=-1)   # NOTE shading: [3], albedo: [3]

        if not self.training and 'masked_point_cloud_indices' in input:
            transformed_points_bak = transformed_points.clone()
            transformed_points = transformed_points[:, input['masked_point_cloud_indices'].bool(), :]
        
            if 'middle_inference' in input:
                # delta_transformed_points = middle_inference['transformed_points'].mean(dim=1).cuda() - transformed_points.mean(dim=1)
                # target_human_transformed_points = target_human_transformed_points.cuda() - delta_transformed_points
                transformed_points = torch.cat([transformed_points, target_human_transformed_points.cuda()], dim=1)

        if 'target_human_values' in input:
            masked_point_cloud_indices = (torch.sigmoid(self._output['binary_segment']).squeeze() < 0.5).float()
            middle_inference = {
                'rgb_points': rgb_points[:, masked_point_cloud_indices.bool(), :],
                'normals_points': normals_points.unsqueeze(0)[:, masked_point_cloud_indices.bool(), :],
                'shading_points': shading_points.unsqueeze(0)[:, masked_point_cloud_indices.bool(), :],
                'albedo_points': albedo_points.unsqueeze(0)[:, masked_point_cloud_indices.bool(), :],
                'transformed_points': transformed_points[:, masked_point_cloud_indices.bool(), :],
                'pcd_center': self.pc.points.detach().mean(dim=0)
            }
            # NOTE target human의 hair도 들고와서 그 point를 들고온다.
            if 'category' in input and input['category'] == 'hair':
                masked_point_cloud_indices = (torch.sigmoid(self._output['binary_segment']).squeeze() > 0.4).float()
                selected_indices = torch.from_numpy(np.load('../utility/selected_indices.npy'))
                selected_mask = ~torch.isin(index_batch, selected_indices.cuda())
                final_vertices = torch.unique(verts[:, index_batch[selected_mask], :][:, masked_point_cloud_indices[selected_mask].bool(), :], dim=1)
                flame_transformed_points = self.densify_point_cloud(final_vertices, final_vertices.shape[1]-5)    
                middle_inference['flame_transformed_points'] = flame_transformed_points

            # NOTE FLAME을 들고오는 방법.
            # if 'category' in input and input['category'] == 'hair':
            #     masked_point_cloud_indices = (torch.sigmoid(self._output['binary_segment']).squeeze() > 0.5).float()

            #     flame_transformed_points = self.densify_point_cloud(torch.unique(verts[:, index_batch, :][:, masked_point_cloud_indices.bool(), :], dim=1), 20)       
            #     flame_rgb_points = torch.zeros_like(flame_transformed_points)
            #     flame_normals_points = torch.zeros_like(flame_transformed_points)
            #     flame_shading_points = torch.zeros_like(flame_transformed_points)
            #     flame_albedo_points = torch.zeros_like(flame_transformed_points)

            #     middle_inference['transformed_points'] = torch.cat([middle_inference['transformed_points'], flame_transformed_points], dim=1)
            #     middle_inference['rgb_points'] = torch.cat([middle_inference['rgb_points'], flame_rgb_points], dim=1)
            #     middle_inference['normals_points'] = torch.cat([middle_inference['normals_points'], flame_normals_points], dim=1)
            #     middle_inference['shading_points'] = torch.cat([middle_inference['shading_points'], flame_shading_points], dim=1)
            #     middle_inference['albedo_points'] = torch.cat([middle_inference['albedo_points'], flame_albedo_points], dim=1)

            return middle_inference
        
        transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

        if 'masked_point_cloud_indices' in input:
            mask_hole_empty_pixels = self._find_empty_pixels(transformed_point_cloud, cameras)


            target_flame_rgb_points = torch.zeros_like(target_human_flame_transformed_points).cuda()
            target_flame_features = torch.cat([target_flame_rgb_points, torch.ones_like(target_flame_rgb_points[..., [0]])], dim=-1)
            target_flame_transformed_point_cloud = Pointclouds(points=target_human_flame_transformed_points.cuda(), features=target_flame_features)
            segment_mask = self._segment(target_flame_transformed_point_cloud, cameras, mask=mask_hole_empty_pixels, render_debug=False)
            segment_mask = segment_mask.detach().cpu()
            target_human_flame_transformed_points = target_human_flame_transformed_points[:, segment_mask, :]
            # foreground_mask = segmented_images[..., 3].reshape(-1, 1)
            # if not self.use_background:
            #     rgb_values = segmented_images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
            # else:
            #     bkgd = torch.sigmoid(self.background * 100)
            #     rgb_values = segmented_images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
            # rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

            # # Convert tensor to numpy and squeeze the batch dimension
            # img_np = rgb_image.squeeze(0).detach().cpu().numpy()

            # # Convert range from [0, 1] to [0, 255] if needed
            # if img_np.max() <= 1:
            #     img_np = (img_np * 255).astype('uint8')

            # # Swap RGB to BGR
            # img_bgr = img_np[:, :, [2, 1, 0]]

            # # Save the image
            # cv2.imwrite('segmented_images.png', img_bgr)

            transformed_points = torch.cat([transformed_points, target_human_flame_transformed_points.cuda()], dim=1)

            target_human_flame_rgb_points = torch.zeros_like(target_human_flame_transformed_points)
            rgb_points = torch.cat([rgb_points, target_human_flame_rgb_points.to(rgb_points.device)], dim=1)

            features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1) 

            normal_begin_index = features.shape[-1]
            target_human_flame_normals_points = torch.zeros_like(target_human_flame_transformed_points)
            normals_points = torch.cat([normals_points, target_human_flame_normals_points.to(normals_points.device)], dim=1)
            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

            shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1] + 3

            target_human_flame_shading_points = torch.zeros_like(target_human_flame_transformed_points).cuda()
            target_human_flame_albedo_points = torch.zeros_like(target_human_flame_transformed_points).cuda()

            shading_points = torch.cat([shading_points, target_human_flame_shading_points], dim=1)
            albedo_points = torch.cat([albedo_points, target_human_flame_albedo_points], dim=1)

            features = torch.cat([features, shading_points, albedo_points], dim=-1)   # NOTE shading: [3], albedo: [3]

            transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.
            # if input['sub_dir'][0] in self.count_sub_dirs.keys():
            #     if self.count_sub_dirs[input['sub_dir'][0]] < self.num_views-1:
            #         self.count_sub_dirs[input['sub_dir'][0]] += 1
            #         self.voting_table[input['indices_tensor'], segment_mask, self.count_sub_dirs[input['sub_dir'][0]]] = 1
            # else:
            #     self.count_sub_dirs[input['sub_dir'][0]] = 0
            #     self.voting_table[input['indices_tensor'], segment_mask, 0] = 1

        # if self.training and ('rank' in input) and (input['rank'] == 0) and (not self.target_training):
        #     segment_mask = self._segment(transformed_point_cloud, cameras, mask=input['mask_object'], render_debug=False)
        #     segment_mask = segment_mask.detach().cpu()
        #     if input['sub_dir'][0] in self.count_sub_dirs.keys():
        #         if self.count_sub_dirs[input['sub_dir'][0]] < self.num_views-1:
        #             self.count_sub_dirs[input['sub_dir'][0]] += 1
        #             self.voting_table[input['indices_tensor'], segment_mask, self.count_sub_dirs[input['sub_dir'][0]]] = 1
        #     else:
        #         self.count_sub_dirs[input['sub_dir'][0]] = 0
        #         self.voting_table[input['indices_tensor'], segment_mask, 0] = 1

        images = self._render(transformed_point_cloud, cameras)

        if not self.training:
            # render landmarks for easier camera format debugging
            landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
            transformed_verts = Pointclouds(points=landmarks2d, features=torch.ones_like(landmarks2d))
            rendered_landmarks = self._render(transformed_verts, cameras, render_kp=True)

        foreground_mask = images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)

        if not self.training and 'masked_point_cloud_indices' in input:
            rgb_points = rgb_points_bak
            transformed_points = transformed_points_bak
            normals_points = normals_points_bak
        # knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        # flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        # index_batch = index_batch.reshape(-1)

        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # training outputs
        output = {
            'img_res': self.img_res,
            'batch_size': batch_size,
            'predicted_mask': foreground_mask,  # mask loss
            'rgb_image': rgb_image,
            'canonical_points': pnts_c_flame,
            # for flame loss
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights,
            'pcd_center': self.pc.points.mean(dim=0),
        }

        if self.normal and self.training:
            output['normal_image'] = (images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask)).reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        if self.training and self.binary_segmentation:
            output['segment_image'] = images[..., segment_begin_index:segment_begin_index+num_segment]

        if not self.training:
            output_testing = {
                'normal_image': images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'shading_image': images[..., shading_begin_index:shading_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'albedo_image': images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'rendered_landmarks': rendered_landmarks.reshape(-1, 3),
                'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
                'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
                'deformed_verts': verts.reshape(-1, 3),
                'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
                'pnts_normal_deformed': normals_points.reshape(batch_size, n_points, 3),
                #'pnts_normal_canonical': canonical_normals,
            }
            # if self.segment:
            #     output_testing['segment_image'] = images[..., segment_begin_index:segment_begin_index+num_segment].reshape(-1, 10)

            if self.deformer_network.deform_c:
                output_testing['unconstrained_canonical_points'] = self.pc.points
            output.update(output_testing)
        output.update(self._output)

        if not self.training and 'masked_point_cloud_indices' in input:
            self._output['mask_hole'] = self._output['mask_hole'].reshape(-1).unsqueeze(0)

        if self.optimize_scene_latent_code and self.training:
            output['scene_latent_code'] = input["scene_latent_code"]

        return output


    def get_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

        pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
        n_points = pnts_c.shape[0]                                                  # NOTE 400
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
            pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
            # NOTE custom #########################
            condition = {}
            condition['scene_latent'] = scene_latent
            #######################################
            # shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)           
            shapedirs = shapedirs.expand(batch_size, -1, -1)
            posedirs = posedirs.expand(batch_size, -1, -1)
            lbs_weights = lbs_weights.expand(batch_size, -1)
            pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
            # beta_shapedirs = beta_shapedirs.expand(batch_size, -1, -1)                                                          # NOTE beta cancel
            pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights) # FLAME-based deformed
            pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
            return pnts_d, pnts_d

        normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        # shapes = shapes.reshape(batch_size, n_points, *shapes.shape[1:]).transpose(0, 1)                                             # NOTE beta cancel
        if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
            scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent)
            # pnts_c: [400, 3], betas: [400, 1, 50], transformations: [400, 1, 6, 4, 4], pose_feature: [400, 1, 36], scene_latent: [400, 1, 32]
        else:
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        grads_inv = grads_batch.inverse()
        normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)
        shading = self.rendering_network(normals_d, cond) # TODO 여기다가 condition을 추가하면 어떻게 될까?????
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
        return pnts_d, rgb_vals, albedo, shading, normals_d


    def get_canonical_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None, shapes=None, gt_beta_shapedirs=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

        pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
        n_points = pnts_c.shape[0]                                                  # NOTE 400
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None, shapes=None, gt_beta_shapedirs=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
            pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
            # NOTE custom #########################
            condition = {}
            condition['scene_latent'] = scene_latent
            #######################################
            # shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)           # NOTE pnts_c_flame: [1, 3]
            # shapedirs = shapedirs.expand(batch_size, -1, -1)
            # posedirs = posedirs.expand(batch_size, -1, -1)
            # lbs_weights = lbs_weights.expand(batch_size, -1)
            # pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
            # beta_shapedirs = gt_beta_shapedirs.unsqueeze(0).expand(batch_size, -1, -1)
            # pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, beta_shapedirs=beta_shapedirs, shapes=shapes) # FLAME-based deformed
            # pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
            # return pnts_d, pnts_d
            pnts_c_flame = pnts_c_flame.reshape(-1)
            return pnts_c_flame, pnts_c_flame

        normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        shapes = shapes.reshape(batch_size, n_points, *shapes.shape[1:]).transpose(0, 1)                                             # NOTE beta cancel
        if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
            scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
            # grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
            grads_batch, pnts_c = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
        else:
            # grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)
            grads_batch, pnts_c = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        # pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        pnts_c = pnts_c.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        # grads_inv = grads_batch.inverse()
        # normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)

        # shading = self.rendering_network(normals_d, cond) 
        shading = self.rendering_network(normals, cond) 
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
        # return pnts_d, rgb_vals, albedo, shading, normals_d
        return pnts_c, rgb_vals, albedo, shading, normals






# class SceneLatentThreeStagesBlendingInferenceModel(nn.Module):
#     # NOTE SceneLatentThreeStages모델을 기반으로 만들었음. beta cancel이 적용되어있고 target human도 함께 들어가도록 설계되어있다.
#     # deform_cc를 추가하였음. 반드시 켜져있어야함.
#     def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background, checkpoint_path, pcd_init=None):
#         super().__init__()
#         shape_params = None
#         self.optimize_latent_code = conf.get_bool('train.optimize_latent_code')
#         self.optimize_scene_latent_code = conf.get_bool('train.optimize_scene_latent_code')

#         # FLAME_lightning
#         self.FLAMEServer = utils.get_class(conf.get_string('model.FLAME_class'))(conf=conf,
#                                                                                 flame_model_path='./flame/FLAME2020/generic_model.pkl', 
#                                                                                 lmk_embedding_path='./flame/FLAME2020/landmark_embedding.npy',
#                                                                                 n_shape=100,
#                                                                                 n_exp=50,
#                                                                                 # shape_params=shape_params,                            # NOTE BetaCancel
#                                                                                 canonical_expression=canonical_expression,
#                                                                                 canonical_pose=canonical_pose)                           # NOTE cuda 없앴음
        
#         # NOTE original code
#         # self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
#         #     self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
#         # self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)

#         self.prune_thresh = conf.get_float('model.prune_thresh', default=0.5)

#         # NOTE custom #########################
#         # scene latent를 위해 변형한 모델들이 들어감.
#         self.latent_code_dim = conf.get_int('model.latent_code_dim')
#         print('[DEBUG] latent_code_dim:', self.latent_code_dim)
#         # GeometryNetworkSceneLatent
#         self.geometry_network = utils.get_class(conf.get_string('model.geometry_class'))(optimize_latent_code=self.optimize_latent_code,
#                                                                                          optimize_scene_latent_code=self.optimize_scene_latent_code,
#                                                                                          latent_code_dim=self.latent_code_dim,
#                                                                                          **conf.get_config('model.geometry_network'))
#         # ForwardDeformerSceneLatentThreeStages
#         self.deformer_network = utils.get_class(conf.get_string('model.deformer_class'))(FLAMEServer=self.FLAMEServer,
#                                                                                         optimize_scene_latent_code=self.optimize_scene_latent_code,
#                                                                                         latent_code_dim=self.latent_code_dim,
#                                                                                         **conf.get_config('model.deformer_network'))
#         # RenderingNetworkSceneLatentThreeStages
#         self.rendering_network = utils.get_class(conf.get_string('model.rendering_class'))(optimize_scene_latent_code=self.optimize_scene_latent_code,
#                                                                                             latent_code_dim=self.latent_code_dim,
#                                                                                             **conf.get_config('model.rendering_network'))
#         #######################################

#         self.ghostbone = self.deformer_network.ghostbone

#         # NOTE custom #########################
#         self.test_target_finetuning = conf.get_bool('test.target_finetuning')
#         self.normal = True if conf.get_float('loss.normal_weight') > 0 and self.training else False
#         if checkpoint_path is not None:
#             print('[DEBUG] init point cloud from previous checkpoint')
#             # n_init_point를 checkpoint으로부터 불러오기 위해..
#             data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
#             n_init_points = data['state_dict']['model.pc.points'].shape[0]
#             init_radius = data['state_dict']['model.radius'].item()
#         elif pcd_init is not None:
#             print('[DEBUG] init point cloud from meta learning')
#             n_init_points = pcd_init['n_init_points']
#             init_radius = pcd_init['init_radius']
#         else:
#             print('[DEBUG] init point cloud from scratch')
#             n_init_points = 400
#             init_radius = 0.5

#         # PointCloudSceneLatentThreeStages
#         self.pc = utils.get_class(conf.get_string('model.pointcloud_class'))(n_init_points=n_init_points,
#                                                                             init_radius=init_radius,
#                                                                             **conf.get_config('model.point_cloud'))    # NOTE .cuda() 없앴음
#         #######################################

#         n_points = self.pc.points.shape[0]
#         self.img_res = img_res
#         self.use_background = use_background
#         if self.use_background:
#             init_background = torch.zeros(img_res[0] * img_res[1], 3).float()                   # NOTE .cuda() 없앰
#             # self.background = nn.Parameter(init_background)                                   # NOTE singleGPU코드에서는 이렇게 작성했지만,
#             self.register_parameter('background', nn.Parameter(init_background))                # NOTE 이렇게 수정해서 혹시나하는 버그를 방지해보고자 한다.
#         else:
#             # self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()           # NOTE singleGPU코드에서는 이렇게 작성했지만,
#             self.register_buffer('background', torch.ones(img_res[0] * img_res[1], 3).float())  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

#         # NOTE original code
#         self.raster_settings = PointsRasterizationSettings(image_size=img_res[0],
#                                                            radius=self.pc.radius_factor * (0.75 ** math.log2(n_points / 100)),
#                                                            points_per_pixel=10,
#                                                            bin_size=0)              # NOTE warning이 뜨길래 bin_size=0 추가함.
#         self.register_buffer('radius', torch.tensor(self.raster_settings.radius))
        
#         # keypoint rasterizer is only for debugging camera intrinsics
#         self.raster_settings_kp = PointsRasterizationSettings(image_size=self.img_res[0],
#                                                               radius=0.007,
#                                                               points_per_pixel=1)  

#         # NOTE ablation #########################################
#         self.enable_prune = conf.get_bool('train.enable_prune')

#         # self.visible_points = torch.zeros(n_points).bool().cuda()                             # NOTE singleGPU 코드에서는 이렇게 작성했지만,
#         if self.enable_prune:
#             if checkpoint_path is not None:
#                 # n_init_point를 checkpoint으로부터 불러오기 위해..
#                 data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
#                 visible_points = data['state_dict']['model.visible_points']                         # NOTE 이거 안해주면 visible이 0이 되어서 훈련이 안됨.
#             else:
#                 visible_points = torch.zeros(n_points).bool()
#             self.register_buffer('visible_points', visible_points)                                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.
#         # self.compositor = AlphaCompositor().cuda()                                            # NOTE singleGPU 코드에서는 이렇게 작성했지만,
#         self.compositor = AlphaCompositor()                                                     # NOTE cuda 할당이 자동으로 되도록 수정해본다.


#     def _compute_canonical_normals_and_feature_vectors(self, p, condition):
#         # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
#         # p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
#         # randomly sample some points in the neighborhood within 0.25 distance

#         # eikonal_points = torch.cat([p, p + (torch.rand(p.shape).cuda() - 0.5) * 0.5], dim=0)                          # NOTE original code, eikonal_points.shape: [6400, 3]
#         eikonal_points = torch.cat([p, p + (torch.rand(p.shape, device=p.device) - 0.5) * 0.5], dim=0)                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

#         if self.optimize_scene_latent_code:
#             condition['scene_latent_gradient'] = torch.cat([condition['scene_latent'], condition['scene_latent']], dim=0).detach()

#         eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points.detach(), condition)
#         n_points = self.pc.points.shape[0] # 400
#         canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1) # 400, 3

#         # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
#         # p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
#         geometry_output = self.geometry_network(p, condition)  # not using SDF to regularize point location, 3200, 4
#         sdf_values = geometry_output[:, 0]

#         feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector

#         if self.training and hasattr(self, "_output"):
#             self._output['sdf_values'] = sdf_values
#             self._output['grad_thetas'] = grad_thetas
#         if not self.training:
#             self._output['pnts_albedo'] = feature_vector

#         return canonical_normals, feature_vector # (400, 3), (400, 3) -> (400, 3) (3200, 3)

#     def _compute_canonical_normals_and_feature_vectors_for_blending(self, p_segmented, p_unsegmented, condition):
#         # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
#         # p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc

#         eikonal_points_segmented = torch.cat([p_segmented, p_segmented + (torch.rand(p_segmented.shape, device=p_segmented.device) - 0.5) * 0.5], dim=0)                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.
#         eikonal_points_unsegmented = torch.cat([p_unsegmented, p_unsegmented + (torch.rand(p_unsegmented.shape, device=p_unsegmented.device) - 0.5) * 0.5], dim=0)                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

#         if self.optimize_scene_latent_code:
#             condition['source_scene_latent_gradient'] = torch.cat([condition['source_scene_latent'], condition['source_scene_latent']], dim=0).detach()
#             condition['target_scene_latent_gradient'] = torch.cat([condition['target_scene_latent'], condition['target_scene_latent']], dim=0).detach()

#         eikonal_output, grad_thetas = self.geometry_network.gradient_blending(eikonal_points_segmented.detach(), eikonal_points_unsegmented.detach(), condition)
#         n_points = self.pc.points.shape[0] # 400
#         canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1) # 400, 3

#         # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
#         # p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc

#         geometry_output = self.geometry_network.forward_blending(p_segmented, p_unsegmented, condition)  # not using SDF to regularize point location, 3200, 4
#         sdf_values = geometry_output[:, 0]

#         feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector

#         if self.training and hasattr(self, "_output"):
#             self._output['sdf_values'] = sdf_values
#             self._output['grad_thetas'] = grad_thetas
#         if not self.training:
#             self._output['pnts_albedo'] = feature_vector

#         return canonical_normals, feature_vector # (400, 3), (400, 3) -> (400, 3) (3200, 3)
    
#     def _segment(self, point, point_cloud, cameras, mask, render_kp=False):
#         rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
#         fragments = rasterizer(point_cloud)

#         mask = mask > 127.5
#         on_pixels = fragments.idx[0, mask.reshape(1, 512, 512)[0]].long()

#         unique_on_pixels = torch.unique(on_pixels[on_pixels >= 0])      
#         segmented_point = point[0, unique_on_pixels].unsqueeze(0)           # NOTE point.shape: [1, 108724, 3]

#         total = torch.tensor(list(range(point.shape[1])), device=unique_on_pixels.device)
#         mask = ~torch.isin(total, unique_on_pixels)
#         unsegmented_mask = total[mask]
#         unsegmented_point = point[0, unsegmented_mask].unsqueeze(0)

#         red_color = torch.tensor([1, 0, 0]).to(point_cloud.device)  # RGB for red
#         for indices in on_pixels:
#             for idx in indices:
#                 # Check for valid index (i.e., not -1)
#                 if idx >= 0:
#                     point_cloud.features_packed()[idx, :3] = red_color

#         fragments = rasterizer(point_cloud)
#         r = rasterizer.raster_settings.radius
#         dists2 = fragments.dists.permute(0, 3, 1, 2)
#         alphas = 1 - dists2 / (r * r)
#         images, weights = self.compositor(
#             fragments.idx.long().permute(0, 3, 1, 2),
#             alphas,
#             point_cloud.features_packed().permute(1, 0),
#         )
#         images = images.permute(0, 2, 3, 1)
#         return unique_on_pixels, unsegmented_mask, images

#     def _render(self, point_cloud, cameras, render_kp=False):
#         rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
#         fragments = rasterizer(point_cloud)
#         r = rasterizer.raster_settings.radius
#         dists2 = fragments.dists.permute(0, 3, 1, 2)
#         alphas = 1 - dists2 / (r * r)
#         images, weights = self.compositor(
#             fragments.idx.long().permute(0, 3, 1, 2),
#             alphas,
#             point_cloud.features_packed().permute(1, 0),
#         )
#         images = images.permute(0, 2, 3, 1)
#         weights = weights.permute(0, 2, 3, 1)
#         # batch_size, img_res, img_res, points_per_pixel
#         if self.enable_prune and self.training and not render_kp:
#             n_points = self.pc.points.shape[0]
#             # the first point for each pixel is visible
#             visible_points = fragments.idx.long()[..., 0].reshape(-1)
#             visible_points = visible_points[visible_points != -1]

#             visible_points = visible_points % n_points
#             self.visible_points[visible_points] = True

#             # points with weights larger than prune_thresh are visible
#             visible_points = fragments.idx.long().reshape(-1)[weights.reshape(-1) > self.prune_thresh]
#             visible_points = visible_points[visible_points != -1]

#             n_points = self.pc.points.shape[0]
#             visible_points = visible_points % n_points
#             self.visible_points[visible_points] = True

#         return images

#     def face_parsing(self, img, label, device):
#         n_classes = 19
#         net = BiSeNet(n_classes=n_classes)
#         net.to(device)
#         save_pth = os.path.join(os.path.join(face_parsing_path, 'res', 'cp'), '79999_iter.pth')
#         net.load_state_dict(torch.load(save_pth))
#         net.eval()

#         to_tensor = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])

#         label_mapping = {
#             'skin': 1,
#             'eyebrows': 2,       # 2,3
#             'eyes': 4,           # 4,5 
#             'ears': 7,           # 7,8
#             'nose': 10,
#             'mouth': 11,         # 11,12,13 (lips)
#             'neck': 14,
#             'necklace': 15,
#             'cloth': 16,
#             'hair': 17,
#             'hat': 18,
#         }

#         with torch.no_grad():
#             img = Image.fromarray(img)
#             image = img.resize((512, 512), Image.BILINEAR)
#             img = to_tensor(image)
#             img = torch.unsqueeze(img, 0)
#             img = img.to(device)
#             out = net(img)[0]
#             parsing = out.squeeze(0).cpu().numpy().argmax(0)

#             condition = (parsing == label_mapping.get(label))
#             locations = np.where(condition)
#             mask_by_parsing = (condition).astype(np.uint8) * 255

#             if locations == []:
#                 print('[WARN] No object detected...')
#                 return []
#             else:
#                 return mask_by_parsing, torch.tensor(list(zip(locations[1], locations[0])))

#     def canonical_mask_generator(self, input, what_to_render):
#         self._output = {}
#         intrinsics = input["intrinsics"].clone()
#         cam_pose = input["cam_pose"].clone()
#         R = cam_pose[:, :3, :3]
#         T = cam_pose[:, :3, 3]
#         flame_pose = input["flame_pose"]
#         expression = input["expression"]
#         shape_params = input["shape"]                               # NOTE BetaCancel
#         batch_size = flame_pose.shape[0]
#         # verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)
#         verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose, shape_params=shape_params)              # NOTE BetaCancel

#         if self.ghostbone:
#             # identity transformation for body
#             # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
#             transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

#         # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
#         cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

#         # make sure the cameras focal length is logged too
#         focal_length = intrinsics[:, [0, 1], [0, 1]]
#         cameras.focal_length = focal_length
#         cameras.principal_point = cameras.get_principal_point()

#         n_points = self.pc.points.shape[0]
#         total_points = batch_size * n_points
#         # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

#         # NOTE custom #########################
#         if self.optimize_latent_code or self.optimize_scene_latent_code:
#             network_condition = dict()
#         else:
#             network_condition = None

#         if self.optimize_latent_code:
#             network_condition['latent'] = input["latent_code"] # [1, 32]
        
#         if self.optimize_scene_latent_code:
#             if self.test_target_finetuning and not self.training:
#                 if what_to_render == 'source':
#                     network_condition['scene_latent'] = input['source_scene_latent_code'].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
#                 elif what_to_render == 'target':
#                     network_condition['scene_latent'] = input['target_scene_latent_code'].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
#             else:
#                 network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]
            
#         ######################################
#         # if self.test_target_finetuning and not self.training and 'canonical_mask' not in input:
        
#         # NOTE mask를 source human의 canonical space에서 찾아야한다. 가장 간단한 방법은 deform 되기 전에 것을 들고오면 된다. 
#         p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
#         p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
#         _, _, _, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
#         knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
#         _, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        
#         index_batch = index_batch.reshape(-1)
#         gt_beta_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, :100]

#         canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(p, network_condition)      # NOTE network_condition 추가

#         flame_canonical_points, flame_canonical_rgb_points, _, _, _ = self.get_canonical_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
#                                                                                                             normals=canonical_normals,
#                                                                                                             feature_vectors=feature_vector,
#                                                                                                             pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
#                                                                                                             betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
#                                                                                                             transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
#                                                                                                             cond=network_condition,
#                                                                                                             shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
#                                                                                                             gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가
        
#         flame_canonical_points = flame_canonical_points.reshape(batch_size, n_points, 3)
#         flame_canonical_rgb_points = flame_canonical_rgb_points.reshape(batch_size, n_points, 3)
#         features = torch.cat([flame_canonical_rgb_points, torch.ones_like(flame_canonical_rgb_points[..., [0]])], dim=-1)           # NOTE [batch_size, num of PCD, num of RGB features (4)]
#         flame_canonical_point_cloud = Pointclouds(points=flame_canonical_points, features=features)                                     # NOTE pytorch3d's pointcloud class.

#         canonical_cameras = PerspectiveCameras(device=expression.device, R=R, T=torch.tensor([0, 0, 4]).unsqueeze(0), K=intrinsics)  
#         focal_length = intrinsics[:, [0, 1], [0, 1]] # make sure the cameras focal length is logged too
#         canonical_cameras.focal_length = focal_length
#         canonical_cameras.principal_point = canonical_cameras.get_principal_point()

#         images = self._render(flame_canonical_point_cloud, canonical_cameras)
#         foreground_mask = images[..., 3].reshape(-1, 1)
#         if not self.use_background:
#             rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
#         else:
#             bkgd = torch.sigmoid(self.background * 100)
#             rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
#         rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)
        
#         # Convert tensor to numpy and squeeze the batch dimension
#         img_np = rgb_image.squeeze(0).detach().cpu().numpy()

#         # Convert range from [0, 1] to [0, 255] if needed
#         if img_np.max() <= 1:
#             img_np = (img_np * 255).astype('uint8')

#         # Swap RGB to BGR
#         img_bgr = img_np[:, :, [2, 1, 0]]

#         # Save the image
#         cv2.imwrite('{}_output_image.png'.format(what_to_render), img_bgr)

#         canonical_mask_filename = '{}_canonical_mask.png'.format(what_to_render)
#         if not os.path.exists(canonical_mask_filename):
#             mask_parsing, _ = self.face_parsing(img_bgr, input['target_category'], device=expression.device)
#             cv2.imwrite(canonical_mask_filename, mask_parsing)
#         else:
#             mask_parsing = cv2.imread(canonical_mask_filename, cv2.IMREAD_GRAYSCALE)

#         segment_mask, unsegment_mask, segmented_images = self._segment(flame_canonical_points, flame_canonical_point_cloud, canonical_cameras, mask=mask_parsing)

#         foreground_mask = segmented_images[..., 3].reshape(-1, 1)
#         if not self.use_background:
#             rgb_values = segmented_images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
#         else:
#             bkgd = torch.sigmoid(self.background * 100)
#             rgb_values = segmented_images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
#         rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

#         # Convert tensor to numpy and squeeze the batch dimension
#         img_np = rgb_image.squeeze(0).detach().cpu().numpy()

#         # Convert range from [0, 1] to [0, 255] if needed
#         if img_np.max() <= 1:
#             img_np = (img_np * 255).astype('uint8')

#         # Swap RGB to BGR
#         img_bgr = img_np[:, :, [2, 1, 0]]

#         # Save the image
#         cv2.imwrite('{}_segmented_images.png'.format(what_to_render), img_bgr)

#         # del flame_canonical_points, flame_canonical_point_cloud, canonical_cameras, rgb_image, rgb_values, segmented_images, img_bgr
#         # torch.cuda.empty_cache()
#         return segment_mask, unsegment_mask
    
#     def forward(self, input):
#         self._output = {}
#         intrinsics = input["intrinsics"].clone()
#         cam_pose = input["cam_pose"].clone()
#         R = cam_pose[:, :3, :3]
#         T = cam_pose[:, :3, 3]
#         flame_pose = input["flame_pose"]
#         expression = input["expression"]
#         shape_params = input["shape"]                               # NOTE BetaCancel
#         batch_size = flame_pose.shape[0]
#         # verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)
#         verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose, shape_params=shape_params)              # NOTE BetaCancel

#         if self.ghostbone:
#             # identity transformation for body
#             # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
#             transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

#         # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
#         cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

#         # make sure the cameras focal length is logged too
#         focal_length = intrinsics[:, [0, 1], [0, 1]]
#         cameras.focal_length = focal_length
#         cameras.principal_point = cameras.get_principal_point()

#         n_points = self.pc.points.shape[0]
#         total_points = batch_size * n_points
#         # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

#         # NOTE custom #########################
#         if self.optimize_latent_code or self.optimize_scene_latent_code:
#             network_condition = dict()
#         else:
#             network_condition = None

#         if self.optimize_latent_code:
#             network_condition['latent'] = input["latent_code"] # [1, 32]
        
#         if self.optimize_scene_latent_code:
#             if self.test_target_finetuning and not self.training:
#                 # network_condition['scene_latent'] = input['target_scene_latent_code'].clone().unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
#                 # network_condition['scene_latent'] = torch.empty(total_points, input['source_scene_latent_code'].shape[1], device=input['source_scene_latent_code'].device)
#                 segment_mask = input['segment_mask']
#                 unsegment_mask = input['unsegment_mask']
#                 expanded_target = input['target_scene_latent_code'].clone().expand(total_points, -1)
#                 expanded_source = input["source_scene_latent_code"].clone().expand(segment_mask.shape[0], -1)

#                 scene_latent_code = expanded_target.clone()
#                 scene_latent_code[segment_mask] = expanded_source
#                 # segment_mask = torch.tensor(list(range(total_points // 2, total_points)), device=input['target_scene_latent_code'].device)
#                 # unsegment_mask = torch.tensor(list(range(total_points // 2)), device=input['target_scene_latent_code'].device)
                
#                 # network_condition['scene_latent'] = scene_latent_code
#                 network_condition['scene_latent'] = input['target_scene_latent_code'].clone().unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)
                
#                 # network_condition['scene_latent'][segment_mask] = input["source_scene_latent_code"].clone().expand(segment_mask.shape[0], -1)

#                 # network_condition['scene_latent'][unsegment_mask] = input["target_scene_latent_code"].expand(unsegment_mask.shape[0], -1)
#                 unique_tensor = torch.unique(network_condition['scene_latent'], dim=0)
#                 # assert unique_tensor.shape[0] != 1, 'source_scene_latent_code and target_scene_latent_code are same.'
#             else:
#                 network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1) # NOTE [1, 320] -> [150982, 320]
#         ######################################
#         # NOTE shape blendshape를 FLAME에서 그대로 갖다쓰기 위해 수정한 코드
#         p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
#         p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
#         shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
#         knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
#         flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
#         index_batch = index_batch.reshape(-1)
#         gt_beta_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, :100]

#         canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(p, network_condition)      # NOTE network_condition 추가

#         ######################################
#         flame_canonical_points, flame_canonical_rgb_points, _, _, _ = self.get_canonical_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
#                                                                                                             normals=canonical_normals,
#                                                                                                             feature_vectors=feature_vector,
#                                                                                                             pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
#                                                                                                             betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
#                                                                                                             transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
#                                                                                                             cond=network_condition,
#                                                                                                             shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
#                                                                                                             gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가

#         flame_canonical_points = flame_canonical_points.reshape(batch_size, n_points, 3)
#         flame_canonical_rgb_points = flame_canonical_rgb_points.reshape(batch_size, n_points, 3)
#         features = torch.cat([flame_canonical_rgb_points, torch.ones_like(flame_canonical_rgb_points[..., [0]])], dim=-1)           # NOTE [batch_size, num of PCD, num of RGB features (4)]
#         flame_canonical_point_cloud = Pointclouds(points=flame_canonical_points, features=features)                                     # NOTE pytorch3d's pointcloud class.

#         canonical_cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  
#         focal_length = intrinsics[:, [0, 1], [0, 1]] # make sure the cameras focal length is logged too
#         canonical_cameras.focal_length = focal_length
#         canonical_cameras.principal_point = canonical_cameras.get_principal_point()

#         images = self._render(flame_canonical_point_cloud, canonical_cameras)
#         foreground_mask = images[..., 3].reshape(-1, 1)
#         if not self.use_background:
#             rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
#         else:
#             bkgd = torch.sigmoid(self.background * 100)
#             rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
#         rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

#         # Convert tensor to numpy and squeeze the batch dimension
#         img_np = rgb_image.squeeze(0).detach().cpu().numpy()

#         # Convert range from [0, 1] to [0, 255] if needed
#         if img_np.max() <= 1:
#             img_np = (img_np * 255).astype('uint8')

#         # Swap RGB to BGR
#         img_bgr = img_np[:, :, [2, 1, 0]]

#         # Save the image
#         cv2.imwrite('output_image2.png', img_bgr)
#         ######################################


#         transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch(pnts_c=p,     # self.pc.points NOTE [400, 3]
#                                                                                                                      normals=canonical_normals,
#                                                                                                                      feature_vectors=feature_vector,
#                                                                                                                      pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
#                                                                                                                      betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
#                                                                                                                      transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
#                                                                                                                      cond=network_condition,
#                                                                                                                      shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
#                                                                                                                      gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가

#         # NOTE transformed_points: x_d
#         transformed_points = transformed_points.reshape(batch_size, n_points, 3)
#         rgb_points = rgb_points.reshape(batch_size, n_points, 3)
#         # point feature to rasterize and composite
#         features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)        # NOTE [batch_size, num of PCD, num of RGB features (4)]


#         # segmented_point = segmented_point.squeeze()
#         # unsegmented_point = unsegmented_point.squeeze()
        
#         # n_points_segmented = segmented_point.shape[0]
#         # n_points_unsegmented = unsegmented_point.shape[0]
#         # total_points_segmented = batch_size * n_points_segmented
#         # total_points_unsegmented = batch_size * n_points_unsegmented

#         # if self.optimize_scene_latent_code:
#         #     if self.test_target_finetuning and not self.training:
#         #         network_condition['source_scene_latent'] = input['source_scene_latent_code'].unsqueeze(1).expand(-1, n_points_segmented, -1).reshape(total_points_segmented, -1)
#         #         network_condition['target_scene_latent'] = input['target_scene_latent_code'].unsqueeze(1).expand(-1, n_points_unsegmented, -1).reshape(total_points_unsegmented, -1)

#         # # NOTE shape blendshape를 FLAME에서 그대로 갖다쓰기 위해 수정한 코드
#         # p = self.pc.points.detach()
#         # # p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
#         # mask_point = {'segment_mask': segment_mask, 'unsegment_mask': unsegment_mask}
#         # p = self.deformer_network.canocanonical_deform_blending(p, cond=network_condition, mask_point=mask_point)       # NOTE deform_cc
        
#         # segmented_point = p[segment_mask]
#         # unsegmented_point = p[unsegment_mask]
#         # # segmented_point = p[:n_points_segmented]
#         # # unsegmented_point = p[n_points_segmented:]

#         # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights_blending(segmented_point, unsegmented_point, cond=network_condition) # NOTE network_condition 추가
#         # knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
#         # flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
#         # index_batch = index_batch.reshape(-1)
#         # gt_beta_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, :100]

#         # canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors_for_blending(segmented_point, unsegmented_point, network_condition)      # NOTE network_condition 추가

#         # transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch_for_blending(pnts_c_segmented=segmented_point,     # NOTE [400, 3]
#         #                                                                                                                         pnts_c_unsegmented=unsegmented_point,
#         #                                                                                                                         segment_mask=segment_mask,
#         #                                                                                                                         unsegment_mask=unsegment_mask,
#         #                                                                                                                         normals=canonical_normals,
#         #                                                                                                                         feature_vectors=feature_vector,
#         #                                                                                                                         pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
#         #                                                                                                                         betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
#         #                                                                                                                         transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
#         #                                                                                                                         cond=network_condition,
#         #                                                                                                                         shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
#         #                                                                                                                         gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가

#         # # NOTE transformed_points: x_d
#         # transformed_points = transformed_points.reshape(batch_size, n_points, 3)
#         # rgb_points = rgb_points.reshape(batch_size, n_points, 3)
#         # # point feature to rasterize and composite
#         # features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)        # NOTE [batch_size, num of PCD, num of RGB features (4)]
        
#         if self.normal and self.training:
#             # render normal image
#             normal_begin_index = features.shape[-1]
#             normals_points = normals_points.reshape(batch_size, n_points, 3)
#             features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)                # NOTE [batch_size, num of PCD, num of normal features (3)]

#         if not self.training:
#             # render normal image
#             normal_begin_index = features.shape[-1]
#             normals_points = normals_points.reshape(batch_size, n_points, 3)
#             features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

#             shading_begin_index = features.shape[-1]
#             albedo_begin_index = features.shape[-1] + 3
#             albedo_points = torch.clamp(albedo_points, 0., 1.)
#             features = torch.cat([features, shading_points.reshape(batch_size, n_points, 3), albedo_points.reshape(batch_size, n_points, 3)], dim=-1)   # NOTE shading: [3], albedo: [3]

#         transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

#         # images = self._segment(transformed_points, transformed_point_cloud, cameras, mask=input['mask_object'])

#         images = self._render(transformed_point_cloud, cameras)

#         if not self.training:
#             # render landmarks for easier camera format debugging
#             landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
#             transformed_verts = Pointclouds(points=landmarks2d, features=torch.ones_like(landmarks2d))
#             rendered_landmarks = self._render(transformed_verts, cameras, render_kp=True)

#         foreground_mask = images[..., 3].reshape(-1, 1)
#         if not self.use_background:
#             rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
#         else:
#             bkgd = torch.sigmoid(self.background * 100)
#             rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)

#         # knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
#         # flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
#         # index_batch = index_batch.reshape(-1)

#         rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

#         # training outputs
#         output = {
#             'img_res': self.img_res,
#             'batch_size': batch_size,
#             'predicted_mask': foreground_mask,  # mask loss
#             'rgb_image': rgb_image,
#             'canonical_points': pnts_c_flame,
#             # for flame loss
#             'index_batch': index_batch,
#             'posedirs': posedirs,
#             'shapedirs': shapedirs,
#             'lbs_weights': lbs_weights,
#             'flame_posedirs': self.FLAMEServer.posedirs,
#             'flame_shapedirs': self.FLAMEServer.shapedirs,
#             'flame_lbs_weights': self.FLAMEServer.lbs_weights,
#         }

#         if self.normal and self.training:
#             output['normal_image'] = (images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask)).reshape(batch_size, self.img_res[0], self.img_res[1], 3)

#         if not self.training:
#             output_testing = {
#                 'normal_image': images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
#                 'shading_image': images[..., shading_begin_index:shading_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
#                 'albedo_image': images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
#                 'rendered_landmarks': rendered_landmarks.reshape(-1, 3),
#                 'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
#                 'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
#                 'deformed_verts': verts.reshape(-1, 3),
#                 'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
#                 'pnts_normal_deformed': normals_points.reshape(batch_size, n_points, 3),
#                 #'pnts_normal_canonical': canonical_normals,
#             }
#             if self.deformer_network.deform_c:
#                 output_testing['unconstrained_canonical_points'] = self.pc.points
#             output.update(output_testing)
#         output.update(self._output)

#         if self.optimize_scene_latent_code and self.training:
#             output['scene_latent_code'] = input["scene_latent_code"]

#         return output


#     def get_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None, shapes=None, gt_beta_shapedirs=None):
#         if pnts_c.shape[0] == 0:
#             return pnts_c.detach()
#         pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

#         # pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

#         total_points = betas.shape[0]
#         batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
#         n_points = pnts_c.shape[0]                                                  # NOTE 400
#         # pnts_c: n_points, 3
#         def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None, shapes=None, gt_beta_shapedirs=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
#             pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
#             # NOTE custom #########################
#             condition = {}
#             condition['scene_latent'] = scene_latent
#             #######################################
#             # shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
#             shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)           
#             shapedirs = shapedirs.expand(batch_size, -1, -1)
#             posedirs = posedirs.expand(batch_size, -1, -1)
#             lbs_weights = lbs_weights.expand(batch_size, -1)
#             pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
#             # beta_shapedirs = beta_shapedirs.expand(batch_size, -1, -1)                                                          # NOTE beta cancel
#             beta_shapedirs = gt_beta_shapedirs.unsqueeze(0).expand(batch_size, -1, -1)
#             pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, beta_shapedirs=beta_shapedirs, shapes=shapes) # FLAME-based deformed
#             pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
#             return pnts_d, pnts_d

#         normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
#         betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
#         transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
#         pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
#         shapes = shapes.reshape(batch_size, n_points, *shapes.shape[1:]).transpose(0, 1)                                             # NOTE beta cancel
#         if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
#             scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
#             grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
#             # pnts_c: [400, 3], betas: [400, 1, 50], transformations: [400, 1, 6, 4, 4], pose_feature: [400, 1, 36], scene_latent: [400, 1, 32]
#         else:
#             grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

#         pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
#         grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

#         grads_inv = grads_batch.inverse()
#         normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
#         feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
#         # some relighting code for inference
#         # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
#         # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)
#         shading = self.rendering_network(normals_d, cond) # TODO 여기다가 condition을 추가하면 어떻게 될까?????
#         albedo = feature_vectors
#         rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
#         return pnts_d, rgb_vals, albedo, shading, normals_d

#     def get_canonical_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None, shapes=None, gt_beta_shapedirs=None):
#         if pnts_c.shape[0] == 0:
#             return pnts_c.detach()
#         pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

#         pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

#         total_points = betas.shape[0]
#         batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
#         n_points = pnts_c.shape[0]                                                  # NOTE 400
#         # pnts_c: n_points, 3
#         def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None, shapes=None, gt_beta_shapedirs=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
#             pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
#             # NOTE custom #########################
#             condition = {}
#             condition['scene_latent'] = scene_latent
#             #######################################
#             # shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
#             shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)           # NOTE pnts_c_flame: [1, 3]
#             # shapedirs = shapedirs.expand(batch_size, -1, -1)
#             # posedirs = posedirs.expand(batch_size, -1, -1)
#             # lbs_weights = lbs_weights.expand(batch_size, -1)
#             # pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
#             # beta_shapedirs = gt_beta_shapedirs.unsqueeze(0).expand(batch_size, -1, -1)
#             # pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, beta_shapedirs=beta_shapedirs, shapes=shapes) # FLAME-based deformed
#             # pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
#             # return pnts_d, pnts_d
#             pnts_c_flame = pnts_c_flame.reshape(-1)
#             return pnts_c_flame, pnts_c_flame

#         normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
#         betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
#         transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
#         pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
#         shapes = shapes.reshape(batch_size, n_points, *shapes.shape[1:]).transpose(0, 1)                                             # NOTE beta cancel
#         if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
#             scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
#             # grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
#             grads_batch, pnts_c = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
#         else:
#             # grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)
#             grads_batch, pnts_c = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

#         # pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
#         pnts_c = pnts_c.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
#         grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

#         # grads_inv = grads_batch.inverse()
#         # normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
#         feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
#         # some relighting code for inference
#         # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
#         # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)

#         # shading = self.rendering_network(normals_d, cond) 
#         shading = self.rendering_network(normals, cond) 
#         albedo = feature_vectors
#         rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
#         # return pnts_d, rgb_vals, albedo, shading, normals_d
#         return pnts_c, rgb_vals, albedo, shading, normals


#     def get_rbg_value_functorch_for_blending(self, pnts_c_segmented, pnts_c_unsegmented, segment_mask, unsegment_mask, normals, feature_vectors, pose_feature, betas, transformations, cond=None, shapes=None, gt_beta_shapedirs=None):
#         if pnts_c_segmented.shape[0] == 0:
#             return pnts_c_segmented.detach()
#         pnts_c_segmented.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]
#         if pnts_c_unsegmented.shape[0] == 0:
#             return pnts_c_unsegmented.detach()
#         pnts_c_unsegmented.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

#         # pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc
#         pnts_c = torch.cat([pnts_c_segmented, pnts_c_unsegmented], dim=0)

#         total_points = betas.shape[0]
#         batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
#         n_points = pnts_c.shape[0]                                                  # NOTE 400
#         # pnts_c: n_points, 3
#         def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None, shapes=None, gt_beta_shapedirs=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
#             pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
#             # NOTE custom #########################
#             condition = {}
#             condition['scene_latent'] = scene_latent
#             #######################################
#             shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)           
#             shapedirs = shapedirs.expand(batch_size, -1, -1)
#             posedirs = posedirs.expand(batch_size, -1, -1)
#             lbs_weights = lbs_weights.expand(batch_size, -1)
#             pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
#             # beta_shapedirs = beta_shapedirs.expand(batch_size, -1, -1)                                                          # NOTE beta cancel
#             beta_shapedirs = gt_beta_shapedirs.unsqueeze(0).expand(batch_size, -1, -1)
#             pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, beta_shapedirs=beta_shapedirs, shapes=shapes) # FLAME-based deformed
#             pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
#             return pnts_d, pnts_d

#         normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
#         betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
#         transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
#         pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
#         shapes = shapes.reshape(batch_size, n_points, *shapes.shape[1:]).transpose(0, 1)                                             # NOTE beta cancel
        
#         if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
#             source_scene_latent = cond['source_scene_latent'].reshape(batch_size, n_points, *cond['source_scene_latent'].shape[1:]).transpose(0, 1)
#             target_scene_latent = cond['target_scene_latent'].reshape(batch_size, n_points, *cond['target_scene_latent'].shape[1:]).transpose(0, 1)

#             grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
#         else:
#             grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

#         pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
#         grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

#         grads_inv = grads_batch.inverse()
#         normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
#         feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
#         # some relighting code for inference
#         # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
#         # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)
#         shading = self.rendering_network(normals_d, cond) # TODO 여기다가 condition을 추가하면 어떻게 될까?????
#         albedo = feature_vectors
#         rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
#         return pnts_d, rgb_vals, albedo, shading, normals_d


class SceneLatentThreeStagesModelSingleGPU(nn.Module):
    # NOTE singleGPU를 기반으로 만들었음. SingleGPU와의 차이점을 NOTE로 기술함.
    # deform_cc를 추가하였음. 반드시 켜져있어야함. 
    def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background, checkpoint_path, pcd_init=None):
        super().__init__()
        self.optimize_latent_code = conf.get_bool('train.optimize_latent_code')
        self.optimize_scene_latent_code = conf.get_bool('train.optimize_scene_latent_code')

        # FLAME_lightning
        self.FLAMEServer = utils.get_class(conf.get_string('model.FLAME_class'))(conf=conf,
                                                                                flame_model_path='./flame/FLAME2020/generic_model.pkl', 
                                                                                lmk_embedding_path='./flame/FLAME2020/landmark_embedding.npy',
                                                                                n_shape=100,
                                                                                n_exp=50,
                                                                                shape_params=shape_params,
                                                                                canonical_expression=canonical_expression,
                                                                                canonical_pose=canonical_pose).cuda()                           # NOTE cuda 없앴음
        
        # NOTE original code
        # self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
        #     self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        # self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)

        self.prune_thresh = conf.get_float('model.prune_thresh', default=0.5)

        # NOTE custom #########################
        # scene latent를 위해 변형한 모델들이 들어감.
        self.latent_code_dim = conf.get_int('model.latent_code_dim')
        print('[DEBUG] latent_code_dim:', self.latent_code_dim)
        # GeometryNetworkSceneLatent
        self.geometry_network = utils.get_class(conf.get_string('model.geometry_class'))(optimize_latent_code=self.optimize_latent_code,
                                                                                         optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                         latent_code_dim=self.latent_code_dim,
                                                                                         **conf.get_config('model.geometry_network'))
        # ForwardDeformerSceneLatentThreeStages
        self.deformer_network = utils.get_class(conf.get_string('model.deformer_class'))(FLAMEServer=self.FLAMEServer,
                                                                                        optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                        latent_code_dim=self.latent_code_dim,
                                                                                        **conf.get_config('model.deformer_network'))
        # RenderingNetworkSceneLatentThreeStages
        self.rendering_network = utils.get_class(conf.get_string('model.rendering_class'))(optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                            latent_code_dim=self.latent_code_dim,
                                                                                            **conf.get_config('model.rendering_network'))
        #######################################

        self.ghostbone = self.deformer_network.ghostbone

        # NOTE custom #########################
        self.normal = True if conf.get_float('loss.normal_weight') > 0 and self.training else False
        if checkpoint_path is not None:
            print('[DEBUG] init point cloud from previous checkpoint')
            # n_init_point를 checkpoint으로부터 불러오기 위해..
            data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            n_init_points = data['state_dict']['model.pc.points'].shape[0]
            init_radius = data['state_dict']['model.radius'].item()
        elif pcd_init is not None:
            print('[DEBUG] init point cloud from meta learning')
            n_init_points = pcd_init['n_init_points']
            init_radius = pcd_init['init_radius']
        else:
            print('[DEBUG] init point cloud from scratch')
            n_init_points = 400
            init_radius = 0.5

        # PointCloudSceneLatentThreeStages
        self.pc = utils.get_class(conf.get_string('model.pointcloud_class'))(n_init_points=n_init_points,
                                                                            init_radius=init_radius,
                                                                            **conf.get_config('model.point_cloud')).cuda()     # NOTE .cuda() 없앴음
        #######################################

        n_points = self.pc.points.shape[0]
        self.img_res = img_res
        self.use_background = use_background
        if self.use_background:
            init_background = torch.zeros(img_res[0] * img_res[1], 3).float().cuda()                   # NOTE .cuda() 없앰
            # self.background = nn.Parameter(init_background)                                   # NOTE singleGPU코드에서는 이렇게 작성했지만,
            self.register_parameter('background', nn.Parameter(init_background))                # NOTE 이렇게 수정해서 혹시나하는 버그를 방지해보고자 한다.
        else:
            # self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()           # NOTE singleGPU코드에서는 이렇게 작성했지만,
            self.register_buffer('background', torch.ones(img_res[0] * img_res[1], 3).float())  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # NOTE original code
        self.raster_settings = PointsRasterizationSettings(image_size=img_res[0],
                                                           radius=self.pc.radius_factor * (0.75 ** math.log2(n_points / 100)),
                                                           points_per_pixel=10)
        self.register_buffer('radius', torch.tensor(self.raster_settings.radius))
        
        # keypoint rasterizer is only for debugging camera intrinsics
        self.raster_settings_kp = PointsRasterizationSettings(image_size=self.img_res[0],
                                                              radius=0.007,
                                                              points_per_pixel=1)

        # NOTE ablation #########################################
        self.enable_prune = conf.get_bool('train.enable_prune')

        # self.visible_points = torch.zeros(n_points).bool().cuda()                             # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        if self.enable_prune:
            if checkpoint_path is not None:
                # n_init_point를 checkpoint으로부터 불러오기 위해..
                data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                visible_points = data['state_dict']['model.visible_points']                         # NOTE 이거 안해주면 visible이 0이 되어서 훈련이 안됨.
            else:
                visible_points = torch.zeros(n_points).bool()
            self.register_buffer('visible_points', visible_points)                                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.
        # self.compositor = AlphaCompositor().cuda()                                            # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        self.compositor = AlphaCompositor().cuda()                                                     # NOTE cuda 할당이 자동으로 되도록 수정해본다.


    def _compute_canonical_normals_and_feature_vectors(self, condition):
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
        # randomly sample some points in the neighborhood within 0.25 distance

        # eikonal_points = torch.cat([p, p + (torch.rand(p.shape).cuda() - 0.5) * 0.5], dim=0)                          # NOTE original code, eikonal_points.shape: [6400, 3]
        eikonal_points = torch.cat([p, p + (torch.rand(p.shape, device=p.device) - 0.5) * 0.5], dim=0)                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        if self.optimize_scene_latent_code:
            condition['scene_latent_gradient'] = torch.cat([condition['scene_latent'], condition['scene_latent']], dim=0).detach()

        eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points.detach(), condition)
        n_points = self.pc.points.shape[0] # 400
        canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1) # 400, 3

        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
        geometry_output = self.geometry_network(p, condition)  # not using SDF to regularize point location, 3200, 4
        sdf_values = geometry_output[:, 0]

        feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector

        if self.training and hasattr(self, "_output"):
            self._output['sdf_values'] = sdf_values
            self._output['grad_thetas'] = grad_thetas
        if not self.training:
            self._output['pnts_albedo'] = feature_vector

        return canonical_normals, feature_vector # (400, 3), (400, 3) -> (400, 3) (3200, 3)

    def _render(self, point_cloud, cameras, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)
        r = rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        images, weights = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            point_cloud.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
        # batch_size, img_res, img_res, points_per_pixel
        if self.enable_prune and self.training and not render_kp:
            n_points = self.pc.points.shape[0]
            # the first point for each pixel is visible
            visible_points = fragments.idx.long()[..., 0].reshape(-1)
            visible_points = visible_points[visible_points != -1]

            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

            # points with weights larger than prune_thresh are visible
            visible_points = fragments.idx.long().reshape(-1)[weights.reshape(-1) > self.prune_thresh]
            visible_points = visible_points[visible_points != -1]

            n_points = self.pc.points.shape[0]
            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

        return images

    def forward(self, input):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        batch_size = flame_pose.shape[0]
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)

        if self.ghostbone:
            # identity transformation for body
            # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
            transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
        cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        # NOTE custom #########################
        if self.optimize_latent_code or self.optimize_scene_latent_code:
            network_condition = dict()
        else:
            network_condition = None

        if self.optimize_latent_code:
            network_condition['latent'] = input["latent_code"] # [1, 32]
        if self.optimize_scene_latent_code:
            network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)   
        ######################################

        canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(network_condition)      # NOTE network_condition 추가

        transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                                                     normals=canonical_normals,
                                                                                                                     feature_vectors=feature_vector,
                                                                                                                     pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                                     cond=network_condition) # NOTE network_condition 추가

        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach(), cond=network_condition) # NOTE network_condition 추가
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        # NOTE transformed_points: x_d
        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        rgb_points = rgb_points.reshape(batch_size, n_points, 3)
        # point feature to rasterize and composite
        features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)

        if self.normal:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)
            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

        if not self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)
            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

            shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1] + 3
            albedo_points = torch.clamp(albedo_points, 0., 1.)
            features = torch.cat([features, shading_points.reshape(batch_size, n_points, 3), albedo_points.reshape(batch_size, n_points, 3)], dim=-1)

        transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

        images = self._render(transformed_point_cloud, cameras)

        if not self.training:
            # render landmarks for easier camera format debugging
            landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
            transformed_verts = Pointclouds(points=landmarks2d, features=torch.ones_like(landmarks2d))
            rendered_landmarks = self._render(transformed_verts, cameras, render_kp=True)

        foreground_mask = images[..., 3].reshape(-1, 1)
        if not self.use_background:             # NOTE True
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)

        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)

        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # training outputs
        output = {
            'img_res': self.img_res,
            'batch_size': batch_size,
            'predicted_mask': foreground_mask,  # mask loss
            'rgb_image': rgb_image,
            'canonical_points': pnts_c_flame,
            # for flame loss
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights,
        }

        if self.normal:
            output['normal_image'] = (images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask)).reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        if not self.training:
            output_testing = {
                'normal_image': images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'shading_image': images[..., shading_begin_index:shading_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'albedo_image': images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'rendered_landmarks': rendered_landmarks.reshape(-1, 3),
                'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
                'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
                'deformed_verts': verts.reshape(-1, 3),
                'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
                'pnts_normal_deformed': normals_points.reshape(batch_size, n_points, 3),
                #'pnts_normal_canonical': canonical_normals,
            }
            if self.deformer_network.deform_c:
                output_testing['unconstrained_canonical_points'] = self.pc.points
            output.update(output_testing)
        output.update(self._output)
        if self.optimize_scene_latent_code:
            output['scene_latent_code'] = input["scene_latent_code"]

        return output


    def get_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

        pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
        n_points = pnts_c.shape[0]                                                  # NOTE 400
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
            pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
            # NOTE custom #########################
            condition = {}
            condition['scene_latent'] = scene_latent
            #######################################
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
            shapedirs = shapedirs.expand(batch_size, -1, -1)
            posedirs = posedirs.expand(batch_size, -1, -1)
            lbs_weights = lbs_weights.expand(batch_size, -1)
            pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
            pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights) # FLAME-based deformed
            pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
            return pnts_d, pnts_d

        normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
            scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent)
            # pnts_c: [400, 3], betas: [400, 1, 50], transformations: [400, 1, 6, 4, 4], pose_feature: [400, 1, 36], scene_latent: [400, 1, 32]
        else:
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        grads_inv = grads_batch.inverse()
        normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)
        shading = self.rendering_network(normals_d, cond) # TODO 여기다가 condition을 추가하면 어떻게 될까?????
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
        return pnts_d, rgb_vals, albedo, shading, normals_d
    



class SceneLatentThreeStagesModelShapeCancel(nn.Module):
    # NOTE 
    # * Shape Cancel이 적용된 부분에 대해 주석을 다 달아두었음.
    # * SingleGPU만 제거했음. 즉, cuda파트만 대체했다고 보면 됨.
    def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background, checkpoint_path, pcd_init=None):
        shape_params = None
        super().__init__()
        self.optimize_latent_code = conf.get_bool('train.optimize_latent_code')
        self.optimize_scene_latent_code = conf.get_bool('train.optimize_scene_latent_code')

        # NOTE shape_params 삭제
        self.FLAMEServer = utils.get_class(conf.get_string('model.FLAME_class'))(conf=conf,
                                                                                flame_model_path='./flame/FLAME2020/generic_model.pkl', 
                                                                                lmk_embedding_path='./flame/FLAME2020/landmark_embedding.npy',
                                                                                n_shape=100,
                                                                                n_exp=50,
                                                                                canonical_expression=canonical_expression,
                                                                                canonical_pose=canonical_pose)                           # NOTE cuda 없앴음
        
        # NOTE original code
        # self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
        #     self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        # self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)

        self.prune_thresh = conf.get_float('model.prune_thresh', default=0.5)

        # NOTE custom #########################
        # scene latent를 위해 변형한 모델들이 들어감.
        self.latent_code_dim = conf.get_int('model.latent_code_dim')
        print('[DEBUG] latent_code_dim:', self.latent_code_dim)
        # GeometryNetworkSceneLatent
        self.geometry_network = utils.get_class(conf.get_string('model.geometry_class'))(optimize_latent_code=self.optimize_latent_code,
                                                                                         optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                         latent_code_dim=self.latent_code_dim,
                                                                                         **conf.get_config('model.geometry_network'))
        # ForwardDeformerSceneLatentThreeStages
        self.deformer_network = utils.get_class(conf.get_string('model.deformer_class'))(FLAMEServer=self.FLAMEServer,
                                                                                        optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                        latent_code_dim=self.latent_code_dim,
                                                                                        **conf.get_config('model.deformer_network'))
        # RenderingNetworkSceneLatentThreeStages
        self.rendering_network = utils.get_class(conf.get_string('model.rendering_class'))(optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                            latent_code_dim=self.latent_code_dim,
                                                                                            **conf.get_config('model.rendering_network'))
        #######################################

        self.ghostbone = self.deformer_network.ghostbone

        # NOTE custom #########################
        if checkpoint_path is not None:
            print('[DEBUG] init point cloud from previous checkpoint')
            # n_init_point를 checkpoint으로부터 불러오기 위해..
            data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            n_init_points = data['state_dict']['model.pc.points'].shape[0]
            init_radius = data['state_dict']['model.radius'].item()
        elif pcd_init is not None:
            print('[DEBUG] init point cloud from meta learning')
            n_init_points = pcd_init['n_init_points']
            init_radius = pcd_init['init_radius']
        else:
            print('[DEBUG] init point cloud from scratch')
            n_init_points = 400
            init_radius = self.pc.radius_factor * (0.75 ** math.log2(n_points / 100)) # 0.5

        # PointCloudSceneLatentThreeStages
        self.pc = utils.get_class(conf.get_string('model.pointcloud_class'))(n_init_points=n_init_points,
                                                                            init_radius=init_radius,
                                                                            **conf.get_config('model.point_cloud'))     # NOTE .cuda() 없앴음
        #######################################

        n_points = self.pc.points.shape[0]
        self.img_res = img_res
        self.use_background = use_background
        if self.use_background:
            init_background = torch.zeros(img_res[0] * img_res[1], 3).float()                   # NOTE .cuda() 없앰
            # self.background = nn.Parameter(init_background)                                   # NOTE singleGPU코드에서는 이렇게 작성했지만,
            self.register_parameter('background', nn.Parameter(init_background))                # NOTE 이렇게 수정해서 혹시나하는 버그를 방지해보고자 한다.
        else:
            # self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()           # NOTE singleGPU코드에서는 이렇게 작성했지만,
            self.register_buffer('background', torch.ones(img_res[0] * img_res[1], 3).float())  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # NOTE original code
        self.raster_settings = PointsRasterizationSettings(image_size=img_res[0],
                                                           radius=init_radius,
                                                           points_per_pixel=10)
        self.register_buffer('radius', torch.tensor(self.raster_settings.radius))
        
        # keypoint rasterizer is only for debugging camera intrinsics
        self.raster_settings_kp = PointsRasterizationSettings(image_size=self.img_res[0],
                                                              radius=0.007,
                                                              points_per_pixel=1)

        # NOTE ablation #########################################
        self.enable_prune = conf.get_bool('train.enable_prune')

        # self.visible_points = torch.zeros(n_points).bool().cuda()                             # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        if self.enable_prune:
            if checkpoint_path is not None:
                # n_init_point를 checkpoint으로부터 불러오기 위해..
                data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                visible_points = data['state_dict']['model.visible_points']                         # NOTE 이거 안해주면 visible이 0이 되어서 훈련이 안됨.
            else:
                visible_points = torch.zeros(n_points).bool()
            self.register_buffer('visible_points', visible_points)                                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.
        # self.compositor = AlphaCompositor().cuda()                                            # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        self.compositor = AlphaCompositor()                                                     # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        self.num_views = 100
        # NOTE 모든 view에 대해서 segment된 point indices를 다 저장한다.
        self.voting_table = torch.zeros((len(conf.get_list('dataset.train.sub_dir')), n_points, self.num_views))
        # NOTE voting_table의 마지막 view의 index를 지정하기 위해, 각 sub_dir이 몇번 나왔는지 세기 위해서 만들었다.
        self.count_sub_dirs = {}


    def _compute_canonical_normals_and_feature_vectors(self, condition):
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
        # randomly sample some points in the neighborhood within 0.25 distance

        # eikonal_points = torch.cat([p, p + (torch.rand(p.shape).cuda() - 0.5) * 0.5], dim=0)                          # NOTE original code, eikonal_points.shape: [6400, 3]
        eikonal_points = torch.cat([p, p + (torch.rand(p.shape, device=p.device) - 0.5) * 0.5], dim=0)                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        if self.optimize_scene_latent_code:
            condition['scene_latent_gradient'] = torch.cat([condition['scene_latent'], condition['scene_latent']], dim=0).detach()

        eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points.detach(), condition)
        n_points = self.pc.points.shape[0] # 400
        canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1) # 400, 3

        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
        geometry_output = self.geometry_network(p, condition)  # not using SDF to regularize point location, 3200, 4
        sdf_values = geometry_output[:, 0]

        feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector

        if self.training and hasattr(self, "_output"):
            self._output['sdf_values'] = sdf_values
            self._output['grad_thetas'] = grad_thetas
        if not self.training:
            self._output['pnts_albedo'] = feature_vector

        return canonical_normals, feature_vector # (400, 3), (400, 3) -> (400, 3) (3200, 3)

    def _segment(self, point_cloud, cameras, mask, render_debug=True, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)

        # mask = mask > 127.5
        mask = mask > 0.5
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask.cpu().numpy().astype(np.uint8), kernel, iterations=1)
        mask = torch.tensor(mask, device=point_cloud.device).bool()
        on_pixels = fragments.idx[0, mask.reshape(1, 512, 512)[0]].long()

        unique_on_pixels = torch.unique(on_pixels[on_pixels >= 0])      
        # segmented_point = point[0, unique_on_pixels].unsqueeze(0)           # NOTE point.shape: [1, 108724, 3]

        # total = torch.tensor(list(range(point.shape[1])), device=unique_on_pixels.device)
        # mask = ~torch.isin(total, unique_on_pixels)
        # unsegmented_mask = total[mask]
        # unsegmented_point = point[0, unsegmented_mask].unsqueeze(0)

        if render_debug:
            red_color = torch.tensor([1, 0, 0], device=point_cloud.device)  # RGB for red
            for indices in on_pixels:
                for idx in indices:
                    # Check for valid index (i.e., not -1)
                    if idx >= 0:
                        point_cloud.features_packed()[idx, :3] = red_color

            fragments = rasterizer(point_cloud)
            r = rasterizer.raster_settings.radius
            dists2 = fragments.dists.permute(0, 3, 1, 2)
            alphas = 1 - dists2 / (r * r)
            images, weights = self.compositor(
                fragments.idx.long().permute(0, 3, 1, 2),
                alphas,
                point_cloud.features_packed().permute(1, 0),
            )
            images = images.permute(0, 2, 3, 1)
            return unique_on_pixels, images
        else:
            return unique_on_pixels
        
    def _render(self, point_cloud, cameras, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)
        r = rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        images, weights = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            point_cloud.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
        # mask_hole = (fragments.idx.long()[..., 0] == -1).all(dim=-1).squeeze()
        mask_hole = (fragments.idx.long()[..., 0].reshape(-1) == -1).reshape(self.img_res)
        if not render_kp:
            self._output['mask_hole'] = mask_hole
        # batch_size, img_res, img_res, points_per_pixel
        if self.enable_prune and self.training and not render_kp:
            n_points = self.pc.points.shape[0]
            # the first point for each pixel is visible
            visible_points = fragments.idx.long()[..., 0].reshape(-1)
            visible_points = visible_points[visible_points != -1]

            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

            # points with weights larger than prune_thresh are visible
            visible_points = fragments.idx.long().reshape(-1)[weights.reshape(-1) > self.prune_thresh]
            visible_points = visible_points[visible_points != -1]

            n_points = self.pc.points.shape[0]
            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

        return images

    def forward(self, input):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        shape_params = input["shape"]          # NOTE beta-cancel
        batch_size = flame_pose.shape[0]
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose, shape_params=shape_params)

        if self.ghostbone:
            # identity transformation for body
            # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
            transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
        cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        # NOTE custom #########################
        if self.optimize_latent_code or self.optimize_scene_latent_code:
            network_condition = dict()
        else:
            network_condition = None

        if self.optimize_latent_code:
            network_condition['latent'] = input["latent_code"] # [1, 32]
        if self.optimize_scene_latent_code:
            network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)   
        ######################################

        # NOTE shape blendshape를 FLAME에서 그대로 갖다쓰기 위해 수정한 코드
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach(), cond=network_condition) # NOTE network_condition 추가
        # shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)
        gt_beta_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, :100]

        canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(network_condition)      # NOTE network_condition 추가

        transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                                                     normals=canonical_normals,
                                                                                                                     feature_vectors=feature_vector,
                                                                                                                     pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                                     cond=network_condition,
                                                                                                                     shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가
        # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        # p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        # # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach(), cond=network_condition) # NOTE network_condition 추가
        # shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        
        # NOTE transformed_points: x_d
        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        rgb_points = rgb_points.reshape(batch_size, n_points, 3)

        if 'masked_point_cloud_indices' in input:
            rgb_points_bak = rgb_points.clone()
            rgb_points = rgb_points[:, input['masked_point_cloud_indices'].bool(), :]

        # point feature to rasterize and composite
        features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)
        if not self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)

            if 'masked_point_cloud_indices' in input:
                normals_points_bak = normals_points.clone()
                normals_points = normals_points[:, input['masked_point_cloud_indices'].bool(), :]

            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

            shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1] + 3
            albedo_points = torch.clamp(albedo_points, 0., 1.)

            shading_points = shading_points.reshape(batch_size, n_points, 3)
            albedo_points = albedo_points.reshape(batch_size, n_points, 3)

            if 'masked_point_cloud_indices' in input:
                shading_points = shading_points[:, input['masked_point_cloud_indices'].bool(), :]
                albedo_points = albedo_points[:, input['masked_point_cloud_indices'].bool(), :]

            # features = torch.cat([features, shading_points.reshape(batch_size, n_points, 3), albedo_points.reshape(batch_size, n_points, 3)], dim=-1)
            features = torch.cat([features, shading_points, albedo_points], dim=-1)

        if 'masked_point_cloud_indices' in input:
            transformed_points_bak = transformed_points.clone()
            transformed_points = transformed_points[:, input['masked_point_cloud_indices'].bool(), :]

        if 'blending_middle_inference' in input:
            middle_inference = {
                'rgb_points': rgb_points,
                'normals_points': normals_points,
                'shading_points': shading_points,
                'albedo_points': albedo_points,
                'transformed_points': transformed_points
            }
            return middle_inference
        transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

        segment_mask = self._segment(transformed_point_cloud, cameras, mask=input['mask_source'], render_debug=False)
        segment_mask = segment_mask.detach().cpu()
        if input['sub_dir'][0] in self.count_sub_dirs.keys():
            if self.count_sub_dirs[input['sub_dir'][0]] < self.num_views-1:
                self.count_sub_dirs[input['sub_dir'][0]] += 1
                self.voting_table[0, segment_mask, self.count_sub_dirs[input['sub_dir'][0]]] = 1
        else:
            self.count_sub_dirs[input['sub_dir'][0]] = 0
            self.voting_table[0, segment_mask, 0] = 1

        images = self._render(transformed_point_cloud, cameras)

        if not self.training:
            # render landmarks for easier camera format debugging
            landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
            transformed_verts = Pointclouds(points=landmarks2d, features=torch.ones_like(landmarks2d))
            rendered_landmarks = self._render(transformed_verts, cameras, render_kp=True)

        foreground_mask = images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)

        # knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        # flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        # index_batch = index_batch.reshape(-1)
        if 'masked_point_cloud_indices' in input:
            rgb_points = rgb_points_bak
            transformed_points = transformed_points_bak
            normals_points = normals_points_bak

        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # training outputs
        output = {
            'img_res': self.img_res,
            'batch_size': batch_size,
            'predicted_mask': foreground_mask,  # mask loss
            'rgb_image': rgb_image,
            'canonical_points': pnts_c_flame,
            # for flame loss
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            # 'beta_shapedirs': beta_shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights
        }

        if not self.training:
            output_testing = {
                'normal_image': images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'shading_image': images[..., shading_begin_index:shading_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'albedo_image': images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'rendered_landmarks': rendered_landmarks.reshape(-1, 3),
                'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
                'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
                'deformed_verts': verts.reshape(-1, 3),
                'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
                'pnts_normal_deformed': normals_points.reshape(batch_size, n_points, 3),
                #'pnts_normal_canonical': canonical_normals,
            }
            if self.deformer_network.deform_c:
                output_testing['unconstrained_canonical_points'] = self.pc.points
            output.update(output_testing)
        
        if not self.training:
            self._output['mask_hole'] = self._output['mask_hole'].reshape(-1).unsqueeze(0) * (input['object_mask'])

        output.update(self._output)
        if self.optimize_scene_latent_code:
            output['scene_latent_code'] = input["scene_latent_code"]

        return output


    def get_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None, shapes=None, gt_beta_shapedirs=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

        pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
        n_points = pnts_c.shape[0]                                                  # NOTE 400
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None, shapes=None, gt_beta_shapedirs=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
            pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
            # NOTE custom #########################
            condition = {}
            condition['scene_latent'] = scene_latent
            #######################################
            # shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)           
            shapedirs = shapedirs.expand(batch_size, -1, -1)
            posedirs = posedirs.expand(batch_size, -1, -1)
            lbs_weights = lbs_weights.expand(batch_size, -1)
            pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
            # beta_shapedirs = beta_shapedirs.expand(batch_size, -1, -1)                                                          # NOTE beta cancel
            beta_shapedirs = gt_beta_shapedirs.unsqueeze(0).expand(batch_size, -1, -1)
            pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, beta_shapedirs=beta_shapedirs, shapes=shapes) # FLAME-based deformed
            pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
            return pnts_d, pnts_d

        normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        shapes = shapes.reshape(batch_size, n_points, *shapes.shape[1:]).transpose(0, 1)                                             # NOTE beta cancel
        if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
            scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
            # pnts_c: [400, 3], betas: [400, 1, 50], transformations: [400, 1, 6, 4, 4], pose_feature: [400, 1, 36], scene_latent: [400, 1, 32]
        else:
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        grads_inv = grads_batch.inverse()
        normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)
        shading = self.rendering_network(normals_d, cond) # TODO 여기다가 condition을 추가하면 어떻게 될까?????
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
        return pnts_d, rgb_vals, albedo, shading, normals_d
    



class SceneLatentThreeStagesModelSingleGPUShapeCancel(nn.Module):
    # NOTE 
    # * Shape Cancel이 적용된 부분에 대해 주석을 다 달아두었음.
    def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background, checkpoint_path, pcd_init=None):
        shape_params = None
        super().__init__()
        self.optimize_latent_code = conf.get_bool('train.optimize_latent_code')
        self.optimize_scene_latent_code = conf.get_bool('train.optimize_scene_latent_code')

        # NOTE shape_params 삭제
        self.FLAMEServer = utils.get_class(conf.get_string('model.FLAME_class'))(conf=conf,
                                                                                flame_model_path='./flame/FLAME2020/generic_model.pkl', 
                                                                                lmk_embedding_path='./flame/FLAME2020/landmark_embedding.npy',
                                                                                n_shape=100,
                                                                                n_exp=50,
                                                                                canonical_expression=canonical_expression,
                                                                                canonical_pose=canonical_pose).cuda()                           # NOTE cuda 없앴음
        
        # NOTE original code
        # self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
        #     self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        # self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)

        self.prune_thresh = conf.get_float('model.prune_thresh', default=0.5)

        # NOTE custom #########################
        # scene latent를 위해 변형한 모델들이 들어감.
        self.latent_code_dim = conf.get_int('model.latent_code_dim')
        print('[DEBUG] latent_code_dim:', self.latent_code_dim)
        # GeometryNetworkSceneLatent
        self.geometry_network = utils.get_class(conf.get_string('model.geometry_class'))(optimize_latent_code=self.optimize_latent_code,
                                                                                         optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                         latent_code_dim=self.latent_code_dim,
                                                                                         **conf.get_config('model.geometry_network'))
        # ForwardDeformerSceneLatentThreeStages
        self.deformer_network = utils.get_class(conf.get_string('model.deformer_class'))(FLAMEServer=self.FLAMEServer,
                                                                                        optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                        latent_code_dim=self.latent_code_dim,
                                                                                        **conf.get_config('model.deformer_network'))
        # RenderingNetworkSceneLatentThreeStages
        self.rendering_network = utils.get_class(conf.get_string('model.rendering_class'))(optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                            latent_code_dim=self.latent_code_dim,
                                                                                            **conf.get_config('model.rendering_network'))
        #######################################

        self.ghostbone = self.deformer_network.ghostbone

        # NOTE custom #########################
        if checkpoint_path is not None:
            print('[DEBUG] init point cloud from previous checkpoint')
            # n_init_point를 checkpoint으로부터 불러오기 위해..
            data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            n_init_points = data['state_dict']['model.pc.points'].shape[0]
            init_radius = data['state_dict']['model.radius'].item()
        elif pcd_init is not None:
            print('[DEBUG] init point cloud from meta learning')
            n_init_points = pcd_init['n_init_points']
            init_radius = pcd_init['init_radius']
        else:
            print('[DEBUG] init point cloud from scratch')
            n_init_points = 400
            init_radius = self.pc.radius_factor * (0.75 ** math.log2(n_points / 100)) # 0.5

        # PointCloudSceneLatentThreeStages
        self.pc = utils.get_class(conf.get_string('model.pointcloud_class'))(n_init_points=n_init_points,
                                                                            init_radius=init_radius,
                                                                            **conf.get_config('model.point_cloud')).cuda()     # NOTE .cuda() 없앴음
        #######################################

        n_points = self.pc.points.shape[0]
        self.img_res = img_res
        self.use_background = use_background
        if self.use_background:
            init_background = torch.zeros(img_res[0] * img_res[1], 3).float().cuda()                   # NOTE .cuda() 없앰
            # self.background = nn.Parameter(init_background)                                   # NOTE singleGPU코드에서는 이렇게 작성했지만,
            self.register_parameter('background', nn.Parameter(init_background))                # NOTE 이렇게 수정해서 혹시나하는 버그를 방지해보고자 한다.
        else:
            # self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()           # NOTE singleGPU코드에서는 이렇게 작성했지만,
            self.register_buffer('background', torch.ones(img_res[0] * img_res[1], 3).float())  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # NOTE original code
        self.raster_settings = PointsRasterizationSettings(image_size=img_res[0],
                                                           radius=init_radius,
                                                           points_per_pixel=10)
        self.register_buffer('radius', torch.tensor(self.raster_settings.radius))
        
        # keypoint rasterizer is only for debugging camera intrinsics
        self.raster_settings_kp = PointsRasterizationSettings(image_size=self.img_res[0],
                                                              radius=0.007,
                                                              points_per_pixel=1)

        # NOTE ablation #########################################
        self.enable_prune = conf.get_bool('train.enable_prune')

        # self.visible_points = torch.zeros(n_points).bool().cuda()                             # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        if self.enable_prune:
            if checkpoint_path is not None:
                # n_init_point를 checkpoint으로부터 불러오기 위해..
                data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                visible_points = data['state_dict']['model.visible_points']                         # NOTE 이거 안해주면 visible이 0이 되어서 훈련이 안됨.
            else:
                visible_points = torch.zeros(n_points).bool()
            self.register_buffer('visible_points', visible_points)                                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.
        # self.compositor = AlphaCompositor().cuda()                                            # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        self.compositor = AlphaCompositor().cuda()                                                     # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        self.num_views = 100
        # NOTE 모든 view에 대해서 segment된 point indices를 다 저장한다.
        # self.voting_table = torch.zeros((len(conf.get_list('dataset.train.sub_dir')), n_points, self.num_views))
        # NOTE voting_table의 마지막 view의 index를 지정하기 위해, 각 sub_dir이 몇번 나왔는지 세기 위해서 만들었다.
        # self.count_sub_dirs = {}


    def _compute_canonical_normals_and_feature_vectors(self, condition):
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
        # randomly sample some points in the neighborhood within 0.25 distance

        # eikonal_points = torch.cat([p, p + (torch.rand(p.shape).cuda() - 0.5) * 0.5], dim=0)                          # NOTE original code, eikonal_points.shape: [6400, 3]
        eikonal_points = torch.cat([p, p + (torch.rand(p.shape, device=p.device) - 0.5) * 0.5], dim=0)                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        if self.optimize_scene_latent_code:
            condition['scene_latent_gradient'] = torch.cat([condition['scene_latent'], condition['scene_latent']], dim=0).detach()

        eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points.detach(), condition)
        n_points = self.pc.points.shape[0] # 400
        canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1) # 400, 3

        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
        geometry_output = self.geometry_network(p, condition)  # not using SDF to regularize point location, 3200, 4
        sdf_values = geometry_output[:, 0]

        feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector

        if self.training and hasattr(self, "_output"):
            self._output['sdf_values'] = sdf_values
            self._output['grad_thetas'] = grad_thetas
        if not self.training:
            self._output['pnts_albedo'] = feature_vector

        return canonical_normals, feature_vector # (400, 3), (400, 3) -> (400, 3) (3200, 3)

    def _segment(self, point_cloud, cameras, mask, render_debug=True, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)

        # mask = mask > 127.5
        mask = mask > 0.5
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask.cpu().numpy().astype(np.uint8), kernel, iterations=1)
        mask = torch.tensor(mask, device=point_cloud.device).bool()
        on_pixels = fragments.idx[0, mask.reshape(1, 512, 512)[0]].long()

        unique_on_pixels = torch.unique(on_pixels[on_pixels >= 0])      
        # segmented_point = point[0, unique_on_pixels].unsqueeze(0)           # NOTE point.shape: [1, 108724, 3]

        # total = torch.tensor(list(range(point.shape[1])), device=unique_on_pixels.device)
        # mask = ~torch.isin(total, unique_on_pixels)
        # unsegmented_mask = total[mask]
        # unsegmented_point = point[0, unsegmented_mask].unsqueeze(0)

        if render_debug:
            red_color = torch.tensor([1, 0, 0], device=point_cloud.device)  # RGB for red
            for indices in on_pixels:
                for idx in indices:
                    # Check for valid index (i.e., not -1)
                    if idx >= 0:
                        point_cloud.features_packed()[idx, :3] = red_color

            fragments = rasterizer(point_cloud)
            r = rasterizer.raster_settings.radius
            dists2 = fragments.dists.permute(0, 3, 1, 2)
            alphas = 1 - dists2 / (r * r)
            images, weights = self.compositor(
                fragments.idx.long().permute(0, 3, 1, 2),
                alphas,
                point_cloud.features_packed().permute(1, 0),
            )
            images = images.permute(0, 2, 3, 1)
            return unique_on_pixels, images
        else:
            return unique_on_pixels
        
    def _render(self, point_cloud, cameras, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)
        r = rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        images, weights = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            point_cloud.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
        # mask_hole = (fragments.idx.long()[..., 0] == -1).all(dim=-1).squeeze()
        mask_hole = (fragments.idx.long()[..., 0].reshape(-1) == -1).reshape(self.img_res)
        if not render_kp:
            self._output['mask_hole'] = mask_hole
        # batch_size, img_res, img_res, points_per_pixel
        if self.enable_prune and self.training and not render_kp:
            n_points = self.pc.points.shape[0]
            # the first point for each pixel is visible
            visible_points = fragments.idx.long()[..., 0].reshape(-1)
            visible_points = visible_points[visible_points != -1]

            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

            # points with weights larger than prune_thresh are visible
            visible_points = fragments.idx.long().reshape(-1)[weights.reshape(-1) > self.prune_thresh]
            visible_points = visible_points[visible_points != -1]

            n_points = self.pc.points.shape[0]
            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

        return images

    def forward(self, input):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        shape_params = input["shape"]          # NOTE beta-cancel
        batch_size = flame_pose.shape[0]
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose, shape_params=shape_params)

        if self.ghostbone:
            # identity transformation for body
            # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
            transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
        cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        # NOTE custom #########################
        if self.optimize_latent_code or self.optimize_scene_latent_code:
            network_condition = dict()
        else:
            network_condition = None

        if self.optimize_latent_code:
            network_condition['latent'] = input["latent_code"] # [1, 32]
        if self.optimize_scene_latent_code:
            network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)   
        ######################################

        # NOTE shape blendshape를 FLAME에서 그대로 갖다쓰기 위해 수정한 코드
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach(), cond=network_condition) # NOTE network_condition 추가
        # shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)
        gt_beta_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, :100]

        canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(network_condition)      # NOTE network_condition 추가

        transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                                                     normals=canonical_normals,
                                                                                                                     feature_vectors=feature_vector,
                                                                                                                     pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                                     cond=network_condition,
                                                                                                                     shapes=shape_params.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     gt_beta_shapedirs=gt_beta_shapedirs) # NOTE network_condition 추가
        # p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        # p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        # # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach(), cond=network_condition) # NOTE network_condition 추가
        # shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        
        # NOTE transformed_points: x_d
        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        rgb_points = rgb_points.reshape(batch_size, n_points, 3)

        if 'masked_point_cloud_indices' in input:
            rgb_points_bak = rgb_points.clone()
            rgb_points = rgb_points[:, input['masked_point_cloud_indices'].bool(), :]

        # point feature to rasterize and composite
        features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)
        if not self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)

            if 'masked_point_cloud_indices' in input:
                normals_points_bak = normals_points.clone()
                normals_points = normals_points[:, input['masked_point_cloud_indices'].bool(), :]

            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

            shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1] + 3
            albedo_points = torch.clamp(albedo_points, 0., 1.)

            shading_points = shading_points.reshape(batch_size, n_points, 3)
            albedo_points = albedo_points.reshape(batch_size, n_points, 3)

            if 'masked_point_cloud_indices' in input:
                shading_points = shading_points[:, input['masked_point_cloud_indices'].bool(), :]
                albedo_points = albedo_points[:, input['masked_point_cloud_indices'].bool(), :]

            # features = torch.cat([features, shading_points.reshape(batch_size, n_points, 3), albedo_points.reshape(batch_size, n_points, 3)], dim=-1)
            features = torch.cat([features, shading_points, albedo_points], dim=-1)

        if 'masked_point_cloud_indices' in input:
            transformed_points_bak = transformed_points.clone()
            transformed_points = transformed_points[:, input['masked_point_cloud_indices'].bool(), :]

        if 'blending_middle_inference' in input:
            middle_inference = {
                'rgb_points': rgb_points,
                'normals_points': normals_points,
                'shading_points': shading_points,
                'albedo_points': albedo_points,
                'transformed_points': transformed_points
            }
            return middle_inference
        transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

        # segment_mask = self._segment(transformed_point_cloud, cameras, mask=input['mask_source'], render_debug=False)
        # segment_mask = segment_mask.detach().cpu()
        # if input['sub_dir'][0] in self.count_sub_dirs.keys():
        #     if self.count_sub_dirs[input['sub_dir'][0]] < self.num_views-1:
        #         self.count_sub_dirs[input['sub_dir'][0]] += 1
        #         self.voting_table[0, segment_mask, self.count_sub_dirs[input['sub_dir'][0]]] = 1
        # else:
        #     self.count_sub_dirs[input['sub_dir'][0]] = 0
        #     self.voting_table[0, segment_mask, 0] = 1

        images = self._render(transformed_point_cloud, cameras)

        if not self.training:
            # render landmarks for easier camera format debugging
            landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
            transformed_verts = Pointclouds(points=landmarks2d, features=torch.ones_like(landmarks2d))
            rendered_landmarks = self._render(transformed_verts, cameras, render_kp=True)

        foreground_mask = images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)

        # knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        # flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        # index_batch = index_batch.reshape(-1)
        if 'masked_point_cloud_indices' in input:
            rgb_points = rgb_points_bak
            transformed_points = transformed_points_bak
            normals_points = normals_points_bak

        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # training outputs
        output = {
            'img_res': self.img_res,
            'batch_size': batch_size,
            'predicted_mask': foreground_mask,  # mask loss
            'rgb_image': rgb_image,
            'canonical_points': pnts_c_flame,
            # for flame loss
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            # 'beta_shapedirs': beta_shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights
        }

        if not self.training:
            output_testing = {
                'normal_image': images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'shading_image': images[..., shading_begin_index:shading_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'albedo_image': images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'rendered_landmarks': rendered_landmarks.reshape(-1, 3),
                'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
                'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
                'deformed_verts': verts.reshape(-1, 3),
                'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
                'pnts_normal_deformed': normals_points.reshape(batch_size, n_points, 3),
                #'pnts_normal_canonical': canonical_normals,
            }
            if self.deformer_network.deform_c:
                output_testing['unconstrained_canonical_points'] = self.pc.points
            output.update(output_testing)
        
        if not self.training:
            self._output['mask_hole'] = self._output['mask_hole'].reshape(-1).unsqueeze(0) * (input['object_mask'])

        output.update(self._output)
        if self.optimize_scene_latent_code:
            output['scene_latent_code'] = input["scene_latent_code"]

        return output


    def get_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None, shapes=None, gt_beta_shapedirs=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)                                                 # NOTE pnts_c.shape: [400, 3]

        pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)      # NOTE deform_cc

        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])                            # NOTE batch_size: 1
        n_points = pnts_c.shape[0]                                                  # NOTE 400
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None, shapes=None, gt_beta_shapedirs=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
            pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
            # NOTE custom #########################
            condition = {}
            condition['scene_latent'] = scene_latent
            #######################################
            # shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)           
            shapedirs = shapedirs.expand(batch_size, -1, -1)
            posedirs = posedirs.expand(batch_size, -1, -1)
            lbs_weights = lbs_weights.expand(batch_size, -1)
            pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
            # beta_shapedirs = beta_shapedirs.expand(batch_size, -1, -1)                                                          # NOTE beta cancel
            beta_shapedirs = gt_beta_shapedirs.unsqueeze(0).expand(batch_size, -1, -1)
            pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, beta_shapedirs=beta_shapedirs, shapes=shapes) # FLAME-based deformed
            pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
            return pnts_d, pnts_d

        normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)                     # NOTE [400, 3] -> [400, 3]
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        shapes = shapes.reshape(batch_size, n_points, *shapes.shape[1:]).transpose(0, 1)                                             # NOTE beta cancel
        if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
            scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent, shapes, gt_beta_shapedirs)
            # pnts_c: [400, 3], betas: [400, 1, 50], transformations: [400, 1, 6, 4, 4], pose_feature: [400, 1, 36], scene_latent: [400, 1, 32]
        else:
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3), [400, 3] -> [400, 3]
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        grads_inv = grads_batch.inverse()
        normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)
        shading = self.rendering_network(normals_d, cond) # TODO 여기다가 condition을 추가하면 어떻게 될까?????
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
        return pnts_d, rgb_vals, albedo, shading, normals_d
    


class SceneLatentThreeStagesModelSingleGPUBatch(nn.Module):
    # NOTE singleGPU를 기반으로 만들었음. SingleGPU와의 차이점을 NOTE로 기술함.
    # deform_cc를 추가하였음. 반드시 켜져있어야함. 
    # Batch를 추가함.
    def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background, checkpoint_path, pcd_init=None):
        super().__init__()
        self.optimize_latent_code = conf.get_bool('train.optimize_latent_code')
        self.optimize_scene_latent_code = conf.get_bool('train.optimize_scene_latent_code')

        # FLAME_lightning
        self.FLAMEServer = utils.get_class(conf.get_string('model.FLAME_class'))(conf=conf,
                                                                                flame_model_path='./flame/FLAME2020/generic_model.pkl', 
                                                                                lmk_embedding_path='./flame/FLAME2020/landmark_embedding.npy',
                                                                                n_shape=100,
                                                                                n_exp=50,
                                                                                shape_params=shape_params,
                                                                                canonical_expression=canonical_expression,
                                                                                canonical_pose=canonical_pose).cuda()                           # NOTE cuda 없앴음
        
        # NOTE original code
        # self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
        #     self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        # self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)

        self.prune_thresh = conf.get_float('model.prune_thresh', default=0.5)

        # NOTE custom #########################
        # scene latent를 위해 변형한 모델들이 들어감.
        self.latent_code_dim = conf.get_int('model.latent_code_dim')
        print('[DEBUG] latent_code_dim:', self.latent_code_dim)
        # GeometryNetworkSceneLatent
        self.geometry_network = utils.get_class(conf.get_string('model.geometry_class'))(optimize_latent_code=self.optimize_latent_code,
                                                                                         optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                         latent_code_dim=self.latent_code_dim,
                                                                                         **conf.get_config('model.geometry_network'))
        # ForwardDeformerSceneLatentThreeStages
        self.deformer_network = utils.get_class(conf.get_string('model.deformer_class'))(FLAMEServer=self.FLAMEServer,
                                                                                        optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                        latent_code_dim=self.latent_code_dim,
                                                                                        **conf.get_config('model.deformer_network'))
        # RenderingNetworkSceneLatentThreeStages
        self.rendering_network = utils.get_class(conf.get_string('model.rendering_class'))(optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                                                            latent_code_dim=self.latent_code_dim,
                                                                                            **conf.get_config('model.rendering_network'))
        #######################################

        self.ghostbone = self.deformer_network.ghostbone

        # NOTE custom #########################
        if checkpoint_path is not None:
            print('[DEBUG] init point cloud from previous checkpoint')
            # n_init_point를 checkpoint으로부터 불러오기 위해..
            data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            n_init_points = data['state_dict']['model.pc.points'].shape[0]
            init_radius = data['state_dict']['model.radius'].item()
        elif pcd_init is not None:
            print('[DEBUG] init point cloud from meta learning')
            n_init_points = pcd_init['n_init_points']
            init_radius = pcd_init['init_radius']
        else:
            print('[DEBUG] init point cloud from scratch')
            n_init_points = 400
            init_radius = 0.5

        # PointCloudSceneLatentThreeStages
        self.pc = utils.get_class(conf.get_string('model.pointcloud_class'))(n_init_points=n_init_points,
                                                                            init_radius=init_radius,
                                                                            **conf.get_config('model.point_cloud')).cuda()     # NOTE .cuda() 없앴음
        #######################################

        n_points = self.pc.points.shape[0]
        self.img_res = img_res
        self.use_background = use_background
        if self.use_background:
            init_background = torch.zeros(img_res[0] * img_res[1], 3).float().cuda()                   # NOTE .cuda() 없앰
            # self.background = nn.Parameter(init_background)                                   # NOTE singleGPU코드에서는 이렇게 작성했지만,
            self.register_parameter('background', nn.Parameter(init_background))                # NOTE 이렇게 수정해서 혹시나하는 버그를 방지해보고자 한다.
        else:
            # self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()           # NOTE singleGPU코드에서는 이렇게 작성했지만,
            self.register_buffer('background', torch.ones(img_res[0] * img_res[1], 3).float())  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # NOTE original code
        self.raster_settings = PointsRasterizationSettings(image_size=img_res[0],
                                                           radius=self.pc.radius_factor * (0.75 ** math.log2(n_points / 100)),
                                                           points_per_pixel=10)
        self.register_buffer('radius', torch.tensor(self.raster_settings.radius))
        
        # keypoint rasterizer is only for debugging camera intrinsics
        self.raster_settings_kp = PointsRasterizationSettings(image_size=self.img_res[0],
                                                              radius=0.007,
                                                              points_per_pixel=1)

        # NOTE ablation #########################################
        self.enable_prune = conf.get_bool('train.enable_prune')

        # self.visible_points = torch.zeros(n_points).bool().cuda()                             # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        if self.enable_prune:
            if checkpoint_path is not None:
                # n_init_point를 checkpoint으로부터 불러오기 위해..
                data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                visible_points = data['state_dict']['model.visible_points']                         # NOTE 이거 안해주면 visible이 0이 되어서 훈련이 안됨.
            else:
                visible_points = torch.zeros(n_points).bool()
            self.register_buffer('visible_points', visible_points)                                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.
        # self.compositor = AlphaCompositor().cuda()                                            # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        self.compositor = AlphaCompositor().cuda()                                                     # NOTE cuda 할당이 자동으로 되도록 수정해본다.


    # def _compute_canonical_normals_and_feature_vectors(self, condition):
    #     p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
    #     p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
    #     # randomly sample some points in the neighborhood within 0.25 distance

    #     # eikonal_points = torch.cat([p, p + (torch.rand(p.shape).cuda() - 0.5) * 0.5], dim=0)                          # NOTE original code, eikonal_points.shape: [6400, 3]
    #     eikonal_points = torch.cat([p, p + (torch.rand(p.shape, device=p.device) - 0.5) * 0.5], dim=0)                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

    #     if self.optimize_scene_latent_code:
    #         condition['scene_latent_gradient'] = torch.cat([condition['scene_latent'], condition['scene_latent']], dim=0).detach()

    #     eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points.detach(), condition)
    #     n_points = self.pc.points.shape[0] # 400
    #     canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1) # 400, 3

    #     p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
    #     p = self.deformer_network.canocanonical_deform(p, cond=condition)       # NOTE deform_cc
    #     geometry_output = self.geometry_network(p, condition)  # not using SDF to regularize point location, 3200, 4
    #     sdf_values = geometry_output[:, 0]

    #     feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector

    #     if self.training and hasattr(self, "_output"):
    #         self._output['sdf_values'] = sdf_values
    #         self._output['grad_thetas'] = grad_thetas
    #     if not self.training:
    #         self._output['pnts_albedo'] = feature_vector

    #     return canonical_normals, feature_vector # (400, 3), (400, 3) -> (400, 3) (3200, 3)
    
    def _compute_canonical_normals_and_feature_vectors(self, points, condition, values):
        p = points.detach()                 

        def _func_canocanonical_deform(points, cond):                                                                                           # NOTE points: [400, 3], cond: [32]
            condition = {}
            condition['scene_latent'] = cond.unsqueeze(0).unsqueeze(1).expand(-1, values['n_points'], -1).reshape(values['n_points'], -1)       # NOTE [32] -> [400, 32]
            points = self.deformer_network.canocanonical_deform(points, cond=condition)                                                         # NOTE deform_cc. points: [400, 3] -> [400, 3]
            eikonal_points = torch.cat([points, points + (torch.rand(points.shape, device=points.device) - 0.5) * 0.5], dim=0)                  # NOTE [400, 3] -> [800, 3]
            if self.optimize_scene_latent_code:
                scene_latent_gradient = torch.cat([condition['scene_latent'], condition['scene_latent']], dim=0).detach()                       # NOTE [400, 32] -> [800, 32]
            return points, eikonal_points, scene_latent_gradient                                                                                # NOTE points: [400, 3], eikonal_points: [800, 3], scene_latent_gradient: [800, 32]

        # NOTE p: [8, 400, 3], condition['scene_latent']: [8, 32] -> p: [8, 400, 3], eikonal_points: [8, 800, 3], scene_latent_gradient: [8, 800, 32]
        p, eikonal_points, scene_latent_gradient = vmap(_func_canocanonical_deform, out_dims=(0, 0, 0), randomness='different')(p, condition['scene_latent'])    # NOTE batch마다 randomness가 달라야한다.  
        
        gradient_list = []
        for i in range(p.shape[0]):                                                                                                             # NOTE p.shape: [8, 400, 3]
            eikonal_points_input = eikonal_points[i, :, :].detach()                                                                             # NOTE eikonal_points_input: [800, 3]
            condition['scene_latent_gradient'] = scene_latent_gradient[i, :, :].detach()                                                        # NOTE scene_latent_gradient: [800, 32]
            _, grad_thetas = self.geometry_network.gradient(eikonal_points_input, condition)                                                    # NOTE grad_thetas: [800, 3]
            gradient_list.append(grad_thetas.unsqueeze(0))
        grad_thetas_list = torch.cat(gradient_list, dim=0)                                                                                      # NOTE grad_thetas_list: [8, 800, 3]
        
        def _func_normal(points, grad_thetas, cond): # NOTE points: [400, 3], grad_thetas: [800, 3], cond: [32]
            n_points = points.shape[0]                                                                                          
            canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1)                                                 # NOTE canonical_normals: [400, 3]                           
            condition = {}
            condition['scene_latent'] = cond.unsqueeze(0).unsqueeze(1).expand(-1, values['n_points'], -1).reshape(values['n_points'], -1)       # NOTE condition['scene_latent]: [400, 32]
            geometry_output = self.geometry_network(points, condition)                                                                          # NOTE geometry_output: [400, 4]
            sdf_values = geometry_output[:, 0]                                                                                                  # NOTE sdf_values: [400]           
            feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)                                                                         # NOTE albedo vector, feature_vector: [400, 3]
            return canonical_normals, sdf_values, feature_vector
        
        # NOTE input의 p는 O_1이 적용된 p이다. 즉 x_c이다. 
        # NOTE canonical_normals: [8, 400, 3], sdf_values: [8, 400], feature_vector: [8, 400, 3]
        canonical_normals, sdf_values, feature_vector = vmap(_func_normal, out_dims=(0, 0, 0), randomness='error')(p, grad_thetas_list, condition['scene_latent'])     

        if self.training and hasattr(self, "_output"):
            self._output['sdf_values'] = sdf_values                         # NOTE sdf_values: [8, 400]
            self._output['grad_thetas'] = grad_thetas_list # grad_thetas    # NOTE grad_thetas: [8, 800, 3]
        if not self.training:
            self._output['pnts_albedo'] = feature_vector
        # NOTE canonical_normals: [8, 400, 3], feature_vector: [8, 400, 3], p: [8, 400, 3]
        return canonical_normals, feature_vector, p
    

    def _render(self, point_cloud, cameras, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)
        r = rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        images, weights = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            point_cloud.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
        # batch_size, img_res, img_res, points_per_pixel
        if self.enable_prune and self.training and not render_kp:
            n_points = self.pc.points.shape[0]
            # the first point for each pixel is visible
            visible_points = fragments.idx.long()[..., 0].reshape(-1)
            visible_points = visible_points[visible_points != -1]

            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

            # points with weights larger than prune_thresh are visible
            visible_points = fragments.idx.long().reshape(-1)[weights.reshape(-1) > self.prune_thresh]
            visible_points = visible_points[visible_points != -1]

            n_points = self.pc.points.shape[0]
            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

        return images

    def forward(self, input):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        batch_size = flame_pose.shape[0]
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)

        if self.ghostbone:
            # identity transformation for body
            # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
            transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
        cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        # NOTE custom #########################
        if self.optimize_latent_code or self.optimize_scene_latent_code:
            network_condition = dict()
        else:
            network_condition = None

        if self.optimize_latent_code:
            network_condition['latent'] = input["latent_code"] # [1, 32]
        
        if self.optimize_scene_latent_code:
            network_condition['scene_latent'] = input["scene_latent_code"]          # NOTE [8, 32]
        ######################################
        points_batch = self.pc.points.unsqueeze(0).expand(batch_size, -1, -1)       # NOTE [400, 3] -> [8, 400, 3] 이렇게 분화해놓고 제때제때 넣어주기만 하면 된다. latent는 순서대로 적용될거고.

        canonical_normals, feature_vector, points_c_batch = self._compute_canonical_normals_and_feature_vectors(points_batch, network_condition, values={'n_points': n_points, 'total_points': total_points})      # NOTE network_condition 추가

        transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch(pnts_c=points_c_batch, # pnts_c=self.pc.points,    
                                                                                                                     normals=canonical_normals,
                                                                                                                     feature_vectors=feature_vector,
                                                                                                                     pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                                     cond=network_condition) # NOTE network_condition 추가

        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        p = self.deformer_network.canocanonical_deform(p, cond=network_condition)       # NOTE deform_cc
        # shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach(), cond=network_condition) # NOTE network_condition 추가
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(p, cond=network_condition) # NOTE network_condition 추가
        # NOTE transformed_points: x_d
        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        rgb_points = rgb_points.reshape(batch_size, n_points, 3)
        # point feature to rasterize and composite
        features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)
        if not self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)
            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

            shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1] + 3
            albedo_points = torch.clamp(albedo_points, 0., 1.)
            features = torch.cat([features, shading_points.reshape(batch_size, n_points, 3), albedo_points.reshape(batch_size, n_points, 3)], dim=-1)

        transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

        images = self._render(transformed_point_cloud, cameras)

        if not self.training:
            # render landmarks for easier camera format debugging
            landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
            transformed_verts = Pointclouds(points=landmarks2d, features=torch.ones_like(landmarks2d))
            rendered_landmarks = self._render(transformed_verts, cameras, render_kp=True)

        foreground_mask = images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)

        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)

        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # training outputs
        output = {
            'img_res': self.img_res,
            'batch_size': batch_size,
            'predicted_mask': foreground_mask,  # mask loss
            'rgb_image': rgb_image,
            'canonical_points': pnts_c_flame,
            # for flame loss
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights,
        }

        if not self.training:
            output_testing = {
                'normal_image': images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'shading_image': images[..., shading_begin_index:shading_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'albedo_image': images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'rendered_landmarks': rendered_landmarks.reshape(-1, 3),
                'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
                'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
                'deformed_verts': verts.reshape(-1, 3),
                'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
                'pnts_normal_deformed': normals_points.reshape(batch_size, n_points, 3),
                #'pnts_normal_canonical': canonical_normals,
            }
            if self.deformer_network.deform_c:
                output_testing['unconstrained_canonical_points'] = self.pc.points
            output.update(output_testing)
        output.update(self._output)
        if self.optimize_scene_latent_code:
            output['scene_latent_code'] = input["scene_latent_code"]

        return output


    def get_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None):
        # NOTE pnts_c: [8, 400, 3], normals: [8, 400, 3], feature_vectors: [8, 400, 3]
        # NOTE pose_feature: [3200, 36], betas: [3200, 50], transformations: [3200, 6, 4, 4], cond['scene_latent]: [8, 32]
        # if pnts_c.shape[0] == 0:       # NOTE original code
        if pnts_c.shape[1] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True) # NOTE latent optimization이 꺼져있을 때, pnts_c.shape: [400, 3] batch가 적용이 안되어있다.

        # pnts_c = self.deformer_network.canocanonical_deform(pnts_c, cond=cond)       # NOTE deform_cc
        total_points = betas.shape[0]
        # NOTE original code
        # batch_size = int(total_points / pnts_c.shape[0])
        # n_points = pnts_c.shape[0]
        # NOTE custom code
        batch_size = int(total_points / pnts_c.shape[1])
        n_points = pnts_c.shape[1]
        
        def _func_deformer_flame(pnts_c, scene_latent=None):        # NOTE pnts_c: [400, 3], scene_latent: [32]
            # NOTE custom #########################
            condition = {}
            condition['scene_latent'] = scene_latent
            #######################################
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)   
            # NOTE (output) shapedirs: [400, 3, 50], posedirs: [400, 36, 3], lbs_weights: [400, 6], pnts_c_flame: [400, 3]
            return shapedirs, posedirs, lbs_weights, pnts_c_flame

        # NOTE (input) pnts_c: [8, 400, 3], cond['scene_latent']: [8, 400, 32]
        scene_latent = cond['scene_latent'].unsqueeze(1).expand(-1, n_points, -1)
        shapedirs, posedirs, lbs_weights, pnts_c_flame = vmap(_func_deformer_flame, out_dims=(0, 0, 0, 0))(pnts_c, scene_latent)  
        # NOTE (output) shapedirs: [8, 400, 3, 50], posedirs: [8, 400, 36, 3], lbs_weights: [8, 400, 6], pnts_c_flame: [8, 400, 3]

        def _func(pnts_c, pnts_c_flame, shapedirs, posedirs, lbs_weights, betas, transformations, pose_feature): # NOTE pnts_c: [3], pnts_c_flame: [8, 3], shapedirs: [8, 3, 50], posedirs: [8, 36, 3], lbs_weights: [8, 6]
            betas = betas.unsqueeze(0)                                  
            transformations = transformations.unsqueeze(0)              # NOTE [1, 6, 4, 4]
            pose_feature = pose_feature.unsqueeze(0)                    # NOTE [1, 36]
            shapedirs = shapedirs.unsqueeze(0)                          # NOTE [1, 3, 50]
            posedirs = posedirs.unsqueeze(0)                            # NOTE [1, 36, 3]
            lbs_weights = lbs_weights.unsqueeze(0)                      # NOTE [1, 6]
            pnts_c_flame = pnts_c_flame.unsqueeze(0)                    # NOTE [1, 3]
            pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights) # FLAME-based deformed. pnts_d: [1, 3]
            pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
            return pnts_d, pnts_d
        
        # normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        normals = rearrange(normals, 'b n d -> (n b) d')                                                                                                    # NOTE [8, 400, 3] -> [400*8, 3]
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)                                                                       # NOTE [3200, 50] -> [8, 400, 50] -> [400, 8, 50]
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)                                         # NOTE [3200, 6, 4, 4] -> [400, 8, 6, 4, 4]
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)                                                  # NOTE [3200, 36] -> [400, 8, 36]
        # if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
        #     # scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
        #     grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent)
        # else:
        #     grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transaformations, pose_feature)

        pnts_d_list, grads_batch_list = [], []
        # NOTE pnts_c로 미분해야한다. 이건 x_c로 x_cc나 x_c_flame으로 하면 안된다. dim은 [400, 3]이어야 한다.
        for i in range(pnts_c.shape[0]):
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(
                pnts_c[i], pnts_c_flame[i], shapedirs[i], posedirs[i], lbs_weights[i], betas[i], transformations[i], pose_feature[i])
            pnts_d_list.append(pnts_d.unsqueeze(1))
            grads_batch_list.append(grads_batch.unsqueeze(1))
        pnts_d = torch.cat(pnts_d_list, dim=1)                      # NOTE [400, 8, 3]
        grads_batch = torch.cat(grads_batch_list, dim=1)            # NOTE [400, 8, 3, 3]

        pnts_d = rearrange(pnts_d, 'b n d -> n (b d)')               # NOTE [8, 400, 3] -> [400, 3*8]
        grads_batch = rearrange(grads_batch, 'b n d e -> n (d b) e') # NOTE [8, 400, 3, 3] -> [400, 3*8, 3]

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)                           # NOTE (400, 24) -> (3200, 3)
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)           

        grads_inv = grads_batch.inverse()
        normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)
        shading = self.rendering_network(normals_d, cond) # TODO 여기다가 condition을 추가하면 어떻게 될까?????
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
        return pnts_d, rgb_vals, albedo, shading, normals_d
    


class SceneLatentModel(nn.Module):
    # NOTE singleGPU를 기반으로 만들었음. SingleGPU와의 차이점을 NOTE로 기술함.
    def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background, checkpoint_path):
        super().__init__()
        self.optimize_latent_code = conf.get_bool('train.optimize_latent_code')
        self.optimize_scene_latent_code = conf.get_bool('train.optimize_scene_latent_code')

        self.FLAMEServer = FLAME_lightning(conf=conf,
                                           flame_model_path='./flame/FLAME2020/generic_model.pkl', 
                                           lmk_embedding_path='./flame/FLAME2020/landmark_embedding.npy',
                                           n_shape=100,
                                           n_exp=50,
                                           shape_params=shape_params,
                                           canonical_expression=canonical_expression,
                                           canonical_pose=canonical_pose)                           # NOTE cuda 없앴음
        
        # NOTE original code
        # self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
        #     self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        # self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)

        self.prune_thresh = conf.get_float('model.prune_thresh', default=0.5)

        # NOTE custom #########################
        # scene latent를 위해 변형한 모델들이 들어감.
        self.geometry_network = GeometryNetworkSceneLatent(optimize_latent_code=self.optimize_latent_code,
                                                           optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                           **conf.get_config('model.geometry_network'))
        self.deformer_network = ForwardDeformerSceneLatent(FLAMEServer=self.FLAMEServer,
                                                           optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                           **conf.get_config('model.deformer_network'))
        self.rendering_network = RenderingNetwork(**conf.get_config('model.rendering_network'))
        #######################################

        self.ghostbone = self.deformer_network.ghostbone

        # NOTE custom #########################
        self.pc = PointCloudSceneLatent(optimize_scene_latent_code=self.optimize_scene_latent_code, 
                                        checkpoint_path=checkpoint_path,
                                        **conf.get_config('model.point_cloud'))     # NOTE .cuda() 없앴음
        #######################################

        n_points = self.pc.points.shape[0]
        self.img_res = img_res
        self.use_background = use_background
        if self.use_background:
            init_background = torch.zeros(img_res[0] * img_res[1], 3).float()                   # NOTE .cuda() 없앰
            # self.background = nn.Parameter(init_background)                                   # NOTE singleGPU코드에서는 이렇게 작성했지만,
            self.register_parameter('background', nn.Parameter(init_background))                # NOTE 이렇게 수정해서 혹시나하는 버그를 방지해보고자 한다.
        else:
            # self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()           # NOTE singleGPU코드에서는 이렇게 작성했지만,
            self.register_buffer('background', torch.ones(img_res[0] * img_res[1], 3).float())  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # NOTE custom #########################
        # 오직 저장하고 불러오기위한 용도.
        if checkpoint_path is not None:
            data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            radius = data['state_dict']['model.radius'].item()
        else:
            radius = self.pc.radius_factor * (0.75 ** math.log2(n_points / 100))
        self.register_buffer('radius', torch.tensor(radius))
        #######################################

        # NOTE original code
        # self.raster_settings = PointsRasterizationSettings(image_size=img_res[0],
        #                                                    radius=self.pc.radius_factor * (0.75 ** math.log2(n_points / 100)),
        #                                                    points_per_pixel=10)
        # NOTE for save and load radius
        self.raster_settings = PointsRasterizationSettings(image_size=img_res[0],
                                                           radius=radius,
                                                           points_per_pixel=10)
        
        # keypoint rasterizer is only for debugging camera intrinsics
        self.raster_settings_kp = PointsRasterizationSettings(image_size=self.img_res[0],
                                                              radius=0.007,
                                                              points_per_pixel=1)

        # NOTE ablation #########################################
        self.enable_prune = conf.get_bool('train.enable_prune')

        # self.visible_points = torch.zeros(n_points).bool().cuda()                             # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        if self.enable_prune:
            if checkpoint_path is not None:
                # n_init_point를 checkpoint으로부터 불러오기 위해..
                data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                visible_points = data['state_dict']['model.visible_points']                         # NOTE 이거 안해주면 visible이 0이 되어서 훈련이 안됨.
            else:
                visible_points = torch.zeros(n_points).bool()
            self.register_buffer('visible_points', visible_points)                                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.
        # self.compositor = AlphaCompositor().cuda()                                            # NOTE singleGPU 코드에서는 이렇게 작성했지만,
        self.compositor = AlphaCompositor()                                                     # NOTE cuda 할당이 자동으로 되도록 수정해본다.


    def _compute_canonical_normals_and_feature_vectors(self, condition):
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        # randomly sample some points in the neighborhood within 0.25 distance

        # eikonal_points = torch.cat([p, p + (torch.rand(p.shape).cuda() - 0.5) * 0.5], dim=0)                          # NOTE original code, eikonal_points.shape: [6400, 3]
        eikonal_points = torch.cat([p, p + (torch.rand(p.shape, device=p.device) - 0.5) * 0.5], dim=0)                  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        if self.optimize_scene_latent_code:
            condition['scene_latent_gradient'] = torch.cat([condition['scene_latent'], condition['scene_latent']], dim=0)

        eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points.detach(), condition)
        # eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points, condition)
        n_points = self.pc.points.shape[0] # 400
        canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1) # 400, 3
        geometry_output = self.geometry_network(self.pc.points.detach(), condition)  # not using SDF to regularize point location, 3200, 4
        sdf_values = geometry_output[:, 0]

        feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector

        if self.training and hasattr(self, "_output"):
            self._output['sdf_values'] = sdf_values
            self._output['grad_thetas'] = grad_thetas
        if not self.training:
            self._output['pnts_albedo'] = feature_vector

        return canonical_normals, feature_vector # (400, 3), (400, 3) -> (400, 3) (3200, 3)

    def _render(self, point_cloud, cameras, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)
        r = rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        images, weights = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            point_cloud.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
        # batch_size, img_res, img_res, points_per_pixel
        if self.enable_prune and self.training and not render_kp:
            n_points = self.pc.points.shape[0]
            # the first point for each pixel is visible
            visible_points = fragments.idx.long()[..., 0].reshape(-1)
            visible_points = visible_points[visible_points != -1]

            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

            # points with weights larger than prune_thresh are visible
            visible_points = fragments.idx.long().reshape(-1)[weights.reshape(-1) > self.prune_thresh]
            visible_points = visible_points[visible_points != -1]

            n_points = self.pc.points.shape[0]
            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

        return images

    def forward(self, input):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        batch_size = flame_pose.shape[0]
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)

        if self.ghostbone:
            # identity transformation for body
            # transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)                               # NOTE original code
            transformations = torch.cat([torch.eye(4, device=transformations.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float(), transformations], 1)         # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)          # NOTE singleGPU에서의 코드
        cameras = PerspectiveCameras(device=expression.device, R=R, T=T, K=intrinsics)  # NOTE cuda 할당이 자동으로 되도록 수정해본다.

        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        # NOTE custom #########################
        if self.optimize_latent_code or self.optimize_scene_latent_code:
            network_condition = dict()
        else:
            network_condition = None

        if self.optimize_latent_code:
            network_condition['latent'] = input["latent_code"] # [1, 32]
        if self.optimize_scene_latent_code:
            network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)  
            # NOTE scene마다 point cloud를 다르게 하기 위해서 쓰임.
            delta_x = self.pc.deform_points(cond=network_condition)           
        ######################################

        canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(network_condition)      # NOTE network_condition 추가

        transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                                                     normals=canonical_normals,
                                                                                                                     feature_vectors=feature_vector,
                                                                                                                     pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                                     cond=network_condition) # NOTE network_condition 추가

        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach(), cond=network_condition) # NOTE network_condition 추가
        # NOTE transformed_points: x_d
        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        rgb_points = rgb_points.reshape(batch_size, n_points, 3)
        # point feature to rasterize and composite
        features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)
        if not self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)
            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

            shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1] + 3
            albedo_points = torch.clamp(albedo_points, 0., 1.)
            features = torch.cat([features, shading_points.reshape(batch_size, n_points, 3), albedo_points.reshape(batch_size, n_points, 3)], dim=-1)

        transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

        images = self._render(transformed_point_cloud, cameras)

        if not self.training:
            # render landmarks for easier camera format debugging
            landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
            transformed_verts = Pointclouds(points=landmarks2d, features=torch.ones_like(landmarks2d))
            rendered_landmarks = self._render(transformed_verts, cameras, render_kp=True)

        foreground_mask = images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)

        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)

        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # training outputs
        output = {
            'img_res': self.img_res,
            'batch_size': batch_size,
            'predicted_mask': foreground_mask,  # mask loss
            'rgb_image': rgb_image,
            'canonical_points': pnts_c_flame,
            # for flame loss
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights,
        }

        if not self.training:
            output_testing = {
                'normal_image': images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'shading_image': images[..., shading_begin_index:shading_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'albedo_image': images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'rendered_landmarks': rendered_landmarks.reshape(-1, 3),
                'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
                'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
                'deformed_verts': verts.reshape(-1, 3),
                'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
                'pnts_normal_deformed': normals_points.reshape(batch_size, n_points, 3),
                #'pnts_normal_canonical': canonical_normals,
            }
            if self.deformer_network.deform_c:
                output_testing['unconstrained_canonical_points'] = self.pc.points
            output.update(output_testing)
        output.update(self._output)
        if self.optimize_scene_latent_code:
            output['delta_x'] = delta_x                     # NOTE for regularization
        if self.optimize_scene_latent_code:
            output['scene_latent_code'] = input["scene_latent_code"]

        return output


    def get_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True) # NOTE latent optimization이 꺼져있을 때, pnts_c.shape: [400, 3] batch가 적용이 안되어있다.
        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])
        n_points = pnts_c.shape[0]
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
            pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
            # NOTE custom #########################
            condition = {}
            condition['scene_latent'] = scene_latent
            #######################################
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
            shapedirs = shapedirs.expand(batch_size, -1, -1)
            posedirs = posedirs.expand(batch_size, -1, -1)
            lbs_weights = lbs_weights.expand(batch_size, -1)
            pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
            pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights) # FLAME-based deformed
            pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
            return pnts_d, pnts_d

        normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
            scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent)
        else:
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3)
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        grads_inv = grads_batch.inverse()
        normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)
        shading = self.rendering_network(normals_d) # TODO 여기다가 condition을 추가하면 어떻게 될까?????
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
        return pnts_d, rgb_vals, albedo, shading, normals_d






class SceneLatentModelSingleGPU(nn.Module):
    def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background, checkpoint_path):
        super().__init__()
        self.optimize_latent_code = conf.get_bool('train.optimize_latent_code')
        self.optimize_scene_latent_code = conf.get_bool('train.optimize_scene_latent_code')

        self.FLAMEServer = FLAME('./flame/FLAME2020/generic_model.pkl', 
                                 './flame/FLAME2020/landmark_embedding.npy',
                                 n_shape=100,
                                 n_exp=50,
                                 shape_params=shape_params,
                                 canonical_expression=canonical_expression,
                                 canonical_pose=canonical_pose).cuda()
        
        self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
            self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)
        self.prune_thresh = conf.get_float('model.prune_thresh', default=0.5)

        # NOTE custom #########################
        # scene latent를 위해 변형한 모델들이 들어감.
        self.geometry_network = GeometryNetworkSceneLatent(optimize_latent_code=self.optimize_latent_code,
                                                           optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                           **conf.get_config('model.geometry_network'))
        self.deformer_network = ForwardDeformerSceneLatent(FLAMEServer=self.FLAMEServer,
                                                           optimize_scene_latent_code=self.optimize_scene_latent_code,
                                                           **conf.get_config('model.deformer_network'))
        self.rendering_network = RenderingNetwork(**conf.get_config('model.rendering_network'))
        #######################################

        self.ghostbone = self.deformer_network.ghostbone
        if self.ghostbone:
            self.FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().cuda(), self.FLAMEServer.canonical_transformations], 1)
        
        # NOTE custom #########################
        self.pc = PointCloudSceneLatent(optimize_scene_latent_code=self.optimize_scene_latent_code, 
                                        checkpoint_path=checkpoint_path,
                                        **conf.get_config('model.point_cloud')).cuda()
        #######################################

        n_points = self.pc.points.shape[0]
        self.img_res = img_res
        self.use_background = use_background
        if self.use_background:
            init_background = torch.zeros(img_res[0] * img_res[1], 3).float().cuda()
            self.background = nn.Parameter(init_background)                 # NOTE 따로 self.register에 넣을 이유는 없음.
        else:
            self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()

        self.raster_settings = PointsRasterizationSettings(image_size=img_res[0],
                                                           radius=self.pc.radius_factor * (0.75 ** math.log2(n_points / 100)),
                                                           points_per_pixel=10)
        # keypoint rasterizer is only for debugging camera intrinsics
        self.raster_settings_kp = PointsRasterizationSettings(image_size=self.img_res[0],
                                                              radius=0.007,
                                                              points_per_pixel=1)

        self.visible_points = torch.zeros(n_points).bool().cuda()   # NOTE self.register에 추가해도 좋을듯?
        self.compositor = AlphaCompositor().cuda()                  # NOTE 얘도 nn.Module임.


    def _compute_canonical_normals_and_feature_vectors(self, condition):
        p = self.pc.points.detach()     # NOTE p.shape: [3200, 3]
        # randomly sample some points in the neighborhood within 0.25 distance

        eikonal_points = torch.cat([p, p + (torch.rand(p.shape).cuda() - 0.5) * 0.5], dim=0)        # eikonal_points.shape: [6400, 3]

        if self.optimize_scene_latent_code:
            condition['scene_latent_gradient'] = torch.cat([condition['scene_latent'], condition['scene_latent']], dim=0)
        eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points.detach(), condition)
        # eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points, condition)
        n_points = self.pc.points.shape[0] # 400
        canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1) # 400, 3
        geometry_output = self.geometry_network(self.pc.points.detach(), condition)  # not using SDF to regularize point location, 3200, 4
        sdf_values = geometry_output[:, 0]

        feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector

        if self.training and hasattr(self, "_output"):
            self._output['sdf_values'] = sdf_values
            self._output['grad_thetas'] = grad_thetas
        if not self.training:
            self._output['pnts_albedo'] = feature_vector

        return canonical_normals, feature_vector # (400, 3), (400, 3) -> (400, 3) (3200, 3)

    def _render(self, point_cloud, cameras, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)
        r = rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        images, weights = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            point_cloud.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
        # batch_size, img_res, img_res, points_per_pixel
        if self.training and not render_kp:
            n_points = self.pc.points.shape[0]
            # the first point for each pixel is visible
            visible_points = fragments.idx.long()[..., 0].reshape(-1)
            visible_points = visible_points[visible_points != -1]

            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

            # points with weights larger than prune_thresh are visible
            visible_points = fragments.idx.long().reshape(-1)[weights.reshape(-1) > self.prune_thresh]
            visible_points = visible_points[visible_points != -1]

            n_points = self.pc.points.shape[0]
            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

        return images

    def forward(self, input):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        batch_size = flame_pose.shape[0]
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)

        if self.ghostbone:
            # identity transformation for body
            transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)       # NOTE original code

        cameras = PerspectiveCameras(device='cuda' , R=R, T=T, K=intrinsics)         # NOTE nn.Module의 상속의 상속의 상속임. 여하튼 self로 넘겨주는게 좋을 듯 하다.
        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points
        # transformations: 8, 6, 4, 4 (batch_size, n_joints, 4, 4) SE3

        # NOTE custom #########################
        if self.optimize_latent_code or self.optimize_scene_latent_code:
            network_condition = dict()
        else:
            network_condition = None

        if self.optimize_latent_code:
            network_condition['latent'] = input["latent_code"] # [1, 32]
        if self.optimize_scene_latent_code:
            network_condition['scene_latent'] = input["scene_latent_code"].unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)  
            # NOTE scene마다 point cloud를 다르게 하기 위해서 쓰임.
            delta_x = self.pc.deform_points(cond=network_condition)           
        ######################################

        canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors(network_condition)      # NOTE network_condition 추가

        transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch(pnts_c=self.pc.points,     # NOTE [400, 3]
                                                                                                                     normals=canonical_normals,
                                                                                                                     feature_vectors=feature_vector,
                                                                                                                     pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                                                                     transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                                                                     cond=network_condition) # NOTE network_condition 추가

        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach(), cond=network_condition) # NOTE network_condition 추가
        # NOTE transformed_points: x_d
        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        rgb_points = rgb_points.reshape(batch_size, n_points, 3)
        # point feature to rasterize and composite
        features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)
        if not self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)
            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

            shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1] + 3
            albedo_points = torch.clamp(albedo_points, 0., 1.)
            features = torch.cat([features, shading_points.reshape(batch_size, n_points, 3), albedo_points.reshape(batch_size, n_points, 3)], dim=-1)

        transformed_point_cloud = Pointclouds(points=transformed_points, features=features)     # NOTE pytorch3d's pointcloud class.

        images = self._render(transformed_point_cloud, cameras)

        if not self.training:
            # render landmarks for easier camera format debugging
            landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
            transformed_verts = Pointclouds(points=landmarks2d, features=torch.ones_like(landmarks2d))
            rendered_landmarks = self._render(transformed_verts, cameras, render_kp=True)

        foreground_mask = images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)

        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)

        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # training outputs
        output = {
            'img_res': self.img_res,
            'batch_size': batch_size,
            'predicted_mask': foreground_mask,  # mask loss
            'rgb_image': rgb_image,
            'canonical_points': pnts_c_flame,
            # for flame loss
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights,
        }

        if not self.training:
            output_testing = {
                'normal_image': images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'shading_image': images[..., shading_begin_index:shading_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'albedo_image': images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'rendered_landmarks': rendered_landmarks.reshape(-1, 3),
                'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
                'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
                'deformed_verts': verts.reshape(-1, 3),
                'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
                'pnts_normal_deformed': normals_points.reshape(batch_size, n_points, 3),
                #'pnts_normal_canonical': canonical_normals,
            }
            if self.deformer_network.deform_c:
                output_testing['unconstrained_canonical_points'] = self.pc.points
            output.update(output_testing)
        output.update(self._output)
        if self.optimize_scene_latent_code:
            output['delta_x'] = delta_x                     # NOTE for regularization

        return output


    def get_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations, cond=None):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True) # NOTE latent optimization이 꺼져있을 때, pnts_c.shape: [400, 3] batch가 적용이 안되어있다.
        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])
        n_points = pnts_c.shape[0]
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature, scene_latent=None):            # NOTE pnts_c: [3], betas: [8, 50], transformations: [8, 6, 4, 4], pose_feature: [8, 36] if batch_size is 8
            pnts_c = pnts_c.unsqueeze(0)            # [3] -> ([1, 3])
            # NOTE custom #########################
            condition = {}
            condition['scene_latent'] = scene_latent
            #######################################
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c, cond=condition)            # NOTE batch_size 1만 가능.
            shapedirs = shapedirs.expand(batch_size, -1, -1)
            posedirs = posedirs.expand(batch_size, -1, -1)
            lbs_weights = lbs_weights.expand(batch_size, -1)
            pnts_c_flame = pnts_c_flame.expand(batch_size, -1)      # NOTE [1, 3] -> [8, 3]
            pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights) # FLAME-based deformed
            pnts_d = pnts_d.reshape(-1)         # NOTE pnts_c는 (3) -> (1, 3)이고 pnts_d는 (8, 3) -> (24)
            return pnts_d, pnts_d

        normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)   # NOTE 여기서 (3200, 50) -> (400, 8, 50)으로 바뀐다.
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        if self.optimize_scene_latent_code:     # NOTE pnts_c: [400, 3]
            scene_latent = cond['scene_latent'].reshape(batch_size, n_points, *cond['scene_latent'].shape[1:]).transpose(0, 1)
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature, scene_latent)
        else:
            grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)       # NOTE (400, 24) -> (3200, 3)
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        grads_inv = grads_batch.inverse()
        normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)
        shading = self.rendering_network(normals_d) # TODO 여기다가 condition을 추가하면 어떻게 될까?????
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.) # TODO ERROR!!! albedo는 8배 늘었으나 shading은 그대로여서 에러가 발생함. 
        return pnts_d, rgb_vals, albedo, shading, normals_d
