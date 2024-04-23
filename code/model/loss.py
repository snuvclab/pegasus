import torch
from torch import nn
from model.vgg_feature import VGGPerceptualLoss


class LossPEGASUS(nn.Module):
    # NOTE Loss_lightning에서 cuda를 추가한 버전이다. 수정은 매번 똑같이 할 것!!
    def __init__(self, mask_weight, var_expression=None, lbs_weight=0,sdf_consistency_weight=0, eikonal_weight=0, vgg_feature_weight=0, optimize_scene_latent_code=False, deform_pcd_weight=0, latent_reg_weight=0, normal_weight=0, segment_weight=0, chamfer_weight=0): # 
        super().__init__()
        
        self.mask_weight = mask_weight
        self.lbs_weight = lbs_weight
        self.sdf_consistency_weight = sdf_consistency_weight
        self.eikonal_weight = eikonal_weight
        self.vgg_feature_weight = vgg_feature_weight
        self.var_expression = var_expression                                                                      # NOTE original code
        # NOTE custom ###############
        # if var_expression is not None:
        #     # self.var_expression = self.var_expression.unsqueeze(1).expand(1, 3, -1).reshape(1, -1).cuda()         # NOTE original code
        #     self.register_buffer('var_expression', var_expression.unsqueeze(1).expand(1, 3, -1).reshape(1, -1))
        # else:
        #     self.var_expression = var_expression
        if self.var_expression is not None:
            self.var_expression = self.var_expression.unsqueeze(1).expand(1, 3, -1).reshape(1, -1).cuda()  
        #############################
        print("Expression variance: ", self.var_expression)

        if self.vgg_feature_weight > 0:
            self.get_vgg_loss = VGGPerceptualLoss().cuda()              # NOTE original code
            # self.get_vgg_loss = VGGPerceptualLoss()                     # NOTE because of lightning

        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='none')
        # self.binary_cross_entropy_loss = nn.BCELoss(reduction='mean')
        self.binary_cross_entropy_loss = nn.BCEWithLogitsLoss(reduction='mean')

        self.optimize_scene_latent_code = optimize_scene_latent_code
        self.deform_pcd_weight = deform_pcd_weight
        self.latent_reg_weight = latent_reg_weight
        self.normal_weight = normal_weight
        self.segment_weight = segment_weight
        self.chamfer_weight = chamfer_weight
        # , pcd_center_reg_weight=0
        # self.pcd_center_reg_weight = pcd_center_reg_weight

    def get_rgb_loss(self, rgb_values, rgb_gt, weight=None):
        if weight is not None:
            rgb_loss = self.l1_loss(rgb_values.reshape(-1, 3) * weight.reshape(-1, 1), rgb_gt.reshape(-1, 3) * weight.reshape(-1, 1))
        else:
            rgb_loss = self.l1_loss(rgb_values.reshape(-1, 3), rgb_gt.reshape(-1, 3))
        return rgb_loss

    def get_normal_loss(self, normal_values, normal_gt, weight=None):
        if weight is not None:
            normal_loss = self.l1_loss(normal_values.reshape(-1, 3) * weight.reshape(-1, 1), normal_gt.reshape(-1, 3) * weight.reshape(-1, 1))
        else:
            normal_loss = self.l1_loss(normal_values.reshape(-1, 3), normal_gt.reshape(-1, 3))
        return normal_loss

    def get_segment_loss(self, segment_values, segment_gt, model_input, weight=None):
        if weight is not None:
            segment_loss = self.binary_cross_entropy_loss(segment_values.reshape(-1, 1) * weight.reshape(-1, 1), segment_gt.reshape(-1, 1) * weight.reshape(-1, 1))
        else:
            segment_loss = self.binary_cross_entropy_loss(segment_values.reshape(-1, 1), segment_gt.reshape(-1, 1))
        return segment_loss

    def get_lbs_loss(self, lbs_weight, gt_lbs_weight, use_var_expression=False):
        # the same function is used for lbs, shapedirs, posedirs.
        if use_var_expression and self.var_expression is not None:
            lbs_loss = torch.mean(self.l2_loss(lbs_weight, gt_lbs_weight) / self.var_expression.to(lbs_weight.device) / 50)
        else:
            lbs_loss = self.l2_loss(lbs_weight, gt_lbs_weight).mean()
        return lbs_loss

    def get_mask_loss(self, predicted_mask, object_mask):
        mask_loss = self.l1_loss(predicted_mask.reshape(-1).float(), object_mask.reshape(-1).float())
        return mask_loss

    def get_gt_blendshape(self, index_batch, flame_lbs_weights, flame_posedirs, flame_shapedirs, ghostbone):
        if ghostbone:
            # gt_lbs_weight = torch.zeros(len(index_batch), 6).cuda()                           # NOTE original code
            gt_lbs_weight = torch.zeros(len(index_batch), 6, device=flame_lbs_weights.device)   # NOTE because of lightning
            gt_lbs_weight[:, 1:] = flame_lbs_weights[index_batch, :]
        else:
            gt_lbs_weight = flame_lbs_weights[index_batch, :]

        gt_shapedirs = flame_shapedirs[index_batch, :, 100:]
        gt_posedirs = torch.transpose(flame_posedirs.reshape(36, -1, 3), 0, 1)[index_batch, :, :]
        # gt_beta_shapedirs = flame_shapedirs[index_batch, :, :100]
        output = {
            'gt_lbs_weights': gt_lbs_weight,
            'gt_posedirs': gt_posedirs,
            'gt_shapedirs': gt_shapedirs,
            # 'gt_beta_shapedirs': gt_beta_shapedirs,
        }
        return output

    def get_sdf_consistency_loss(self, sdf_values):
        return torch.mean(sdf_values * sdf_values)

    def get_eikonal_loss(self, grad_theta):
        assert grad_theta.shape[1] == 3
        assert len(grad_theta.shape) == 2
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_pcd_center_reg_loss(self, pcd_center):
        zero_coord = torch.zeros_like(pcd_center)
        pcd_center_reg_loss = self.l1_loss(pcd_center, zero_coord)
        return pcd_center_reg_loss

    def forward(self, model_outputs, ground_truth, model_input=None):
        predicted_mask = model_outputs['predicted_mask']
        object_mask = ground_truth['object_mask']
        mask_loss = self.get_mask_loss(predicted_mask, object_mask)

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_image'], ground_truth['rgb'])
        loss = rgb_loss + self.mask_weight * mask_loss

        out = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'mask_loss': mask_loss,
        }

        if self.vgg_feature_weight > 0:
            bz = model_outputs['batch_size']
            img_res = model_outputs['img_res']
            gt = ground_truth['rgb'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

            predicted = model_outputs['rgb_image'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

            vgg_loss = self.get_vgg_loss(predicted, gt)
            out['vgg_loss'] = vgg_loss
            out['loss'] += vgg_loss * self.vgg_feature_weight

        if self.sdf_consistency_weight > 0:
            assert self.eikonal_weight > 0
            sdf_consistency_loss = self.get_sdf_consistency_loss(model_outputs['sdf_values'])
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_thetas'])
            out['loss'] += sdf_consistency_loss * self.sdf_consistency_weight + eikonal_loss * self.eikonal_weight
            out['sdf_consistency'] = sdf_consistency_loss
            out['eikonal'] = eikonal_loss

        if self.lbs_weight != 0:
            num_points = model_outputs['lbs_weights'].shape[0]
            ghostbone = model_outputs['lbs_weights'].shape[-1] == 6
            outputs = self.get_gt_blendshape(model_outputs['index_batch'], model_outputs['flame_lbs_weights'],
                                             model_outputs['flame_posedirs'], model_outputs['flame_shapedirs'],
                                             ghostbone)

            lbs_loss = self.get_lbs_loss(model_outputs['lbs_weights'].reshape(num_points, -1),
                                             outputs['gt_lbs_weights'].reshape(num_points, -1),
                                             )

            out['loss'] += lbs_loss * self.lbs_weight * 0.1
            out['lbs_loss'] = lbs_loss

            gt_posedirs = outputs['gt_posedirs'].reshape(num_points, -1)
            posedirs_loss = self.get_lbs_loss(model_outputs['posedirs'].reshape(num_points, -1) * 10,
                                              gt_posedirs* 10,
                                              )
            out['loss'] += posedirs_loss * self.lbs_weight * 10.0
            out['posedirs_loss'] = posedirs_loss

            gt_shapedirs = outputs['gt_shapedirs'].reshape(num_points, -1)
            shapedirs_loss = self.get_lbs_loss(model_outputs['shapedirs'].reshape(num_points, -1)[:, :50*3] * 10,
                                               gt_shapedirs * 10,
                                               use_var_expression=True,
                                               )
            out['loss'] += shapedirs_loss * self.lbs_weight * 10.0
            out['shapedirs_loss'] = shapedirs_loss

            # gt_beta_shapedirs = outputs['gt_beta_shapedirs'].reshape(num_points, -1)
            # beta_shapedirs_loss = self.get_lbs_loss(model_outputs['beta_shapedirs'].reshape(num_points, -1)[:, :100*3] * 10,
            #                                         gt_beta_shapedirs * 10)
            # out['loss'] += beta_shapedirs_loss * self.lbs_weight * 10.0
            # out['beta_shapedirs_loss'] = beta_shapedirs_loss
        
        if self.optimize_scene_latent_code and self.deform_pcd_weight > 0:
            deform_pcd_loss = self.get_sdf_consistency_loss(model_outputs['delta_x'])
            out['loss'] += deform_pcd_loss * self.deform_pcd_weight
            out['deform_pcd_loss'] = deform_pcd_loss
        
        if self.optimize_scene_latent_code and self.latent_reg_weight > 0:
            latent_reg_loss = self.get_sdf_consistency_loss(model_outputs['scene_latent_code'])
            out['loss'] += latent_reg_loss * self.latent_reg_weight
            out['latent_reg_loss'] = latent_reg_loss
        
        if self.normal_weight > 0:
            if 'target' not in model_input['sub_dir'][0]:
                normal_image_output = model_outputs['normal_image'].squeeze().reshape(-1, 3)
                mask_object_gt = ground_truth["mask_object"].reshape(-1, 1).float()
                object_mask_gt = ground_truth["object_mask"].reshape(-1, 1).float()

                normal_original_output = normal_image_output * (1 - mask_object_gt) + mask_object_gt
                normal_original_output = normal_original_output * object_mask_gt + (1 - object_mask_gt)
                
                normal_rendering_output = normal_image_output * mask_object_gt + (1 - mask_object_gt)
                normal_rendering_output = normal_rendering_output * object_mask_gt + (1 - object_mask_gt)

                normal_loss = self.get_normal_loss(normal_original_output, ground_truth['normal_original']) + self.get_normal_loss(normal_rendering_output, ground_truth['normal_rendering'])
                out['loss'] += normal_loss * self.normal_weight
                out['normal_loss'] = normal_loss

            elif 'target' in model_input['sub_dir'][0]:
                normal_image_output = model_outputs['normal_image'].squeeze().reshape(-1, 3)
                object_mask_gt = ground_truth["object_mask"].reshape(-1, 1).float()

                normal_rendering_output = normal_image_output * object_mask_gt + (1 - object_mask_gt)

                normal_loss = self.get_normal_loss(normal_rendering_output, ground_truth['normal_rendering'])
                out['loss'] += normal_loss * self.normal_weight
                out['normal_loss'] = normal_loss
        
        if self.segment_weight > 0:
            segment_image_output = model_outputs['segment_image'].squeeze().reshape(-1, 1)
            mask_object_gt = ground_truth["mask_object"].reshape(-1, 1).float()

            segment_loss = self.get_segment_loss(segment_image_output, mask_object_gt, model_input)
            out['loss'] += segment_loss * self.segment_weight
            out['segment_loss'] = segment_loss
        
        # if self.pcd_center_reg_weight > 0:
        #     pcd_center_reg_loss = self.get_pcd_center_reg_loss(model_outputs['pcd_center'])
        #     out['loss'] += pcd_center_reg_loss * self.pcd_center_reg_weight
        #     out['pcd_center_reg_loss'] = pcd_center_reg_loss

        return out



class Loss_lightning_singeGPU(nn.Module):
    def __init__(self, mask_weight, var_expression=None, lbs_weight=0,sdf_consistency_weight=0, eikonal_weight=0, vgg_feature_weight=0, optimize_scene_latent_code=False, deform_pcd_weight=0, latent_reg_weight=0, normal_weight=0):
        super().__init__()
        
        self.mask_weight = mask_weight
        self.lbs_weight = lbs_weight
        self.sdf_consistency_weight = sdf_consistency_weight
        self.eikonal_weight = eikonal_weight
        self.vgg_feature_weight = vgg_feature_weight
        self.var_expression = var_expression                                                                      # NOTE original code
        # NOTE custom ###############
        # if var_expression is not None:
        #     # self.var_expression = self.var_expression.unsqueeze(1).expand(1, 3, -1).reshape(1, -1).cuda()         # NOTE original code
        #     self.register_buffer('var_expression', var_expression.unsqueeze(1).expand(1, 3, -1).reshape(1, -1))
        # else:
        #     self.var_expression = var_expression
        if self.var_expression is not None:
            self.var_expression = self.var_expression.unsqueeze(1).expand(1, 3, -1).reshape(1, -1).cuda()   
        #############################
        print("Expression variance: ", self.var_expression)

        if self.vgg_feature_weight > 0:
            self.get_vgg_loss = VGGPerceptualLoss().cuda()            # NOTE original code
            # self.get_vgg_loss = VGGPerceptualLoss()                     # NOTE because of lightning

        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='none')

        self.optimize_scene_latent_code = optimize_scene_latent_code
        self.deform_pcd_weight = deform_pcd_weight
        self.latent_reg_weight = latent_reg_weight
        self.normal_weight = normal_weight

    def get_rgb_loss(self, rgb_values, rgb_gt, weight=None):
        if weight is not None:
            rgb_loss = self.l1_loss(rgb_values.reshape(-1, 3) * weight.reshape(-1, 1), rgb_gt.reshape(-1, 3) * weight.reshape(-1, 1))
        else:
            rgb_loss = self.l1_loss(rgb_values.reshape(-1, 3), rgb_gt.reshape(-1, 3))
        return rgb_loss

    def get_normal_loss(self, normal_values, normal_gt, weight=None):
        if weight is not None:
            normal_loss = self.l1_loss(normal_values.reshape(-1, 3) * weight.reshape(-1, 1), normal_gt.reshape(-1, 3) * weight.reshape(-1, 1))
        else:
            normal_loss = self.l1_loss(normal_values.reshape(-1, 3), normal_gt.reshape(-1, 3))
        return normal_loss

    def get_lbs_loss(self, lbs_weight, gt_lbs_weight, use_var_expression=False):
        # the same function is used for lbs, shapedirs, posedirs.
        if use_var_expression and self.var_expression is not None:
            lbs_loss = torch.mean(self.l2_loss(lbs_weight, gt_lbs_weight) / self.var_expression / 50)
        else:
            lbs_loss = self.l2_loss(lbs_weight, gt_lbs_weight).mean()
        return lbs_loss

    def get_mask_loss(self, predicted_mask, object_mask):
        mask_loss = self.l1_loss(predicted_mask.reshape(-1).float(), object_mask.reshape(-1).float())
        return mask_loss

    def get_gt_blendshape(self, index_batch, flame_lbs_weights, flame_posedirs, flame_shapedirs, ghostbone):
        if ghostbone:
            # gt_lbs_weight = torch.zeros(len(index_batch), 6).cuda()                           # NOTE original code
            gt_lbs_weight = torch.zeros(len(index_batch), 6, device=flame_lbs_weights.device)   # NOTE because of lightning
            gt_lbs_weight[:, 1:] = flame_lbs_weights[index_batch, :]
        else:
            gt_lbs_weight = flame_lbs_weights[index_batch, :]

        gt_shapedirs = flame_shapedirs[index_batch, :, 100:]
        gt_posedirs = torch.transpose(flame_posedirs.reshape(36, -1, 3), 0, 1)[index_batch, :, :]
        # gt_beta_shapedirs = flame_shapedirs[index_batch, :, :100]
        output = {
            'gt_lbs_weights': gt_lbs_weight,
            'gt_posedirs': gt_posedirs,
            'gt_shapedirs': gt_shapedirs,
            # 'gt_beta_shapedirs': gt_beta_shapedirs,
        }
        return output

    def get_sdf_consistency_loss(self, sdf_values):
        return torch.mean(sdf_values * sdf_values)

    def get_eikonal_loss(self, grad_theta):
        assert grad_theta.shape[1] == 3
        assert len(grad_theta.shape) == 2
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def forward(self, model_outputs, ground_truth):
        predicted_mask = model_outputs['predicted_mask']
        object_mask = ground_truth['object_mask']
        mask_loss = self.get_mask_loss(predicted_mask, object_mask)

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_image'], ground_truth['rgb'])
        loss = rgb_loss + self.mask_weight * mask_loss

        out = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'mask_loss': mask_loss,
        }
        if self.vgg_feature_weight > 0:
            bz = model_outputs['batch_size']
            img_res = model_outputs['img_res']
            gt = ground_truth['rgb'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

            predicted = model_outputs['rgb_image'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

            vgg_loss = self.get_vgg_loss(predicted, gt)
            out['vgg_loss'] = vgg_loss
            out['loss'] += vgg_loss * self.vgg_feature_weight

        if self.sdf_consistency_weight > 0:
            assert self.eikonal_weight > 0
            sdf_consistency_loss = self.get_sdf_consistency_loss(model_outputs['sdf_values'])
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_thetas'])
            out['loss'] += sdf_consistency_loss * self.sdf_consistency_weight + eikonal_loss * self.eikonal_weight
            out['sdf_consistency'] = sdf_consistency_loss
            out['eikonal'] = eikonal_loss

        if self.lbs_weight != 0:
            num_points = model_outputs['lbs_weights'].shape[0]
            ghostbone = model_outputs['lbs_weights'].shape[-1] == 6
            outputs = self.get_gt_blendshape(model_outputs['index_batch'], model_outputs['flame_lbs_weights'],
                                             model_outputs['flame_posedirs'], model_outputs['flame_shapedirs'],
                                             ghostbone)

            lbs_loss = self.get_lbs_loss(model_outputs['lbs_weights'].reshape(num_points, -1),
                                             outputs['gt_lbs_weights'].reshape(num_points, -1),
                                             )

            out['loss'] += lbs_loss * self.lbs_weight * 0.1
            out['lbs_loss'] = lbs_loss

            gt_posedirs = outputs['gt_posedirs'].reshape(num_points, -1)
            posedirs_loss = self.get_lbs_loss(model_outputs['posedirs'].reshape(num_points, -1) * 10,
                                              gt_posedirs* 10,
                                              )
            out['loss'] += posedirs_loss * self.lbs_weight * 10.0
            out['posedirs_loss'] = posedirs_loss

            gt_shapedirs = outputs['gt_shapedirs'].reshape(num_points, -1)
            shapedirs_loss = self.get_lbs_loss(model_outputs['shapedirs'].reshape(num_points, -1)[:, :50*3] * 10,
                                               gt_shapedirs * 10,
                                               use_var_expression=True,
                                               )
            out['loss'] += shapedirs_loss * self.lbs_weight * 10.0
            out['shapedirs_loss'] = shapedirs_loss

            # gt_beta_shapedirs = outputs['gt_beta_shapedirs'].reshape(num_points, -1)
            # beta_shapedirs_loss = self.get_lbs_loss(model_outputs['beta_shapedirs'].reshape(num_points, -1)[:, :100*3] * 10,
            #                                         gt_beta_shapedirs * 10)
            # out['loss'] += beta_shapedirs_loss * self.lbs_weight * 10.0
            # out['beta_shapedirs_loss'] = beta_shapedirs_loss

        if self.optimize_scene_latent_code and self.deform_pcd_weight > 0:
            deform_pcd_loss = self.get_sdf_consistency_loss(model_outputs['delta_x'])
            out['loss'] += deform_pcd_loss * self.deform_pcd_weight
            out['deform_pcd_loss'] = deform_pcd_loss
        
        if self.optimize_scene_latent_code and self.latent_reg_weight > 0:
            latent_reg_loss = self.get_sdf_consistency_loss(model_outputs['scene_latent_code'])
            out['loss'] += latent_reg_loss * self.latent_reg_weight
            out['latent_reg_loss'] = latent_reg_loss
        
        if self.normal_weight > 0:
            normal_image_output = model_outputs['normal_image'].squeeze().reshape(-1, 3)
            mask_object_gt = ground_truth["mask_object"].reshape(-1, 1).float()
            object_mask_gt = ground_truth["object_mask"].reshape(-1, 1).float()

            normal_original_output = normal_image_output * (1 - mask_object_gt) + mask_object_gt
            normal_original_output = normal_original_output * object_mask_gt + (1 - object_mask_gt)
            
            normal_rendering_output = normal_image_output * mask_object_gt + (1 - mask_object_gt)
            normal_rendering_output = normal_rendering_output * object_mask_gt + (1 - object_mask_gt)

            normal_loss = self.get_normal_loss(normal_original_output, ground_truth['normal_original']) + self.get_normal_loss(normal_rendering_output, ground_truth['normal_rendering'])
            out['loss'] += normal_loss * self.normal_weight
            out['normal_loss'] = normal_loss

            # import sys
            # sys.path.append('..')
            # import utils.hutils as hutils
            # import cv2
            # import numpy as np

            # device = model_outputs['rgb_image'].device
            # normal_image_output = model_outputs['normal_image'].squeeze().reshape(-1, 3)

            # # mask_object_gt = ground_truth["mask_object"].reshape(-1, 1).float()

            # mask_np = ground_truth["mask_object"].reshape(512, 512).cpu().numpy()
            # _, mask_binary = cv2.threshold(mask_np, 0.5, 1, cv2.THRESH_BINARY)
            # # mask_binary = 1.0 - mask_binary
            # mask_binary_uint8 = (mask_binary * 255).astype(np.uint8)
            # dist_transform = cv2.distanceTransform(mask_binary_uint8, cv2.DIST_L2, 5)
            # min_val = dist_transform.min()
            # max_val = dist_transform.max()
            # avg_val = (min_val + max_val) / 5 # 60
            # normalized_dist_transform = (dist_transform - min_val) / (avg_val - min_val) * 255
            # normalized_dist_transform_uint8 = np.clip(normalized_dist_transform, 0, 255).astype(np.uint8)
            # mask_object_gt = normalized_dist_transform_uint8 / 255.0
            # mask_object_gt = torch.tensor(mask_object_gt, device=device).reshape(-1, 1).float()
            # # cv2.imwrite('normalized_distance_transform.png', normalized_dist_transform_uint8)

            # object_mask_gt = ground_truth["object_mask"].reshape(-1, 1).float()

            # normal_original_output = normal_image_output * (1 - mask_object_gt) + mask_object_gt
            # normal_original_output = normal_original_output * object_mask_gt + (1 - object_mask_gt)

            # normal_rendering_output = normal_image_output * mask_object_gt + (1 - mask_object_gt)
            # normal_rendering_output = normal_rendering_output * object_mask_gt + (1 - object_mask_gt)

            # normal_original_gt = ground_truth['normal_original'].squeeze() * (1 - mask_object_gt) + mask_object_gt
            # normal_original_gt = normal_original_gt * object_mask_gt + (1 - object_mask_gt)

            # normal_rendering_gt = ground_truth['normal_rendering'].squeeze() * mask_object_gt + (1 - mask_object_gt)
            # normal_rendering_gt = normal_rendering_gt * object_mask_gt + (1 - object_mask_gt)

            # hutils.visualize_rgb_to_file(normal_original_output.detach().cpu(), 'normal_original_output.png')
            # hutils.visualize_rgb_to_file(normal_rendering_output.detach().cpu(), 'normal_rendering_output.png')
            # # hutils.visualize_rgb_to_file(ground_truth['normal_original'].reshape(-1, 3).detach().cpu(), 'normal_original_gt.png')
            # # hutils.visualize_rgb_to_file(ground_truth['normal_rendering'].reshape(-1, 3).detach().cpu(), 'normal_rendering_gt.png')
            # hutils.visualize_rgb_to_file(normal_original_gt.detach().cpu(), 'normal_original_gt.png')
            # hutils.visualize_rgb_to_file(normal_rendering_gt.detach().cpu(), 'normal_rendering_gt.png')

            # normal_loss = self.get_normal_loss(normal_original_output, ground_truth['normal_original']) + self.get_normal_loss(normal_rendering_output, ground_truth['normal_rendering'])
            # out['loss'] += normal_loss * self.normal_weight
            # out['normal_loss'] = normal_loss

        return out