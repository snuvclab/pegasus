import numpy as np
import torch
import torchvision
import trimesh
from PIL import Image
import os
import cv2
import wandb
from einops import rearrange
import warnings
warnings.filterwarnings("ignore")

SAVE_OBJ_LIST = [1]

def save_pcl_to_ply(filename, points, colors=None, normals=None):
    save_dir=os.path.dirname(os.path.abspath(filename))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if colors is not None:
        colors = colors.cpu().detach().numpy()
    if normals is not None:
        normals = normals.cpu().detach().numpy()
    mesh = trimesh.Trimesh(vertices=points.detach().cpu().numpy(),vertex_normals = normals, vertex_colors = colors)
    #there is a bug in trimesh of it only saving normals when we tell the exporter explicitly to do so for point clouds.
    #thus we are calling the exporter directly instead of mesh.export(...)
    f = open(filename, "wb")
    data = trimesh.exchange.ply.export_ply(mesh, vertex_normal=True)
    f.write(data)
    f.close()
    return


# def plot(img_index, model_outputs, ground_truth, path, epoch, img_res, is_eval=False, first=False, custom_settings=None):
def plot(img_index, model_outputs, ground_truth, path, epoch, img_res, first=False, custom_settings=None):
    # arrange data to plot
    batch_size = model_outputs['batch_size']
    # plot_images(model_outputs, ground_truth, path, epoch, img_index, 1, img_res, batch_size, is_eval, custom_settings)
    plot_images(model_outputs, ground_truth, path, epoch, img_index, 1, img_res, batch_size, custom_settings)

    canonical_color = torch.clamp(model_outputs['pnts_albedo'], 0., 1.)
    # if not is_eval:
    #     return
    if custom_settings is not None and 'novel_view' not in custom_settings: # hyunsoo added
        for idx, img_idx in enumerate(img_index):
            # wo_epoch_path = path[idx].replace('/epoch_{}'.format(epoch), '')
            wo_epoch_path = os.path.dirname(path[idx])
            if img_idx in SAVE_OBJ_LIST:
                deformed_color = model_outputs["pnts_color_deformed"].reshape(batch_size, -1, 3)[idx]
                filename = '{0}/{1:04d}_deformed_color_{2}.ply'.format(wo_epoch_path, epoch, img_idx)
                save_pcl_to_ply(filename, model_outputs['deformed_points'].reshape(batch_size, -1, 3)[idx],
                                normals=model_outputs["pnts_normal_deformed"].reshape(batch_size, -1, 3)[idx],
                                colors=deformed_color)

                filename = '{0}/{1:04d}_deformed_albedo_{2}.ply'.format(wo_epoch_path, epoch, img_idx)
                save_pcl_to_ply(filename, model_outputs['deformed_points'].reshape(batch_size, -1, 3)[idx],
                                normals=model_outputs["pnts_normal_deformed"].reshape(batch_size, -1, 3)[idx],
                                colors=canonical_color)
        if first:
            # wo_epoch_path = path[0].replace('/epoch_{}'.format(epoch), '')
            wo_epoch_path = os.path.dirname(path[0])
            filename = '{0}/{1:04d}_canonical_points_albedo.ply'.format(wo_epoch_path, epoch)
            save_pcl_to_ply(filename, model_outputs["canonical_points"], colors=canonical_color)

            if 'unconstrained_canonical_points' in model_outputs:
                filename = '{0}/{1:04d}_unconstrained_canonical_points.ply'.format(wo_epoch_path, epoch)
                save_pcl_to_ply(filename, model_outputs['unconstrained_canonical_points'],
                                colors=canonical_color)
        # if epoch == 0 or is_eval:
        if epoch == 0:
            if first:
                # wo_epoch_path = path[0].replace('/epoch_{}'.format(epoch), '')
                wo_epoch_path = os.path.dirname(path[0])
                filename = '{0}/{1:04d}_canonical_verts.ply'.format(wo_epoch_path, epoch)
                save_pcl_to_ply(filename, model_outputs['canonical_verts'].reshape(-1, 3),
                                colors=get_lbs_color(model_outputs['flame_lbs_weights']))


def plot_image(rgb, path, img_index, plot_nrow, img_res, type, fill=False):
    rgb_plot = lin2img(rgb, img_res)

    tensor = torchvision.utils.make_grid(rgb_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = np.clip(tensor, 0., 1.)
    tensor = (tensor * scale_factor).astype(np.uint8)

    if fill:
        kernel = np.ones((3, 3), np.uint8)
        tensor = cv2.erode(tensor, kernel, iterations=1)            # NOTE 흰색 노이즈를 제거
        tensor = cv2.dilate(tensor, kernel, iterations=1)           # NOTE 구멍이나 간격을 메움.

        img = Image.fromarray(tensor)
        if not os.path.exists('{0}/{1}_erode_dilate'.format(path, type)):
            os.mkdir('{0}/{1}_erode_dilate'.format(path, type))
        img.save('{0}/{2}_erode_dilate/{1}.png'.format(path, img_index, type))
    else:
        img = Image.fromarray(tensor)
        if not os.path.exists('{0}/{1}'.format(path, type)):
            os.mkdir('{0}/{1}'.format(path, type))
        img.save('{0}/{2}/{1}.png'.format(path, img_index, type))


def plot_mask(mask_tensor, path, img_index, plot_nrow, img_res, type):
    # Reshape the tensor into a square 2D tensor
    mask_tensor = mask_tensor.cpu()
    side_length = int(torch.sqrt(torch.tensor(mask_tensor.numel())).item())
    mask_2d = mask_tensor.reshape(side_length, side_length)

    # Convert the 2D tensor to a NumPy array
    mask_np = mask_2d.numpy().astype(np.uint8) * 255

    mask_np = cv2.resize(mask_np, img_res)

    img = Image.fromarray(mask_np)
    if not os.path.exists('{0}/{1}'.format(path, type)):
        os.mkdir('{0}/{1}'.format(path, type))
    img.save('{0}/{2}/{1}.png'.format(path, img_index, type))

def get_lbs_color(lbs_points):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Paired')
    red = cmap.colors[5]
    cyan = cmap.colors[3]
    blue = cmap.colors[1]
    pink = [1, 1, 1]

    if lbs_points.shape[-1] == 5:
        colors = torch.from_numpy(
            np.stack([np.array(cyan), np.array(blue), np.array(pink), np.array(pink), np.array(pink)])[
                None]).cuda()
    else:
        colors = torch.from_numpy(
            np.stack([np.array(red), np.array(cyan), np.array(blue), np.array(pink), np.array(pink), np.array(pink)])[
                None]).cuda()
    lbs_points = (colors * lbs_points[:, :, None]).sum(1)
    return lbs_points


# def plot_images(model_outputs, ground_truth, path, epoch, img_index, plot_nrow, img_res, batch_size, is_eval, custom_settings):
def plot_images(model_outputs, ground_truth, path, epoch, img_index, plot_nrow, img_res, batch_size, custom_settings):
    num_samples = img_res[0] * img_res[1]

    device = ground_truth['rgb'].device
    wandb_image_num = 0
    if 'rgb' in ground_truth:
        wandb_image_num += 1
        rgb_gt = ground_truth['rgb']
        if 'rendered_landmarks' in model_outputs:
            rendered_landmarks = model_outputs['rendered_landmarks'].reshape(batch_size, num_samples, 3)
            rgb_gt = rgb_gt * (1 - rendered_landmarks) + rendered_landmarks * torch.tensor([1, 0, 0]).to(device) # .cuda()
    else:
        rgb_gt = None
    rgb_points = model_outputs['rgb_image']
    rgb_points = rgb_points.reshape(batch_size, num_samples, 3)

    if 'rendered_landmarks' in model_outputs:
        rendered_landmarks = model_outputs['rendered_landmarks'].reshape(batch_size, num_samples, 3)
        rgb_points_rendering = rgb_points * (1 - rendered_landmarks) + rendered_landmarks * torch.tensor([1, 0, 0]).to(device) # .cuda()
        output_vs_gt = rgb_points_rendering
    else:
        output_vs_gt = rgb_points

    normal_points = model_outputs['normal_image']
    normal_points = normal_points.reshape(batch_size, num_samples, 3)       # NOTE result: (1, 262144, 3)

    if rgb_gt is not None:
        wandb_image_num += 2
        output_vs_gt = torch.cat((output_vs_gt, rgb_gt, normal_points), dim=0)
    else:
        output_vs_gt = torch.cat((output_vs_gt, normal_points), dim=0)

    if 'shading_image' in model_outputs:
        wandb_image_num += 2
        output_vs_gt = torch.cat((output_vs_gt, model_outputs['shading_image'].reshape(batch_size, num_samples, 3)), dim=0)
        output_vs_gt = torch.cat((output_vs_gt, model_outputs['albedo_image'].reshape(batch_size, num_samples, 3)), dim=0)

    # NOTE 추가함.
    # if 'segment_image' in model_outputs:
    #     segment_points = model_outputs['segment_image'].squeeze(0)
    #     segment_points = segment_points.reshape(batch_size, img_res[0], img_res[1], 10)
    #     segment_rgb_underneath = rgb_points.reshape(batch_size, img_res[0], img_res[1], 3).squeeze().detach().cpu().numpy()
    #     segment_parsing = segment_points.squeeze().argmax(dim=-1).detach().cpu().numpy()
    #     segment_image = vis_parsing_maps(segment_parsing, device=output_vs_gt.device).reshape(batch_size, num_samples, 3) / 255.
    #     output_vs_gt = torch.cat((output_vs_gt, segment_image), dim=0)

    output_vs_gt_plot = lin2img(output_vs_gt, img_res)
    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=batch_size).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)
    
    if custom_settings is not None and 'wandb_logger' in custom_settings:
        wandb_logger = custom_settings['wandb_logger'] # hyunsoo added

        wandb_tensor = torchvision.utils.make_grid(output_vs_gt_plot[:wandb_image_num, ...],
                                                   scale_each=False,
                                                   normalize=False,
                                                   nrow=output_vs_gt.shape[0]).cpu().detach().numpy()
        scale_factor = 255
        wandb_tensor = (wandb_tensor * scale_factor).astype(np.uint8) # (516, 3600, 3)

        wandb_image = rearrange(wandb_tensor, 'c h w -> h w c')
        wandb_logger.experiment.log({"Eval":[wandb.Image((wandb_image).astype(np.uint8))], "global_step": custom_settings['global_step']})

    # hyunsoo added
    if custom_settings is None or 'novel_view' not in custom_settings:
        novel_view = ''
    else:
        novel_view = '_{}'.format(custom_settings['novel_view'])
    
    img = Image.fromarray(tensor)
    wo_epoch_path = os.path.dirname(path[0]) # path[0].replace('/epoch_{}{}'.format(epoch, novel_view), '')
    # if not os.path.exists('{0}/rendering{1}'.format(wo_epoch_path, novel_view)):
    
    # try:
    #     idx = str(img_index[0])
    # except:
    #     idx = str(img_index)

    # if custom_settings is not None and 'step' in custom_settings:
    #     idx += '_{}'.format(custom_settings['step'])

    rendering_switch = {
        'rendering_grid': True,
        'rendering_rgb': True,
        'rendering_rgb_dilate_erode': True,
        'rendering_normal': True,
        'rendering_normal_dilate_erode': True,
        'rendering_albedo': True,
        'rendering_shading': True,
        'rendering_segment': True,
        'rendering_mask_hole': True
    }

    if custom_settings is not None and 'rendering_select' in custom_settings:
        rendering_switch = custom_settings['rendering_select']
    
    if rendering_switch['rendering_grid']:
        os.makedirs('{0}/rendering{1}'.format(wo_epoch_path, novel_view), exist_ok=True)
        img.save('{0}/rendering{2}/epoch_{1:04d}.png'.format(wo_epoch_path, epoch, novel_view))

    if rendering_switch['rendering_rgb']:
        plot_image(rgb_points[[0]], path[0], img_index, plot_nrow, img_res, 'rgb')
    if rendering_switch['rendering_rgb_dilate_erode']:
        plot_image(rgb_points[[0]], path[0], img_index, plot_nrow, img_res, 'rgb', fill=True)
    if rendering_switch['rendering_normal']:
        plot_image(normal_points[[0]], path[0], img_index, plot_nrow, img_res, 'normal')
    if rendering_switch['rendering_normal_dilate_erode']:
        plot_image(normal_points[[0]], path[0], img_index, plot_nrow, img_res, 'normal', fill=True)
    if 'albedo_image' in model_outputs and rendering_switch['rendering_albedo']:
        plot_image(model_outputs['albedo_image'].reshape(batch_size, num_samples, 3)[[0]], path[0], img_index, plot_nrow, img_res, 'albedo', fill=True)
    if 'shading_image' in model_outputs and rendering_switch['rendering_shading']:
        plot_image(model_outputs['shading_image'].reshape(batch_size, num_samples, 3)[[0]], path[0], img_index, plot_nrow, img_res, 'shading', fill=True)
    if 'segment_image' in model_outputs and rendering_switch['rendering_segment']:
        plot_mask(model_outputs['segment_image'] > 0.5, path[0], img_index, plot_nrow, img_res, 'segment')
    if 'mask_hole' in model_outputs and rendering_switch['rendering_mask_hole']:
        plot_mask(model_outputs['mask_hole'] > 0.5, path[0], img_index, plot_nrow, img_res, 'mask_hole')

    # if is_eval:
    #     if img_index.ndim > 0: # hyunsoo added
    #         for i, idx in enumerate(img_index):
    #             plot_image(rgb_points[[i]], path[i], idx, plot_nrow, img_res, 'rgb')
    #             plot_image(normal_points[[i]], path[i], idx, plot_nrow, img_res, 'normal', fill=True)
    #             # albedo
    #             plot_image(model_outputs['albedo_image'].reshape(batch_size, num_samples, 3)[[i]], path[i], idx, plot_nrow, img_res, 'albedo', fill=True)
    #             plot_image(model_outputs['shading_image'].reshape(batch_size, num_samples, 3)[[i]], path[i], idx, plot_nrow, img_res, 'shading', fill=True)
                
    #             if 'segment_image' in model_outputs:
    #                 plot_mask(model_outputs['segment_image'] > 0.5, path[i], idx, plot_nrow, img_res, 'segment')
    #             if 'mask_hole' in model_outputs:
    #                 plot_mask(model_outputs['mask_hole'] > 0.5, path[i], idx, plot_nrow, img_res, 'mask_hole')
    #     else:
    #         # Hyunsoo added
    #         i = 0
    #         idx = img_index.item()
    #         plot_image(rgb_points[[i]], path[i], idx, plot_nrow, img_res, 'rgb')
    #         plot_image(normal_points[[i]], path[i], idx, plot_nrow, img_res, 'normal', fill=True)
    #         # albedo
    #         plot_image(model_outputs['albedo_image'].reshape(batch_size, num_samples, 3)[[i]], path[i], idx, plot_nrow, img_res, 'albedo', fill=True)
    #         plot_image(model_outputs['shading_image'].reshape(batch_size, num_samples, 3)[[i]], path[i], idx, plot_nrow, img_res, 'shading', fill=True)
                
    #         if 'segment_image' in model_outputs:
    #             plot_mask(model_outputs['segment_image'] > 0.5, path[i], idx, plot_nrow, img_res, 'segment')
    #         if 'mask_hole' in model_outputs:
    #             plot_mask(model_outputs['mask_hole'] > 0.5, path[i], idx, plot_nrow, img_res, 'mask_hole')

    del output_vs_gt


def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])


def vis_parsing_maps(parsing_anno, stride=1, device='cuda'):
    # Colors for all 20 parts
    # NOTE parsing_anno.shape: (512, 512)
    # NOTE im.shape: (512, 512, 3)
    cmap = torch.from_numpy(np.array([(204, 0, 0), 
                                      (51, 51, 255), 
                                      (0, 255, 255), 
                                      (102, 204, 0), 
                                      (255, 255, 0),
                                      (0, 0, 153), 
                                      (0, 0, 204), 
                                      (0, 204, 0),
                                      (85, 255, 0),
                                      (0, 0, 0)], dtype=np.uint8), device=device)
    color = torch.sum(parsing_anno[:, :, None] * cmap[None, :, :], 1)
    
    return color
    
    # im = np.array(im)
    # vis_im = im.copy().astype(np.uint8)
    # vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    # vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    # vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    # num_of_class = np.max(vis_parsing_anno)

    # part_colors = [[255, 255, 255],     # 0              # BG, White
    #                [255, 85, 0],    # 1                 # Beard, Bright Orange
    #                [170, 255, 0],   # 2                 # Ears, Lime Green
    #                [255, 170, 0],    # 3                # Eyebrows, Gold or Golden Yellow
    #                [255, 0, 170],   # 4                 # Eyes, Magenta
    #                [255, 255, 170],     # 5             # Hair $ Earrings, Light Yellow
    #                [255, 0, 255],    # 6                # Hat, Magenta (or Pure Magenta)
    #                [85, 0, 255],   # 7                  # Mouth, Blue Violet
    #                [0, 0, 255],    # 8                  # Nose, Blue (or Pure Blue)
    #                [85, 255, 0]]   # 9                  # Eyeglasses, Bright Lime Green

    # for pi in range(0, num_of_class):
    #     index = np.where(vis_parsing_anno == pi)
    #     vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    # vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # # print(vis_parsing_anno_color.shape, vis_im.shape)
    # vis_im = cv2.addWeighted(vis_im, 0.4, vis_parsing_anno_color, 0.6, 0) # cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR)

    # vis_im = torch.tensor(vis_im, device=device)
    # return vis_im           # (512, 512, 3)