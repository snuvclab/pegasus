import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import lossfunc

import cv2
import argparse
import os

import matplotlib.pyplot as plt

np.random.seed(0)


def projection(points, K, w2c, no_intrinsics=False):
    rot = w2c[:, np.newaxis, :3, :3]
    points_cam = torch.sum(points[..., np.newaxis, :] * rot, -1) + w2c[:, np.newaxis, :3, 3] # Rx + t
    if no_intrinsics:
        return points_cam

    points_cam_projected = points_cam
    points_cam_projected[..., :2] /= points_cam[..., [2]]
    points_cam[..., [2]] *= -1 # flip z axis???? why?

    i = points_cam_projected[..., 0] * K[0] + K[2] # K[0] = fx, K[2] = cx
    j = points_cam_projected[..., 1] * K[1] + K[3] # K[1] = fy, K[3] = cy
    points2d = torch.stack([i, j, points_cam_projected[..., -1]], dim=-1) # [i, j, z]
    return points2d


def inverse_projection(points2d, K, c2w):
    i = points2d[:, :, 0]
    j = points2d[:, :, 1]
    dirs = torch.stack([(i - K[2]) / K[0], (j - K[3]) / K[1], torch.ones_like(i) * -1], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:, np.newaxis, :3, :3], -1)
    rays_d = F.normalize(rays_d, dim=-1)
    rays_o = c2w[:, np.newaxis, :3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def render_vertices_save_image_with_depth(vertices, file_name='rendered_vertices_with_depth.png'):
    if not torch.is_tensor(vertices):
        raise TypeError("vertices must be a PyTorch tensor")
    # vertices = rotate_vertices_torch(vertices)
    if vertices.device.type == 'cuda':
        vertices = vertices.detach().cpu()  # Move tensor to CPU for processing
    
    data_tensor = vertices
    # Extract x, y, and z coordinates
    x = data_tensor[0, :, 0].numpy()
    y = data_tensor[0, :, 1].numpy()
    z = data_tensor[0, :, 2].numpy()

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x, y, z)

    # Label the axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Title
    ax.set_title('3D Scatter Plot')

    # Save the plot
    plt.savefig('./3d_scatter_plot.png', dpi=300)

    # Close the plot display to ensure it works in headless environments
    plt.close()


def rotate_vertices_torch(vertices):
    # Convert degrees to radians for rotations
    theta_x = torch.deg2rad(torch.tensor(90.0)).to(vertices.device)  # 90 degrees rotation around X-axis
    theta_z = torch.deg2rad(torch.tensor(90.0)).to(vertices.device)  # 90 degrees rotation around Z-axis
    
    # Rotation matrices
    Rx = torch.tensor([[1, 0, 0],
                       [0, torch.cos(theta_x), -torch.sin(theta_x)],
                       [0, torch.sin(theta_x), torch.cos(theta_x)]]).to(vertices.device)
    
    Rz = torch.tensor([[torch.cos(theta_z), -torch.sin(theta_z), 0],
                       [torch.sin(theta_z), torch.cos(theta_z), 0],
                       [0, 0, 1]]).to(vertices.device)
    
    # Combine rotations by matrix multiplication
    R = torch.mm(Rz, Rx)  # First rotate around X, then rotate the result around Z
    
    # Apply rotation to all vertices
    # vertices shape is [1, N, 3], we reshape to [N, 3] for matrix multiplication
    vertices_rotated = torch.mm(vertices.reshape(-1, 3), R.t())  # Transpose R for correct multiplication
    
    # Reshape back to original shape [1, N, 3]
    return vertices_rotated.reshape(1, -1, 3)


class Optimizer(object):
    def __init__(self, device='cuda:0'):
        deca_cfg.model.use_tex = False
        # TODO: landmark_embedding.npy with eyes to optimize iris parameters
        deca_cfg.model.flame_lmk_embedding_path = os.path.join(deca_cfg.deca_dir, 'data',
                                                               'landmark_embedding_with_eyes.npy')
        deca_cfg.rasterizer_type = 'pytorch3d' # or 'standard'
        self.deca = DECA(config=deca_cfg, device=device)

    def optimize(self, shape, exp, landmark, pose, name, visualize_images, savefolder, intrinsics, json_path, size,
                 save_name, GLOBAL_POSE, GLOBAL_TRANS):
        '''
        By ChatGPT. Maybe there's something wrong comments.
        This code defines a method optimize in an object-oriented fashion. The method performs optimization on 4 parameters: shape, exp, landmark, and pose.
        
        The optimization is done using the Adam optimizer with a learning rate of 0.01. The optimization steps are run for 1001 iterations. 
        The loss function is the mean squared error between the predicted 2D keypoints (trans_landmarks2d) and the ground truth landmark plus some regularization terms to smooth the results over time.

        The parameters are initialized as PyTorch tensors and registered as parameters of the optimizer. The w2c_p is the camera-to-world transformation matrix.

        In every iteration, the FLAME module is used to generate vertices and 2D and 3D landmarks given the parameters shape, exp, and pose. 
        The vertices and landmarks are then projected into a 2D image plane using the projection function and the camera intrinsic parameters cam_intrinsics.

        The optimization steps update the parameters using backpropagation on the computed loss. 
        The optimization process and loss value are printed every 100 iterations.

        The global variables GLOBAL_POSE and use_iris are also used in the code.
        
        GLOBAL_POSE: if true, optimize global rotation, otherwise, only optimize head rotation (shoulder stays un-rotated)
        if GLOBAL_POSE is set to false, global translation is used.
        '''
        num_img = pose.shape[0]
        # we need to project to [-1, 1] instead of [0, size], hence modifying the cam_intrinsics as below
        cam_intrinsics = torch.tensor(
            [-1 * intrinsics[0] / size * 2, intrinsics[1] / size * 2, intrinsics[2] / size * 2 - 1,
             intrinsics[3] / size * 2 - 1]).float().cuda()
        
        # 2023.02.22. 만듦
        # GLOBAL_TRANS = False
        if GLOBAL_TRANS: # 2023.02.22. 추가. 원래는 GLOBAL_POSE였음
            translation_p = torch.tensor([0, 0, -4]).float().cuda() # initial translation, [3]
        else:
            translation_p = torch.tensor([0, 0, -4]).unsqueeze(0).expand(num_img, -1).float().cuda() # image 갯수만큼 translation을 만들어준다., [N, 3]

        if GLOBAL_POSE:
            pose = torch.cat([torch.zeros_like(pose[:, :3]), pose], dim=1) # pose: [N, 6] -> [N, 9], 앞에 0, 0, 0 추가했음
        if landmark.shape[1] == 70:
            # use iris landmarks, optimize gaze direction
            use_iris = True
        if use_iris:
            pose = torch.cat([pose, torch.zeros_like(pose[:, :6])], dim=1) # pose: [N, 9] -> [N, 15], 뒤 여섯자리에 0, 0, 0, 0, 0, 0 추가했음

        translation_p = nn.Parameter(translation_p) 
        # if GLOBAL_POSE is True, translation_p is [3]: only use same translation, 
        # otherwise, [N, 3]: there are N images translation
        # NOTE 자꾸 고개가 돌아가고 이상하게 나오는 것 같아서 pose는 일단 뺐긴 했는데.. 그럼 덜커덩하는 문제가 생길 수도 있다. 다시 확인을 해봐야할듯.
        # pose = nn.Parameter(pose) # assume iris is True,

        # if GLOBAL_POSE is True, pose is [N, 15]: optimize every poses (head, neck, mouth with jaw, eye), 
        # otherwise, [N, 12]: optimize some poses (neck, mouth with jaw, eye)
        exp = nn.Parameter(exp)
        shape = nn.Parameter(shape)

        # set optimizer
        if json_path is None:
            opt_p = torch.optim.Adam(
                [translation_p, pose, exp, shape],
                lr=1e-2)
        else:
            opt_p = torch.optim.Adam(
                [translation_p, pose, exp],
                lr=1e-2)

        # optimization steps
        len_landmark = landmark.shape[1]
        for k in range(1001): # 1001 -> 3001
            # pose_params[:, :3] (head rotation), neck_pose_params (3), pose_params[:, 3:] (jaw: mouth etc), eye_pose_params (6)
            full_pose = pose 
            if not use_iris:
                # iris가 optimize 대상이 아니므로, 뒤에 0, 0, 0, 0, 0, 0 추가, parameter에는 안들어가있는거임
                full_pose = torch.cat([full_pose, torch.zeros_like(full_pose[..., :6])], dim=1) 
            if not GLOBAL_POSE:
                # 앞에 0, 0, 0 추가 (head rotation 전부 0), parameter에는 안들어가있는거임
                full_pose = torch.cat([torch.zeros_like(full_pose[:, :3]), full_pose], dim=1) 
            verts_p, landmarks2d_p, landmarks3d_p = self.deca.flame(shape_params=shape.expand(num_img, -1),
                                                                    expression_params=exp,
                                                                    full_pose=full_pose)
            # CAREFUL: FLAME head is scaled by 4 to fit unit sphere tightly
            verts_p *= 4
            landmarks3d_p *= 4
            landmarks2d_p *= 4

            render_vertices_save_image_with_depth(verts_p[:1, ...])
            # perspective projection
            # Global rotation is handled in FLAME, set camera rotation matrix to identity
            ident = torch.eye(3).float().cuda().unsqueeze(0).expand(num_img, -1, -1) # 이미 flame이 돌아가있기 때문에 camera가 identity여도 된다

            if GLOBAL_TRANS: # 2023.02.22. 추가. 원래는 GLOBAL_POSE였음
                w2c_p = torch.cat([ident, translation_p.unsqueeze(0).expand(num_img, -1).unsqueeze(2)], dim=2) # [R|t]->[I|t=(0, 0, -4)] [N, 3, 4] 한개를 여러개로
            else:
                w2c_p = torch.cat([ident, translation_p.unsqueeze(2)], dim=2) # [R|t]->[I|t=(0, 0, -4) Same as above. [N, 3, 4] 원래 여러개.

            trans_landmarks2d = projection(landmarks2d_p, cam_intrinsics, w2c_p) # predicted 2D keypoints
            ## landmark loss
            landmark_loss2 = lossfunc.l2_distance(trans_landmarks2d[:, :len_landmark, :2], landmark[:, :len_landmark])
            total_loss = landmark_loss2 + torch.mean(torch.square(shape)) * 1e-2 + torch.mean(torch.square(exp)) * 1e-2
            total_loss += torch.mean(torch.square(exp[1:] - exp[:-1])) * 1e-1
            total_loss += torch.mean(torch.square(pose[1:] - pose[:-1])) * 10
            if not GLOBAL_POSE:
                total_loss += torch.mean(torch.square(translation_p[1:] - translation_p[:-1])) * 10

            opt_p.zero_grad()
            total_loss.backward()
            opt_p.step()

            # visualize
            if k % 100 == 0:
                with torch.no_grad():
                    loss_info = '----iter: {}, time: {}\n'.format(k,
                                                                  datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                    loss_info = loss_info + f'landmark_loss: {landmark_loss2}'
                    print(loss_info)
                    trans_verts = projection(verts_p[::50], cam_intrinsics, w2c_p[::50])
                    # trans_landmarks2d_for_visual = projection(landmarks2d_p, cam_intrinsics, w2c_p)
                    shape_images = self.deca.render.render_shape(verts_p[::50], trans_verts) # verts_p[::50].shape: [25, 5023, 3] | trans_verts.shape: [25, 5023, 3]
                    visdict = {
                        'inputs': visualize_images,
                        'gt_landmarks2d': util.tensor_vis_landmarks(visualize_images, landmark[::50]),
                        'landmarks2d': util.tensor_vis_landmarks(visualize_images, trans_landmarks2d.detach()[::50]),
                        'shape_images': shape_images
                    }
                    cv2.imwrite(os.path.join(savefolder, 'optimize_vis_{}.jpg'.format(str(k).zfill(4))), self.deca.visualize(visdict))

                    # shape_images = self.deca.render.render_shape(verts_p, trans_verts)
                    # print(shape_images.shape)

                    save = True
                    if save:
                        save_intrinsics = [-1 * intrinsics[0] / size, intrinsics[1] / size, intrinsics[2] / size,
                                           intrinsics[3] / size] # normalized intrinsics
                        dict = {}
                        frames = []
                        for i in range(num_img):
                            frames.append({'file_path': './image/' + name[i],
                                           'world_mat': w2c_p[i].detach().cpu().numpy().tolist(),
                                           'expression': exp[i].detach().cpu().numpy().tolist(),
                                           'pose': full_pose[i].detach().cpu().numpy().tolist(),
                                           'bbox': torch.stack(
                                               [torch.min(landmark[i, :, 0]), torch.min(landmark[i, :, 1]),
                                                torch.max(landmark[i, :, 0]), torch.max(landmark[i, :, 1])],
                                               dim=0).detach().cpu().numpy().tolist(),
                                           'flame_keypoints': trans_landmarks2d[i, :,
                                                              :2].detach().cpu().numpy().tolist()
                                           })

                        dict['frames'] = frames
                        dict['intrinsics'] = save_intrinsics
                        dict['shape_params'] = shape[0].cpu().numpy().tolist()
                        with open(os.path.join(savefolder, save_name + '.json'), 'w') as fp:
                            json.dump(dict, fp) # flame_params.json을 최종 저장한다.

        with torch.no_grad():
            for j in range(num_img):
                trans_verts = projection(verts_p[j].unsqueeze(0), cam_intrinsics, w2c_p[j].unsqueeze(0))
                # trans_landmarks2d_for_visual = projection(landmarks2d_p, cam_intrinsics, w2c_p)
                shape_images = self.deca.render.render_shape(verts_p[j].unsqueeze(0), trans_verts)
                # gt_landmarks2d = util.tensor_vis_landmarks(visualize_images, landmark[j].unsqueeze(0))
                # landmarks2d = util.tensor_vis_landmarks(visualize_images, trans_landmarks2d.detach()[j].unsqueeze(0))
                visdict = {
                    'inputs': visualize_images,
                    # 'gt_landmarks2d': gt_landmarks2d,
                    # 'landmarks2d': landmarks2d,
                    'shape_images': shape_images
                }
                for vis_name in ['shape_images']: # ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                    if vis_name not in visdict.keys():
                        continue
                    image = util.tensor2image(visdict[vis_name][0])
                    cv2.imwrite(os.path.join(savefolder, 'deca', name[j] + '_optim_' + vis_name +'.jpg'), image)
                # for i in range(visdict[vis_name].shape[0]):
                #     image = util.tensor2image(visdict[vis_name][i])
                #     cv2.imwrite(os.path.join(savefolder, 'deca', name[i] + '_' + vis_name +'.jpg'), image)
            # cv2.imwrite(os.path.join(savefolder, 'optimize_vis.jpg'), self.deca.visualize(visdict))

            # shape_images = self.deca.render.render_shape(verts_p, trans_verts)
            # print(shape_images.shape)


    def run(self, deca_code_file, face_kpts_file, iris_file, savefolder, image_path, json_path, intrinsics, size, 
            save_name, global_pose, global_trans):
        deca_code = json.load(open(deca_code_file, 'r')) # code.json from DECA
        face_kpts = json.load(open(face_kpts_file, 'r'))
        try:
            iris_kpts = json.load(open(iris_file, 'r'))
        except:
            iris_kpts = None
            print("Not using Iris keypoint")
        visualize_images = []
        shape = []
        exps = []
        landmarks = []
        poses = []
        name = []
        num_img = len(deca_code)
        # ffmpeg extracted frames, index starts with 1
        for k in range(1, num_img + 1):
            shape.append(torch.tensor(deca_code[str(k)]['shape']).float().cuda())
            exps.append(torch.tensor(deca_code[str(k)]['exp']).float().cuda())
            poses.append(torch.tensor(deca_code[str(k)]['pose']).float().cuda()) # from deca
            name.append(str(k))
            landmark = np.array(face_kpts['{}.png'.format(str(k))]).astype(np.float32)
            if iris_kpts is not None:
                iris = np.array(iris_kpts['{}.png'.format(str(k))]).astype(np.float32).reshape(2, 2)
                landmark = np.concatenate([landmark, iris[[1,0], :]], 0)
            landmark = landmark / size * 2 - 1
            landmarks.append(torch.tensor(landmark).float().cuda())
            if k % 50 == 1:
                image = cv2.imread(image_path + '/{}.png'.format(str(k))).astype(np.float32) / 255.
                image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
                visualize_images.append(torch.from_numpy(image[None, :, :, :]).cuda())

        shape = torch.cat(shape, dim=0)
        if json_path is None:
            shape = torch.mean(shape, dim=0).unsqueeze(0) # 각 frame마다 shape값이 다를텐데, 이를 다 평균냄. 같다고 가정하기 때문에.
        else:
            shape = torch.tensor(json.load(open(json_path, 'r'))['shape_params']).float().cuda().unsqueeze(0) # shape video에서 shape params를 들고온다. 말 그대로 얼굴의 형상을 들고온다.
        exps = torch.cat(exps, dim=0) # expression은 각 frame마다 다르다.
        landmarks = torch.stack(landmarks, dim=0) # landmark는 각 frame마다 다르다.
        poses = torch.cat(poses, dim=0) # pose는 각 frame마다 다르다.
        visualize_images = torch.cat(visualize_images, dim=0)
        # optimize
        self.optimize(shape, exps, landmarks, poses, name, visualize_images, savefolder, intrinsics, json_path, size, 
                      save_name, global_pose, global_trans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='Path to images and deca and landmark jsons')
    parser.add_argument('--shape_from', type=str, default='.', help="Use shape parameter from this video if given.")
    parser.add_argument('--save_name', type=str, default='flame_params', help='Name for json')
    parser.add_argument('--fx', type=float, default=1500)
    parser.add_argument('--fy', type=float, default=1500)
    parser.add_argument('--cx', type=float, default=256)
    parser.add_argument('--cy', type=float, default=256)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--global_pose', default=True, type=lambda x: x.lower() in ['true', '1']) # 원래도 켜져있었다.
    parser.add_argument('--global_trans', default=True, type=lambda x: x.lower() in ['true', '1']) # 원래는 global pose와 같은 것이었다.
    parser.add_argument('--device', type=int, default=0)
    # GLOBAL_POSE: if true, optimize global rotation, otherwise, only optimize head rotation (shoulder stays un-rotated)
    # if GLOBAL_POSE is set to false, global translation is used.
    # GLOBAL_POSE = True
    args = parser.parse_args()
    model = Optimizer()

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.device)

    image_path = os.path.join(args.path, 'image')
    if args.shape_from == '.':
        args.shape_from = None
        json_path = None
    else:
        json_path = os.path.join(args.shape_from, args.save_name + '.json')
    print("Optimizing: {}".format(args.path))
    intrinsics = [args.fx, args.fy, args.cx, args.cy]                                   # unnormalized
    model.run(deca_code_file=os.path.join(args.path, 'code.json'),                      # deca code, keypoint from face alignment, iris from iris detection
              face_kpts_file=os.path.join(args.path, 'keypoint.json'),
              iris_file=os.path.join(args.path, 'iris.json'), savefolder=args.path, image_path=image_path,
              json_path=json_path, intrinsics=intrinsics, size=args.size, save_name=args.save_name, global_pose=args.global_pose, global_trans=args.global_trans)

