import os
import torch
import numpy as np
import cv2
import json
import imageio
import skimage
from tqdm import tqdm
import utils.general as utils
import copy
import natsort


def find_index_closest_to_target(lst, target):
    # Since the list is sorted and increasing, we can use binary search
    low, high = 0, len(lst) - 1
    closest_index = -1

    while low <= high:
        mid = (low + high) // 2

        # Check if the mid element is less than or equal to the target
        if lst[mid] <= target:
            closest_index = mid
            low = mid + 1
        else:
            high = mid - 1

    return closest_index

# Updated function to find the index based on both name and number
def find_closest_index_by_name_and_number(file_paths, target_name, target_number):
    # Extracting the relevant parts from the file paths
    extracted_info = [(path.split('/')[5], int(path.split('/')[-1].split('.')[0])) for path in file_paths]

    # Finding the closest index
    closest_index = None
    for i, (name, num) in enumerate(extracted_info):
        if name == target_name and str(num) in target_number:
            return i
        if name == target_name and num < int(target_number.split('.')[0]):
            closest_index = i

    return closest_index

def find_indices_of_substring(lst, substring):
    start_index = end_index = -1

    for i, s in enumerate(lst):
        if substring in s:
            # If start_index is not set yet, set it to the current index
            if start_index == -1:
                start_index = i
            # Update end_index to the current index
            end_index = i

    return start_index, end_index


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self,
                 conf,
                 data_folder,
                 subject_name,
                 json_name,
                 sub_dir,
                 img_res,
                 mode,
                 subsample_type=None,
                 subsample=1,
                 hard_mask=False,
                 only_json=False,
                 use_mean_expression=False,
                 use_var_expression=False,
                 use_background=False,
                 load_images=False,
                 ):
        """
        sub_dir: list of scripts/testing subdirectories for the subject, e.g. [MVI_1810, MVI_1811]
        Data structure:
            RGB images in data_folder/subject_name/subject_name/sub_dir[i]/image
            foreground masks in data_folder/subject_name/subject_name/sub_dir[i]/mask
            json files containing FLAME parameters in data_folder/subject_name/subject_name/sub_dir[i]/json_name
        json file structure:
            frames: list of dictionaries, which are structured like:
                file_path: relative path to image
                world_mat: camera extrinsic matrix (world to camera). Camera rotation is actually the same for all frames,
                           since the camera is fixed during capture.
                           The FLAME head is centered at the origin, scaled by 4 times.
                expression: 50 dimension expression parameters
                pose: 15 dimension pose parameters
                flame_keypoints: 2D facial keypoints calculated from FLAME
            shape_params: 100 dimension FLAME shape parameters, shared by all scripts and testing frames of the subject
            intrinsics: camera focal length fx, fy and the offsets of the principal point cx, cy
        img_res: a list containing height and width, e.g. [256, 256] or [512, 512]
        subsample: subsampling the images to reduce frame rate, mainly used for inference and evaluation
        hard_mask: whether to use boolean segmentation mask or not
        only_json: used for testing, when there is no GT images or masks. If True, only load json.
        use_background: if False, replace with white background. Otherwise, use original background
        load_images: if True, load images at the beginning instead of at each iteration
        use_mean_expression: if True, use mean expression of the training set as the canonical expression
        use_var_expression: if True, blendshape regularization weight will depend on the variance of expression
                            (more regularization if variance is small in the training set.)
        """
        sub_dir = [str(dir) for dir in sub_dir]
        self.img_res = img_res
        self.use_background = use_background
        self.load_images = load_images
        self.hard_mask = hard_mask

        # NOTE custom #######################
        self.mode = mode
        self.shape_frame = conf.get_bool('dataset.shape_frame')             # 프레임별로 shape가 다를 수 있다고 가정한 dataset에 맞도록 되어있음. 일반적으로는 False로 두면 됨.
        self.shape_scene = conf.get_bool('dataset.shape_scene')
        self.shape_standard = conf.get_bool('dataset.shape_standard')
        self.shape_standard_scene = conf.get_string('dataset.shape_standard_scene')
        self.shape_test = conf.get_string('dataset.shape_test')
        a = self.shape_frame
        b = self.shape_scene
        c = self.shape_standard
        assert (a and not b and not c) or (not a and b and not c) or (not a and not b and c), "[ERROR] Only one variable can be True!"
        self.normal = True if conf.get_float('loss.normal_weight') > 0 and mode == 'train' else False
        self.segment = True if conf.get_float('loss.segment_weight') > 0 and mode == 'train' else False
        self.mask_object = conf.get_bool('dataset.mask_object')
        self.mask_source = conf.get_bool('dataset.mask_source')
        # self.mask_original = conf.get_bool('dataset.mask_original')
        #####################################

        self.data = {
            "image_paths": [],
            "mask_paths": [],
            # camera extrinsics
            "world_mats": [],
            # FLAME expression and pose parameters
            "expressions": [],
            "flame_pose": [],
            # saving image names and subdirectories
            "img_name": [],
            "sub_dir": []
        }
        # add normal_original_paths to self.data if self.normal is True
        if self.normal:
            # NOTE original부분
            self.data["normal_original_paths"] = []
            # NOTE rendering 부분
            self.data["normal_rendering_paths"] = []
        # if self.segment:
        #     self.data["segment_paths"] = []
        if self.mask_object:
            self.data["mask_object_paths"] = []

        if self.mask_source:
            self.data["mask_source_paths"] = []
        # if self.mask_original:
        #     self.data["mask_original_paths"] = []

        # NOTE custom #######################
        if self.shape_frame or self.shape_scene:
            self.data["shapes"] = []

        black_list_dir = os.path.join(data_folder, subject_name, subject_name, "blacklist.json")
        if os.path.exists(black_list_dir):
            # Assuming the JSON data is stored in a file named "data.json"
            with open(black_list_dir, "r") as file:
                self.blacklist = json.load(file)
        else:
            self.blacklist = None
        # NOTE image metric은 더이상 안써서 지우겠음.
        # if self.mode == 'test' and conf.get_bool('test.image_metric'):
        #     train_frames = np.load(os.path.join(data_folder, subject_name, subject_name, 'train.npy')).tolist()
        #     train_frames = [f"{num}.png" for num in train_frames]

        #     all_blacklist = []
        #     for k in self.blacklist.keys():
        #         if 'source' not in k and 'target' not in k:
        #             all_blacklist += self.blacklist[k]
        #     all_blacklist = list(set(all_blacklist))

        #     for k in self.blacklist.keys():
        #         self.blacklist[k] = natsort.natsorted(list(set(all_blacklist+train_frames)))
        
        num_images = []
        #####################################
        for i, dir in enumerate(sub_dir):
            instance_dir = os.path.join(data_folder, subject_name, subject_name, dir)
            assert os.path.exists(instance_dir), "Data directory {} is empty".format(instance_dir)

            cam_file = '{0}/{1}'.format(instance_dir, json_name)

            with open(cam_file, 'r') as f:
                camera_dict = json.load(f)
            
            # NOTE custom #######################
            length_cam = len(camera_dict['frames'])
            length_img = len(os.listdir(os.path.join(instance_dir, 'image')))
            num_images.append(length_cam)
            if length_img != length_cam:
                print("[WARN] In directory {}, number of camera_dict {} != number of real image {}".format(dir, length_cam, length_img))
            #####################################

            for frame in camera_dict['frames']:
                # world to camera matrix
                world_mat = np.array(frame['world_mat']).astype(np.float32)
                # camera to world matrix
                self.data["world_mats"].append(world_mat)
                self.data["expressions"].append(np.array(frame['expression']).astype(np.float32))
                self.data["flame_pose"].append(np.array(frame['pose']).astype(np.float32))
                self.data["sub_dir"].append(dir)
                image_path = '{0}/{1}.png'.format(instance_dir, frame["file_path"])
                self.data["image_paths"].append(image_path)
                self.data["mask_paths"].append(image_path.replace('image', 'mask'))
                self.data["img_name"].append(int(frame["file_path"].split('/')[-1]))

                # NOTE custom #######################
                if self.shape_frame:
                    self.data["shapes"].append(np.array(frame['shape_params']).astype(np.float32))
                if self.shape_scene:
                    self.data["shapes"].append(np.array(camera_dict['shape_params']).astype(np.float32))
                if self.normal:
                    self.data["normal_original_paths"].append(image_path.replace('image', 'normal_original'))
                    self.data["normal_rendering_paths"].append(image_path.replace('image', 'normal_rendering'))
                if self.mask_object:
                    self.data["mask_object_paths"].append(image_path.replace('image', 'mask_object'))
                if self.mask_source:
                    self.data["mask_source_paths"].append(image_path.replace('image', 'mask_source'))
                # if self.mask_original:
                #     self.data["mask_original_paths"].append(image_path.replace('image', 'mask_original'))
                #####################################
            
        # NOTE custom #######################
        if self.shape_frame or self.shape_scene:
            self.shape_params = None

        if self.shape_standard and self.mode != 'test':
            shape_standard_instance_dir = os.path.join(data_folder, subject_name, subject_name, self.shape_standard_scene)
            assert os.path.exists(shape_standard_instance_dir), "Data directory {} is empty".format(shape_standard_instance_dir)
            
            shape_standard_cam_file = '{0}/{1}'.format(shape_standard_instance_dir, json_name)
            with open(shape_standard_cam_file, 'r') as f:
                shape_standard_camera_dict = json.load(f)

            self.shape_params = torch.tensor(shape_standard_camera_dict['shape_params']).float().unsqueeze(0)
            print('[DEBUG] standard shape from {}'.format(shape_standard_cam_file))
        
        if self.mode == 'test':     # NOTE test를 할 때는 shape parameter를 따로 지정해주도록 하자.
            shape_standard_instance_dir = os.path.join(data_folder, subject_name, subject_name, self.shape_test)
            assert os.path.exists(shape_standard_instance_dir), "Data directory {} is empty".format(shape_standard_instance_dir)
            
            shape_standard_cam_file = '{0}/{1}'.format(shape_standard_instance_dir, json_name)
            with open(shape_standard_cam_file, 'r') as f:
                shape_standard_camera_dict = json.load(f)

            self.shape_params = torch.tensor(shape_standard_camera_dict['shape_params']).float().unsqueeze(0)
            print('[DEBUG] standard shape from {}'.format(shape_standard_cam_file))
        #####################################

        self.gt_dir = instance_dir
        # self.shape_params = torch.tensor(camera_dict['shape_params']).float().unsqueeze(0)
        focal_cxcy = camera_dict['intrinsics']

        # if isinstance(subsample, int) and subsample > 1:
        #     for k, v in self.data.items():
        #         self.data[k] = v[::subsample]
        # elif isinstance(subsample, list):
        #     if len(subsample) == 2:
        #         subsample = list(range(subsample[0], subsample[1]))
        #     for k, v in self.data.items():
        #         self.data[k] = [v[s] for s in subsample]

        # NOTE custom #######################
        def cumulative_sum(lst):
            result = []
            cumsum = 0
            for num in lst:
                cumsum += num
                result.append(cumsum)
            return result
        index_list = cumulative_sum(num_images)

        subsample_type = str(subsample_type)
        if subsample_type == 'frames':
            subsampled_frames = subsample
            if isinstance(subsampled_frames, list) and len(subsampled_frames) > 0:
                print('[DEBUG] subsampling the data by a list of frames: {}'.format(subsampled_frames))
                if len(num_images) != len(subsampled_frames):
                    raise ValueError('The number of subdirectories and the number of subsampled frames should be the same')
                subsample = [0]*len(num_images)
                for idx, num_img in enumerate(num_images):
                    subsample[idx] = num_img // subsampled_frames[idx]
                    if subsample[idx] == 0:
                        subsample[idx] = 1
            elif isinstance(subsampled_frames, int) and subsampled_frames > 1:
                print('[DEBUG] all of the frame should be the same as {}'.format(subsampled_frames))
                subsampled_frames = [subsampled_frames] * len(num_images)
                subsample = [0]*len(num_images)
                for idx, num_img in enumerate(num_images):
                    subsample[idx] = num_img // subsampled_frames[idx]
            else:
                raise ValueError('subsampled_frames should be a list of integers')
        elif subsample_type == 'ratio':
            print('[DEBUG] subsampling the data by a ratio of {}'.format(subsample))
            if isinstance(subsample, list):
                subsample = subsample*len(num_images)
            else:
                print('[INFO] no blacklist applied.')
        else:
            raise ValueError('subsample_type should be either "frames" or "ratio"')


        def pop_blacklist(data_list, blacklist):
            '''
            if data_list has the blacklist's item, then remove it and return
            '''
            sub_dir_name = data_list[0].split('/')[-4]
            black_file_list = blacklist[sub_dir_name]
            new_data_list = copy.deepcopy(data_list)
            removed_indices = []  # List to store the indices of removed items

            for idx, value in enumerate(data_list):
                if value.split('/')[-1] in black_file_list:
                    # print(value)
                    new_data_list.remove(value)
                    removed_indices.append(idx)
            return new_data_list, removed_indices
        
        def pop_blacklist_using_indices(data_list, removed_indices):
            '''
            if data_list has the blacklist's item, then remove it and return
            '''
            new_data_list = copy.deepcopy(data_list)
            for idx in reversed(removed_indices):
                new_data_list.pop(idx)
            return new_data_list


        if isinstance(subsample, int) and subsample > 1:
            for k, v in self.data.items():
                self.data[k] = v[::subsample]
            print('[DEBUG] subsampling the data by a factor of {} (int type)'.format(subsample))
            for i in zip(sub_dir, num_images):
                print('[DEBUG] sub directory: {} | frames: {}'.format(i[0], i[1]))
            print('[DEBUG] total frames: {} -> subsampled: {}'.format(index_list[-1], len(self.data['image_paths'])))

        # NOTE 이 코드는 blacklist를 적용해서 mask_obj나 mask_rgb에서 blacklist가 제대로 제거된다. train의 경우에만 해주고 test일 경우에는 그냥 다 해라.
        elif isinstance(subsample, list): #  and (self.mode != 'test'):
            print('[DEBUG] subsampling the data by a factor of {} (list type)'.format(subsample))
            if len(subsample) != len(sub_dir):
                raise ValueError('[ERROR] subsample list length should be equal to the number of subdirectories')
            
            print_info = True
            removed_indices_list = []
            for k, v in self.data.items():
                temp_list = []
                for i, s in enumerate(subsample):
                    assert subsample[i] != 0, 'Please except the train dataset: {}'.format(sub_dir[i])
                    if s == 1:
                        if i == 0:
                            sampled_list = v[:index_list[i]]
                            if self.blacklist is not None:
                                if k == 'image_paths':
                                    sampled_list, removed_indices = pop_blacklist(sampled_list, self.blacklist)
                                    removed_indices_list.append(removed_indices)
                                else:
                                    sampled_list = pop_blacklist_using_indices(sampled_list, removed_indices_list[i])
                            temp_list += sampled_list
                            if print_info:
                                print('[DEBUG] sub directory: {} | sampling rate: {} | frames: {}'.format(sub_dir[i], s, len(sampled_list)))
                                print('[DEBUG] first frame: {}, last frame: {}'.format(sampled_list[0], sampled_list[-1]))
                                if self.blacklist is not None and len(sampled_list) != len(v[:index_list[i]]):
                                    print('[DEBUG] before black list: {} -> after black list: {}'.format(len(v[:index_list[i]]), len(sampled_list)))
                        else:
                            sampled_list = v[index_list[i-1]:index_list[i]]
                            if self.blacklist is not None:
                                if k == 'image_paths':
                                    sampled_list, removed_indices = pop_blacklist(sampled_list, self.blacklist)
                                    removed_indices_list.append(removed_indices)
                                else:
                                    sampled_list = pop_blacklist_using_indices(sampled_list, removed_indices_list[i])
                            temp_list += sampled_list
                            if print_info:
                                print('[DEBUG] sub directory: {} | sampling rate: {} | frames: {}'.format(sub_dir[i], s, len(sampled_list)))
                                print('[DEBUG] first frame: {}, last frame: {}'.format(sampled_list[0], sampled_list[-1]))
                                if self.blacklist is not None and len(sampled_list) != len(v[index_list[i-1]:index_list[i]]):
                                    print('[DEBUG] before black list: {} -> after black list: {}'.format(len(v[index_list[i-1]:index_list[i]]), len(sampled_list)))
                    elif s > 1:
                        if i == 0:
                            sampled_list = v[:index_list[i]][::s]
                            if self.blacklist is not None:
                                if k == 'image_paths':
                                    sampled_list, removed_indices = pop_blacklist(sampled_list, self.blacklist)
                                    removed_indices_list.append(removed_indices)
                                else:
                                    sampled_list = pop_blacklist_using_indices(sampled_list, removed_indices_list[i])
                            temp_list += sampled_list
                            if print_info:
                                print('[DEBUG] sub directory: {} | sampling rate: {} | frames: {} -> {}'.format(sub_dir[i], s, len(v[:index_list[i]]), len(sampled_list)))
                                print('[DEBUG] first frame: {}, last frame: {}'.format(sampled_list[0], sampled_list[-1]))
                                if self.blacklist is not None and len(sampled_list) != len(v[:index_list[i]][::s]):
                                    print('[DEBUG] before black list: {} -> after black list: {}'.format(len(v[:index_list[i]][::s]), len(sampled_list)))
                        else:
                            sampled_list = v[index_list[i-1]:index_list[i]][::s]
                            if self.blacklist is not None:
                                if k == 'image_paths':
                                    sampled_list, removed_indices = pop_blacklist(sampled_list, self.blacklist)
                                    removed_indices_list.append(removed_indices)
                                else:
                                    sampled_list = pop_blacklist_using_indices(sampled_list, removed_indices_list[i])
                            temp_list += sampled_list
                            if print_info:
                                print('[DEBUG] sub directory: {} | sampling rate: {} | frames: {} -> {}'.format(sub_dir[i], s, len(v[index_list[i-1]:index_list[i]]), len(sampled_list)))
                                print('[DEBUG] first frame: {}, last frame: {}'.format(sampled_list[0], sampled_list[-1]))
                                if self.blacklist is not None and len(sampled_list) != len(v[index_list[i-1]:index_list[i]][::s]):
                                    print('[DEBUG] before black list: {} -> after black list: {}'.format(len(v[index_list[i-1]:index_list[i]][::s]), len(sampled_list)))
                    if k == 'image_paths':
                        assert sampled_list[0].split('/')[-4] == sampled_list[-1].split('/')[-4], 'sub directory name should be the same'
                    
                self.data[k] = temp_list
                print_info = False
            print('[DEBUG] total frames: {} -> subsampled: {}'.format(index_list[-1], len(self.data['image_paths'])))
        
        self.print_info = True
        #####################################

        self.data["expressions"] = torch.from_numpy(np.stack(self.data["expressions"], 0))
        self.data["flame_pose"] = torch.from_numpy(np.stack(self.data["flame_pose"], 0))
        self.data["world_mats"] = torch.from_numpy(np.stack(self.data["world_mats"], 0)).float()
        # NOTE custom #######################
        if self.shape_frame or self.shape_scene:
            self.data["shapes"] = torch.from_numpy(np.stack(self.data["shapes"], 0)).float()
        #####################################
        
        # construct intrinsic matrix
        intrinsics = np.zeros((4, 4))

        # from whatever camera convention to pytorch3d
        intrinsics[0, 0] = focal_cxcy[0] * 2
        intrinsics[1, 1] = focal_cxcy[1] * 2
        intrinsics[0, 2] = (focal_cxcy[2] * 2 - 1.0) * -1
        intrinsics[1, 2] = (focal_cxcy[3] * 2 - 1.0) * -1

        intrinsics[3, 2] = 1.
        intrinsics[2, 3] = 1.
        self.intrinsics = intrinsics

        if intrinsics[0, 0] < 0:
            intrinsics[:, 0] *= -1
            self.data["world_mats"][:, 0, :] *= -1
        self.data["world_mats"][:, :3, 2] *= -1
        self.data["world_mats"][:, 2, 3] *= -1

        if use_mean_expression:
            self.mean_expression = torch.mean(self.data["expressions"], 0, keepdim=True)
        else:
            self.mean_expression = torch.zeros_like(self.data["expressions"][[0], :])
        if use_var_expression:
            self.var_expression = torch.var(self.data["expressions"], 0, keepdim=True)
        else:
            self.var_expression = None

        self.intrinsics = torch.from_numpy(self.intrinsics).float()
        self.only_json = only_json

        images = []
        masks = []
        if load_images and not only_json:
            print("[INFO] Loading all images, this might take a while.")
            for idx in tqdm(range(len(self.data["image_paths"]))):
                rgb = torch.from_numpy(load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1).transpose(1,0)).float()
                object_mask = torch.from_numpy(load_mask(self.data["mask_paths"][idx], self.img_res).reshape(-1))
                if not self.use_background:         
                    if not hard_mask:           
                        rgb = rgb * object_mask.unsqueeze(1).float() + (1 - object_mask.unsqueeze(1).float())
                    else:
                        rgb = rgb * (object_mask.unsqueeze(1) > 0.5) + ~(object_mask.unsqueeze(1) > 0.5)
                images.append(rgb)
                masks.append(object_mask)

        self.data['images'] = images
        self.data['masks'] = masks

        # split_train_test = conf.get_bool('dataset.split_train_test')
        # if split_train_test:
        #     split_train_test_index = conf.get_int('dataset.split_train_test_index')
        #     split_train_test_dataset = conf.get_string('dataset.split_train_test_dataset')
        #     # start_idx, end_idx = find_indices_of_substring(self.data['image_paths'], split_train_test_dataset)
        #     separate_index = find_closest_index_by_name_and_number(self.data['image_paths'], split_train_test_dataset, '{}.png'.format(split_train_test_index)) # find_index_closest_to_target(self.data['img_name'], start_idx+split_train_test_index)

        #     for k in self.data.keys():
        #         if mode == 'train':
        #             self.data[k] = self.data[k][:separate_index]
        #         elif mode == 'test':
        #             self.data[k] = self.data[k][separate_index:]
        #     print('*'*50)
        #     print('[INFO] split the train and test dataset of the same dataset! (index: {})'.format(split_train_test_index))
        #     print('[INFO] {} dataset first frame: {}, last frame: {}'.format(mode, self.data['img_name'][0], self.data['img_name'][-1]))
        #     print('*'*50)
        #     assert conf.get_list('dataset.train.sub_dir')[0] == conf.get_list('dataset.test.sub_dir')[0], 'sub_dir should be the same'
        
        
        if self.mode == 'train':
            # NOTE train에 쓰인 sequence를 알아내기 위해.
            train_array = np.array(natsort.natsorted(list(set(natsort.natsorted(self.data['img_name']))))) # NOTE train에 쓰인 애들을 다 모아놓은 것이다. 이것들을 전부다 제외하면 이게 test가 된다.
            np.save(os.path.join(data_folder, subject_name, subject_name, 'train.npy'), train_array)
                

    def __len__(self):
        return len(self.data["image_paths"])

    def __getitem__(self, idx):
        sample = {
            "idx": torch.LongTensor([idx]),
            "img_name": torch.LongTensor([self.data["img_name"][idx]]),
            "sub_dir": self.data["sub_dir"][idx],
            "intrinsics": self.intrinsics,
            "expression": self.data["expressions"][idx],
            "flame_pose": self.data["flame_pose"][idx],
            "cam_pose": self.data["world_mats"][idx],
            # "shape": self.data["shapes"][idx] if self.shape_frame or self.shape_scene else self.shape_params.squeeze(),
            # "mask_object": torch.from_numpy(load_mask(self.data["mask_object_paths"][idx], self.img_res).reshape(-1)) if self.mode == 'train' and 'target' not in self.data["sub_dir"][idx] else None
            # "mask_object": torch.from_numpy(load_mask(self.data["mask_object_paths"][idx], self.img_res).reshape(-1)) if ('target' not in self.data["sub_dir"][idx]) and ('source' not in self.data["sub_dir"][idx]) else None
            "object_mask": torch.from_numpy(load_mask(self.data["mask_paths"][idx], self.img_res).reshape(-1))      # NOTE th+sh+db를 할 때 흰 부분 빈 공간을 채우기 위해 사용한다.
            }
        # NOTE 한개의 variable(frame, scene, standard)만 True이기 때문에 걱정ㄴㄴ
        if (self.shape_frame or self.shape_scene) and not self.mode == 'test':
            sample['shape'] = self.data["shapes"][idx]
        elif self.shape_standard or self.mode == 'test':
            sample['shape'] = self.shape_params.squeeze()
        # if self.mask_original:
        #     sample["mask_original"] = torch.from_numpy(load_mask(self.data["mask_original_paths"][idx], self.img_res).reshape(-1))

        if all(key not in self.data["sub_dir"][idx] for key in ["target", "source"]) and not self.mask_source and self.mask_object:
            sample["mask_object"] = torch.from_numpy(load_mask(self.data["mask_object_paths"][idx], self.img_res).reshape(-1))

        ground_truth = {}

        if not self.only_json:                          # NOTE True
            if not self.load_images:                    # NOTE True
                ground_truth["object_mask"] = torch.from_numpy(load_mask(self.data["mask_paths"][idx], self.img_res).reshape(-1))
                rgb = torch.from_numpy(load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1).transpose(1, 0)).float()
                if not self.use_background:             # NOTE True
                    if not self.hard_mask:              # NOTE True
                        ground_truth['rgb'] = rgb * ground_truth["object_mask"].unsqueeze(1).float() + (1 - ground_truth["object_mask"].unsqueeze(1).float())
                    else:
                        ground_truth['rgb'] = rgb * (ground_truth["object_mask"].unsqueeze(1) > 0.5) + ~(ground_truth["object_mask"].unsqueeze(1) > 0.5)
                else:
                    ground_truth['rgb'] = rgb
                
                # if self.normal and 'target' not in self.data["sub_dir"][idx]:
                if self.normal and all(key not in self.data["sub_dir"][idx] for key in ["target", "source"]) and self.mask_object:
                    ground_truth["mask_object"] = torch.from_numpy(load_mask(self.data["mask_object_paths"][idx], self.img_res).reshape(-1))

                    normal_original = torch.from_numpy(load_rgb(self.data["normal_original_paths"][idx], self.img_res).reshape(3, -1).transpose(1, 0)).float()
                    normal_rendering = torch.from_numpy(load_rgb(self.data["normal_rendering_paths"][idx], self.img_res).reshape(3, -1).transpose(1, 0)).float()

                    if not self.use_background:             # NOTE True
                        if not self.hard_mask:              # NOTE True
                            # NOTE mask_object를 제외한 영역에 대해서 normal loss가 적용되도록 구성. normal_original: [262144, 3], ground_truth["mask_object"]: [262144, ] -> normal_original_by_mask_object: [262144, 3]
                            normal_original_by_mask_object = normal_original * (1 - ground_truth["mask_object"].unsqueeze(1).float()) + ground_truth["mask_object"].unsqueeze(1).float()                        # 모자를 제외한 영역
                            ground_truth['normal_original'] = normal_original_by_mask_object * ground_truth["object_mask"].unsqueeze(1).float() + (1 - ground_truth["object_mask"].unsqueeze(1).float())        # foreground를 따는 역할

                            normal_rendering_by_mask_object = normal_rendering * ground_truth["mask_object"].unsqueeze(1).float() + (1 - ground_truth["mask_object"].unsqueeze(1).float())                      # 모자를 포함한 영역
                            ground_truth['normal_rendering'] = normal_rendering_by_mask_object * ground_truth["object_mask"].unsqueeze(1).float() + (1 - ground_truth["object_mask"].unsqueeze(1).float())      # foreground를 따는 역할

                            # ground_truth['normal_original'] = normal_original * ground_truth["object_mask"].unsqueeze(1).float() + (1 - ground_truth["object_mask"].unsqueeze(1).float())
                            # ground_truth['normal_rendering'] = normal_rendering * ground_truth["object_mask"].unsqueeze(1).float() + (1 - ground_truth["object_mask"].unsqueeze(1).float())
                        else:
                            normal_original_by_mask_object = normal_original * (~(ground_truth["mask_object"].unsqueeze(1) > 0.5)) + (ground_truth["mask_object"].unsqueeze(1) > 0.5)
                            ground_truth['normal_original'] = normal_original_by_mask_object * (ground_truth["object_mask"].unsqueeze(1) > 0.5) + ~(ground_truth["object_mask"].unsqueeze(1) > 0.5)

                            normal_rendering_by_mask_object = normal_rendering * (ground_truth["mask_object"].unsqueeze(1) > 0.5) + ~(ground_truth["mask_object"].unsqueeze(1) > 0.5)
                            ground_truth['normal_rendering'] = normal_rendering_by_mask_object * (ground_truth["object_mask"].unsqueeze(1) > 0.5) + ~(ground_truth["object_mask"].unsqueeze(1) > 0.5)
                    else:
                        ground_truth['normal_original'] = normal_original
                        ground_truth['normal_rendering'] = normal_rendering
                
                if self.segment:
                    ground_truth["mask_object"] = torch.from_numpy(load_mask(self.data["mask_object_paths"][idx], self.img_res).reshape(-1))
                    
                if self.mask_source:
                    sample["mask_source"] = torch.from_numpy(load_mask(self.data["mask_source_paths"][idx], self.img_res).reshape(-1)).unsqueeze(1) * ground_truth["object_mask"].unsqueeze(1).float()

                # if self.normal and 'target' in self.data["sub_dir"][idx]:
                #     normal_rendering = torch.from_numpy(load_rgb(self.data["normal_rendering_paths"][idx], self.img_res).reshape(3, -1).transpose(1, 0)).float()

                #     if not self.use_background:             # NOTE True
                #         if not self.hard_mask:              # NOTE True
                #             ground_truth['normal_rendering'] = normal_rendering * ground_truth["object_mask"].unsqueeze(1).float() + (1 - ground_truth["object_mask"].unsqueeze(1).float())
                #         else:
                #             ground_truth['normal_rendering'] = normal_rendering * (ground_truth["object_mask"].unsqueeze(1) > 0.5) + ~(ground_truth["object_mask"].unsqueeze(1) > 0.5)
                #     else:
                #         ground_truth['normal_rendering'] = normal_rendering
                
                # if self.segment:
                #     if self.data["sub_dir"][idx].split('_')[0] == 'hat':
                #         segment = torch.from_numpy(load_semantic(self.data["segment_paths"][idx], self.img_res).reshape(-1, self.img_res[0] * self.img_res[1]).transpose(1, 0)).float()
                #     else:
                #         segment = torch.from_numpy(load_semantic_no_hat(self.data["segment_paths"][idx], self.img_res).reshape(-1, self.img_res[0] * self.img_res[1]).transpose(1, 0)).float()
                #     ground_truth['segment'] = segment * ground_truth["object_mask"].unsqueeze(1).float()

            else:
                ground_truth = {
                    'rgb': self.data['images'][idx],
                    'object_mask': self.data['masks'][idx],
                }

        sample['object_mask'] = ground_truth['object_mask']
        
        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # this function is borrowed from https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py
        # get list of dictionaries and returns sample, ground_truth as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    try:
                        ret[k] = torch.stack([obj[k] for obj in entry])
                    except:
                        ret[k] = [obj[k] for obj in entry]
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

def load_rgb(path, img_res):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    img = cv2.resize(img, (int(img_res[0]), int(img_res[1])))
    img = img.transpose(2, 0, 1)
    return img


def load_mask(path, img_res):
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)

    alpha = cv2.resize(alpha, (int(img_res[0]), int(img_res[1])))
    object_mask = alpha / 255

    return object_mask

def load_semantic(path, img_res):
    img = imageio.imread(path, as_gray=True)
    h, w = img.shape
    semantics = np.zeros((h, w, 10))
    # identity
    # beard
    semantics[:, :, 2] = ((img == 7) + (img == 8)) >= 1                                                 # ears
    semantics[:, :, 3] = ((img == 2) + (img == 3)) >= 1                                                 # eyebrows
    semantics[:, :, 4] = ((img == 4) + (img == 5)) >= 1                                                 # eyes
    semantics[:, :, 5] = ((img == 17) + (img == 9)) >= 1                                                # hair & earrings
    semantics[:, :, 6] = (img == 18) >= 1                                                               # hat
    semantics[:, :, 7] = ((img == 11) + (img == 12) + (img == 13)) >= 1                                 # mouth
    semantics[:, :, 8] = (img == 10) >= 1                                                               # nose
    semantics[:, :, 9] = (img == 6) >= 1                                                                # eyeglasses

    semantics = cv2.resize(semantics, (int(img_res[0]), int(img_res[1])))
    semantics = semantics.transpose(2, 0, 1)
    return semantics

def load_semantic_no_hat(path, img_res):
    img = imageio.imread(path, as_gray=True)
    h, w = img.shape
    semantics = np.zeros((h, w, 10))
    # identity
    # beard
    semantics[:, :, 2] = ((img == 7) + (img == 8)) >= 1                                                 # ears
    semantics[:, :, 3] = ((img == 2) + (img == 3)) >= 1                                                 # eyebrows
    semantics[:, :, 4] = ((img == 4) + (img == 5)) >= 1                                                 # eyes
    semantics[:, :, 5] = ((img == 17) + (img == 9) + (img == 18)) >= 1                                                # hair & earrings & hat
    # semantics[:, :, 6] = (img == 18) >= 1                                                               # hat
    semantics[:, :, 7] = ((img == 11) + (img == 12) + (img == 13)) >= 1                                 # mouth
    semantics[:, :, 8] = (img == 10) >= 1                                                               # nose
    semantics[:, :, 9] = (img == 6) >= 1                                                                # eyeglasses

    # NOTE 이렇게 하면 
    semantics = cv2.resize(semantics, (int(img_res[0]), int(img_res[1])))
    semantics = semantics.transpose(2, 0, 1)                                                            # NOTE (512, 512, 10)에서 (10, 512, 512)로 만든 것.
    return semantics