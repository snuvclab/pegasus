import os
from datetime import datetime
import sys
from typing import Any, Dict
import torch
import time
import utils.general as utils
import utils.plots as plt
from functools import partial
from utils.rotation_converter import *
from kornia.geometry.conversions import camtoworld_to_worldtocam_Rt, worldtocam_to_camtoworld_Rt, Rt_to_matrix4x4
import numpy as np
import shutil
from tqdm import tqdm
import copy
from shutil import copyfile
import natsort
import warnings
warnings.filterwarnings("ignore")
import cv2
print = partial(print, flush=True)
from pyhocon import ConfigFactory
import gc


def has_files(directory_path):
    """
    Check if the given directory contains any files.
    
    Args:
    - directory_path (str): The path to the directory.
    
    Returns:
    - bool: True if there are files in the directory, False otherwise.
    """
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            return True
    return False

def int_to_tensor(n, device):
    # Convert the integer to its binary representation and strip the '0b' prefix
    binary_str = bin(n)[2:]
    
    # Pad the binary string with zeros to ensure it has a length of 4
    padded_str = binary_str.rjust(4, '0')
    
    # Convert the binary string into a list of integers
    binary_list = [int(bit) for bit in padded_str]
    
    # Convert the list of integers into a PyTorch tensor
    tensor = torch.tensor(binary_list, device=device)
    
    return tensor

def find_checkpoint_file(directory, epoch_number, ckpt_type):
    if ckpt_type == 'lightning':
        file_list = natsort.natsorted(os.listdir(directory))
    elif ckpt_type =='torch':
        file_list = natsort.natsorted(os.listdir(os.path.join(directory, 'ModelParameters')))

    last_file_exists = False
    
    if ckpt_type == 'lightning':
        if epoch_number == 'last':
            search_prefix = 'last'
            last_files = [] 
        else:
            search_prefix = f'epoch={epoch_number}-step='

    elif ckpt_type == 'torch':
        if epoch_number == 'latest':
            search_prefix = 'latest'
            last_files = []
        else:
            search_prefix = epoch_number

    for file_name in file_list:
        if ckpt_type == 'lightning':
            if file_name.startswith(search_prefix) and file_name.endswith('.ckpt'):
                if epoch_number == 'last':
                    last_files.append(file_name)
                    last_file_exists = True
                else:
                    return file_name
        elif ckpt_type == 'torch':
            if file_name.startswith(search_prefix) and file_name.endswith('.pth'):
                if epoch_number == 'latest':
                    last_files.append(file_name)
                    last_file_exists = True
                else:
                    return file_name

    if last_file_exists:
        last_files = natsort.natsorted(last_files)
        if len(last_files) > 1:
            return last_files[-2]
        else:
            return last_files[0]
    else:
        return None    


class PEGASUSModule(pl.LightningModule):
    '''
    NOTE
    * previous name: SceneLatentCategoryMultiCompositionModelModule
    * TrainRunnerSceneLatentCategoryMultiComposition을 lightning으로 바꾼 코드이다. 더 빠르게 돌리기 위함이다.
    '''
    def __init__(self, **kwargs):
        super().__init__()
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        
        ############################### For Train Settings ################################
        self.save_hyperparameters()         # NOTE lightning을 위한 부분
        self.automatic_optimization=False   # NOTE lightning을 위한 부분

        self.conf = kwargs['conf']
        self.max_batch = self.conf.get_int('train.max_batch')                                                                                
        self.batch_size = min(int(self.conf.get_int('train.max_points_training') / 400), self.max_batch)
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.optimize_expression = self.conf.get_bool('train.optimize_expression')
        self.optimize_pose = self.conf.get_bool('train.optimize_camera')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        # NOTE wandb관련 두줄은 exp_runner_lightning.py로 옮겨놓았다.

        # NOTE custom #################
        self.category_specific = self.conf.get_bool('dataset.category_specific')
        self.num_category = self.conf.get_int('dataset.num_category')
        self.category_latent_dim = self.conf.get_int('dataset.category_latent_dim')
        self.optimize_latent_code = self.conf.get_bool('train.optimize_latent_code')
        self.optimize_scene_latent_code = self.conf.get_bool('train.optimize_scene_latent_code')
        self.target_training = self.conf.get_bool('train.target_training', default=False)

        if self.optimize_latent_code:
            self.latent_dim = 32
        else:
            self.latent_dim = 0
        if self.optimize_scene_latent_code:
            if not self.category_specific:
                self.scene_latent_dim = 32
            else:
                self.scene_latent_dim = 32 - self.category_latent_dim
        else:
            self.scene_latent_dim = 0
        
        if self.category_specific:
            self.category_dict = {
                'source': 0,
                'beard': 1,
                'ears': 2,
                'eyebrows': 3,
                'eyes': 4,
                'hair': 5,
                'hat': 6,
                'mouth': 7,
                'nose': 8,
                'eyeglasses': 9
            }
            assert len(self.category_dict) == self.num_category
            self.source_category_dict = {
                'ears': 2,
                'eyebrows': 3,
                'eyes': 4,
                'hair': 5,
                'mouth': 7,
                'nose': 8
            }

            # if self.target_training:
            self.target_category_dict = {
                'ears': 2,
                'eyebrows': 3,
                'eyes': 4,
                'hair': 5,
                'mouth': 7,
                'nose': 8
            }

        self.print_info = True
        dataset_train_subdir = self.conf.get_list('dataset.train.sub_dir')
        # self.max_points = self.conf.get_int('model.point_cloud.max_points')
        # self.min_radius = self.conf.get_float('model.point_cloud.min_radius')
        ###############################

        # NOTE customize in self.optimize_inputs
        self.optimize_inputs = self.optimize_expression or self.optimize_pose or self.optimize_latent_code or self.optimize_scene_latent_code
        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(dataset_train_subdir)

        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        self.checkpoints_path = os.path.join(self.train_dir, 'checkpoints')

        self.voting_save_path = os.path.join(self.train_dir, 'mask_indices')
        self.voting_table_save_path = os.path.join(self.train_dir, 'voting_table')
        
        # NOTE directory 만드는 내용, checkpoint 저장을 위한 폴더를 만드는 내용 싹다 삭제.
        is_continue = False
        # ckpt_filenames = natsort.natsorted(os.listdir(self.checkpoints_path))
        # if len(ckpt_filenames) > 0:
        #     is_continue = True
        # self.ckpt_file_name = ckpt_filenames[-1] if is_continue else None
        if kwargs['path_ckpt'] is not None:
            is_continue = True
            self.ckpt_file_name = os.path.basename(kwargs['path_ckpt'])
        if is_continue and kwargs['path_ckpt'] is None:
            kwargs['path_ckpt'] = os.path.join(self.checkpoints_path, self.ckpt_file_name)

        self.file_backup(kwargs['path_conf'])

        self.use_background = self.conf.get_bool('dataset.use_background', default=False)

        self.train_dataset = kwargs['train_dataset']                    # NOTE train_dataset은 datamodule로 불러온다.
        self.test_dataset = kwargs['test_dataset']                      # NOTE test_dataset은 datamodule로 불러온다. 원래는 plot_dataset이었음.
        
        self.start_epoch = 0

        self.train_from_scratch = False

        if is_continue:
            print('[INFO] continue training...')
            pcd_init = {}
            saved_model_state = torch.load(os.path.join(self.checkpoints_path, self.ckpt_file_name), 
                                           map_location=self.device)

            pcd_init['n_init_points'] = saved_model_state["state_dict"]['model.pc.points'].shape[0]
            try:
                pcd_init['init_radius'] = saved_model_state["state_dict"]['model.pc.radius'].item()
            except:  # NOTE 이전 방식.
                pcd_init['init_radius'] = saved_model_state["state_dict"]['model.radius'].item()

            self.start_epoch = saved_model_state['epoch']
        else:
            meta_learning = self.conf.get_bool('train.meta_learning.load_from_meta_learning')
            meta_learning_path = self.conf.get_string('train.meta_learning.path')
            meta_learning_type = self.conf.get_string('train.meta_learning.type')
            meta_learning_filename = find_checkpoint_file(meta_learning_path, self.conf.get_string('train.meta_learning.epoch_num'), meta_learning_type)

            if meta_learning:
                print('[INFO] meta learning from {}'.format(meta_learning_path))
                pcd_init = {}
                pretrain_path = os.path.join(meta_learning_path)
                if meta_learning_type == 'torch':
                    saved_model_state = torch.load(os.path.join(pretrain_path, 'ModelParameters', meta_learning_filename),
                                                   map_location=self.device)
                    saved_input_state = torch.load(os.path.join(pretrain_path, 'InputParameters', meta_learning_filename),
                                                   map_location=self.device)
                    pcd_init['n_init_points'] = saved_model_state["model_state_dict"]['pc.points'].shape[0]
                    pcd_init['init_radius'] = saved_model_state['radius']
                elif meta_learning_type == 'lightning':
                    saved_model_state = torch.load(os.path.join(pretrain_path, meta_learning_filename),
                                                   map_location=self.device)
                    pcd_init['n_init_points'] = saved_model_state["state_dict"]['model.pc.points'].shape[0]
                    pcd_init['init_radius'] = saved_model_state["state_dict"]['model.pc.radius'].item()
                else:
                    raise NotImplementedError
                
                self.start_epoch = saved_model_state['epoch']
            else:
                print('[INFO] training from scratch...')
                pcd_init = None
                self.train_from_scratch = True

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf, 
                                                                                shape_params=self.train_dataset.shape_params,
                                                                                img_res=self.train_dataset.img_res,
                                                                                canonical_expression=self.train_dataset.mean_expression,
                                                                                canonical_pose=self.conf.get_float('dataset.canonical_pose', default=0.2),
                                                                                use_background=self.use_background,
                                                                                checkpoint_path=kwargs['path_ckpt'],
                                                                                pcd_init=pcd_init)

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'), 
                                                                              var_expression=self.train_dataset.var_expression, 
                                                                              optimize_scene_latent_code=self.optimize_scene_latent_code)
        
        self.lr = self.conf.get_float('train.learning_rate')
        # NOTE optimizer 관련 내용은 따로 method로 분리.
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        # NOTE scheduler 관련 내용도 따로 method로 분리.
        self.upsample_freq = self.conf.get_int('train.upsample_freq', default=5)
        # settings for camera optimization
        if self.optimize_inputs:
            num_training_frames = len(self.train_dataset)
            param = []
            if self.optimize_expression:
                init_expression = torch.cat((self.train_dataset.data["expressions"], torch.randn(self.train_dataset.data["expressions"].shape[0], max(self.model.deformer_network.num_exp - 50, 0)).float()), dim=1)
                self.expression = torch.nn.Embedding(num_training_frames, self.model.deformer_network.num_exp, _weight=init_expression)
                param += list(self.expression.parameters())

            if self.optimize_pose:
                self.flame_pose = torch.nn.Embedding(num_training_frames, 15, _weight=self.train_dataset.data["flame_pose"])
                self.camera_pose = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.data["world_mats"][:, :3, 3])
                param += list(self.flame_pose.parameters()) + list(self.camera_pose.parameters())
            
            # NOTE hyunsoo added
            if self.optimize_latent_code:
                self.latent_codes = torch.nn.Embedding(num_training_frames, self.latent_dim)
                torch.nn.init.uniform_(
                    self.latent_codes.weight.data,
                    0.0,
                    1.0,
                )
                param += list(self.latent_codes.parameters())
                print('[DEBUG] Latent code is used. The latent dimension is {0}x{1}.'.format(num_training_frames, self.latent_dim))

            if self.optimize_scene_latent_code:
                if self.target_training:
                    length_target_subdir = 1
                    self.target_scene_latent_codes = torch.nn.Embedding(length_target_subdir, self.scene_latent_dim)
                    torch.nn.init.normal_(
                        self.target_scene_latent_codes.weight.data,
                        0.0,
                        0.25,
                    )
                    param += list(self.target_scene_latent_codes.parameters())
                    
                    self.target_scene_latent_category_codes = torch.nn.Embedding(len(self.target_category_dict.keys()) * length_target_subdir, self.scene_latent_dim)
                    torch.nn.init.normal_(
                        self.target_scene_latent_category_codes.weight.data,
                        0.0,
                        0.25,
                    )
                    param += list(self.target_scene_latent_category_codes.parameters())

                else:
                    self.dataset_train_subdir = dataset_train_subdir
                    self.scene_latent_codes = torch.nn.Embedding(len(self.dataset_train_subdir), self.scene_latent_dim)
                    torch.nn.init.normal_(
                        self.scene_latent_codes.weight.data,
                        0.0,
                        0.25,
                    )
                    param += list(self.scene_latent_codes.parameters())
                    print('[DEBUG] Scene latent code is used. The latent dimension is {0}x{1}.'.format(len(self.dataset_train_subdir), self.scene_latent_dim))

                if self.category_specific:
                    self.zero_latent_codes = torch.nn.Embedding(1, self.scene_latent_dim)
                    torch.nn.init.normal_(
                        self.zero_latent_codes.weight.data,
                        0.0,
                        0.25,
                    )
                    param += list(self.zero_latent_codes.parameters())

                    if not self.target_training:
                        self.source_scene_latent_codes = torch.nn.Embedding(1, self.scene_latent_dim)
                        torch.nn.init.normal_(
                            self.source_scene_latent_codes.weight.data,
                            0.0,
                            0.25,
                        )
                        param += list(self.source_scene_latent_codes.parameters())
                        
                        self.source_scene_latent_category_codes = torch.nn.Embedding(len(self.source_category_dict.keys()), self.scene_latent_dim)
                        torch.nn.init.normal_(
                            self.source_scene_latent_category_codes.weight.data,
                            0.0,
                            0.25,
                        )
                        param += list(self.source_scene_latent_category_codes.parameters())

            self.input_params = param
            # self.optimizer_cam = torch.optim.SparseAdam(param, self.conf.get_float('train.learning_rate_cam'))            # NOTE lightning에서는 안되는걸로 알아서 cam도 함께 한번에 한다.

        # self.masked_point_cloud_indices = {}

        self.voting_threshold = self.conf.get_float('train.voting_threshold')

        # if os.path.exists(self.voting_save_path):
        #     filename = natsort.natsorted(os.listdir(self.voting_save_path))[-1]
        #     self.masked_point_cloud_indices = torch.load(os.path.join(self.voting_save_path, filename), map_location=self.device)
        # else:
        #     os.makedirs(self.voting_save_path, exist_ok=True)
        
        # NOTE 불러오는건 무조건 is_eval일때만 가능하다!
        if os.path.exists(self.voting_table_save_path) and kwargs['is_eval']:
            if has_files(self.voting_table_save_path):
                # current_epoch = self.ckpt_file_name.split('-')[0].split('=')[1]
                for filename in natsort.natsorted(os.listdir(self.voting_table_save_path)):
                    if str(self.start_epoch) == filename.split('-')[0].split('=')[1]:
                        voting_table_ckpt_filename = filename
                        break         
                print('[INFO] voting table is loaded from {}'.format(os.path.join(self.voting_table_save_path, voting_table_ckpt_filename)))       
                self.model.voting_table = torch.load(os.path.join(self.voting_table_save_path, voting_table_ckpt_filename), map_location=self.device)
        
        os.makedirs(self.voting_table_save_path, exist_ok=True)

        self.masked_point_cloud_indices = {}

        if self.target_training:
            self.dataset_target_train_subdir = self.conf.get_list('dataset.train.sub_dir')
        # NOTE voting table을 이용해서 masked point cloud indices를 만들 때 쓰이는 코드이다. 즉, voting table이 초기화된 상태이면 의미가 없다.
        if os.path.exists(self.voting_table_save_path) and kwargs['is_eval']:
            for i in range(self.model.voting_table.shape[0]):
                row_sum = self.model.voting_table[i, :, :].sum(dim=-1)
                mask = (row_sum > self.voting_threshold).float()
                count = torch.sum(mask == 1).item()
                if self.target_training:
                    sub_dir = self.conf.get_list('dataset.train.sub_dir')[i]
                else:
                    sub_dir = self.dataset_train_subdir[i]
                self.masked_point_cloud_indices[sub_dir] = mask
                print('[INFO] sub_dir {} | count {}'.format(sub_dir, count))

        if not is_continue:
            if meta_learning:
                if meta_learning_type == 'torch':
                    # try:
                    self.model.load_state_dict(saved_model_state["model_state_dict"], strict=False)
                    # except:
                    #     # Get the state_dict from the checkpoint
                    #     checkpoint_state_dict = saved_model_state["model_state_dict"]

                    #     # Get the state_dict of the current model
                    #     model_state_dict = self.model.state_dict()

                    #     # For each key in the model's state dict
                    #     for key in model_state_dict.keys():
                    #         # If the key exists in the checkpoint state dict
                    #         if key in checkpoint_state_dict:
                    #             # If sizes match, just copy
                    #             if model_state_dict[key].size() == checkpoint_state_dict[key].size():
                    #                 model_state_dict[key] = checkpoint_state_dict[key]
                    #             # If sizes don't match, copy what you can and fill the rest with zeros
                    #             else:
                    #                 shape = model_state_dict[key].shape
                    #                 tmp = torch.zeros(shape)
                    #                 print('[WARN] miss match the key {}'.format(key))

                    #                 if len(shape) == 1:
                    #                     slice_size = min(checkpoint_state_dict[key].shape[0], shape[0])
                    #                     tmp[:slice_size] = checkpoint_state_dict[key][:slice_size]
                    #                 elif len(shape) == 2:
                    #                     slice_size_0 = min(checkpoint_state_dict[key].shape[0], shape[0])
                    #                     slice_size_1 = min(checkpoint_state_dict[key].shape[1], shape[1])
                    #                     tmp[:slice_size_0, :slice_size_1] = checkpoint_state_dict[key][:slice_size_0, :slice_size_1]
                
                
                    #                 model_state_dict[key] = tmp

                        # Load the modified state dict back into the model
                        # self.model.load_state_dict(model_state_dict)
                    self.zero_latent_codes.load_state_dict(saved_input_state['zero_latent_codes_state_dict'])
                    
                    if not self.target_training:
                        self.scene_latent_codes.load_state_dict(saved_input_state['scene_latent_codes_state_dict'])
                        self.source_scene_latent_codes.load_state_dict(saved_input_state['source_scene_latent_codes_state_dict'])
                        self.source_scene_latent_category_codes.load_state_dict(saved_input_state['source_scene_latent_category_codes_state_dict'])
                elif meta_learning_type == 'lightning':
                    new_state_dict = {}
                    for key, value in saved_model_state['state_dict'].items():
                        if 'model.' in key:
                            new_key = key.replace('model.', '')
                            new_state_dict[new_key] = value
                    self.model.load_state_dict(new_state_dict, strict=False)
                    self.scene_latent_codes.weight.data.copy_(saved_model_state["state_dict"]['scene_latent_codes.weight'])
                    self.source_scene_latent_codes.weight.data.copy_(saved_model_state["state_dict"]['source_scene_latent_codes.weight'])
                    self.source_scene_latent_category_codes.weight.data.copy_(saved_model_state["state_dict"]['source_scene_latent_category_codes.weight'])
                    self.zero_latent_codes.weight.data.copy_(saved_model_state["state_dict"]['zero_latent_codes.weight'])
            else:
                print('[INFO] No pretrain model is used.')

        # NOTE checkpoint 들고오는 파트는 다 분산되어있다.
        # 2) n_points는 point_cloud.py에서 들고온다.
        # 3) batch_size는 어차피 1로 고정이다.
        # 4) self.model.pc.init은 point_cloud.py에서 해준다.
        # 5) radius는 pointavatar.py init에 구현했다.

        print('[INFO] shell command : {0}'.format(' '.join(sys.argv)))

        # self.n_batches = kwargs['n_batches']          # NOTE datamodule에서 구현했다.
        self.img_res = self.test_dataset.img_res
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.save_freq = self.conf.get_int('train.save_freq', default=1)

        self.GT_lbs_milestones = self.conf.get_list('train.GT_lbs_milestones', default=[])
        self.GT_lbs_factor = self.conf.get_float('train.GT_lbs_factor', default=0.5)
        for acc in self.GT_lbs_milestones:
            if self.start_epoch > acc:
                self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor
        # if len(self.GT_lbs_milestones) > 0 and self.start_epoch >= self.GT_lbs_milestones[-1]:
        #    self.loss.lbs_weight = 0.

        # NOTE 옮겨왔음.
        self.start_time = torch.cuda.Event(enable_timing=True)
        self.end_time = torch.cuda.Event(enable_timing=True)

        # NOTE ablation #########################################
        self.enable_prune = self.conf.get_bool('train.enable_prune')
        self.enable_upsample = self.conf.get_bool('train.enable_upsample')
        self.acc_loss = {}
        # self.total_loss = 0

        self.datamodule = kwargs['datamodule']
        self.print_freq = self.conf.get_int('train.print_freq')

        # if not self.training:
        def extract_category(s):
            return s.split('_')[0]
        
        # self.conf.put('dataset.test.subsample', 1)
        # self.conf.put('dataset.test.load_images', False)
        self.dataset_test_subdir = self.conf.get_list('dataset.test.sub_dir')

        self.test_default_rendering = self.conf.get_bool('test.default_rendering')
        self.test_mask_rendering = self.conf.get_bool('test.mask_rendering')
        self.test_target_default_rendering = self.conf.get_bool('test.target_default_rendering', default=False)
        self.test_scene_latent_interpolation = self.conf.get_bool('test.scene_latent_interpolation')
        self.test_scene_latent_interpolation_category = self.conf.get_string('test.scene_latent_interpolation_category')
        self.test_multi_composition = self.conf.get_bool('test.multi_composition')
        self.test_multi_composition_list = sorted(self.conf.get_list('test.multi_composition_list'), key=lambda x: self.category_dict[extract_category(x)])

        # self.test_target_finetuning = self.conf.get_bool('test.target_finetuning')
        # NOTE test_target_finetuning_list는 target human에게 composition하고 싶은 latent를 선택하는 것이다.
        # self.test_target_finetuning_subdir = self.conf.get_string('test.target_finetuning_subdir')

        # if self.test_target_finetuning:
        #     result_segment_masks_save_path = os.path.join(self.train_dir, 'segment_mask_result.pth')
        #     self.segment_masks = torch.load(result_segment_masks_save_path,
        #                                     map_location=self.device)
            
        # self.test_db_sub_dir = self.conf.get_list('test.db_sub_dir')
        
        self.test_interpolation_step_size = 10
        try:
            self.test_epoch = saved_model_state['epoch']
            print('[INFO] Test loading checkpoint from {0} epoch'.format(self.test_epoch))
        except:
            self.test_epoch = 0

        # if kwargs['path_ckpt'] is not None and os.path.exists(kwargs['path_ckpt']):
            # print(str(kwargs['path_ckpt']))
            # saved_model_state = torch.load(str(kwargs['path_ckpt']), map_location=lambda storage, loc: storage)
            # self.test_epoch = saved_model_state['epoch']
            # print('[INFO] Loading checkpoint from {0} epoch'.format(self.test_epoch))
        
        self.test_target_human_chamfer_blending = self.conf.get_bool('test.target_human.chamfer_blending', default=False)
        self.test_target_human_blending = self.conf.get_bool('test.target_human.blending', default=False)
        self.test_target_interpolation = self.conf.get_bool('test.target_interpolation', default=False)
        # if self.test_target_human_chamfer_blending:
        #     # NOTE target human이 lightning으로 만든 경우에 불러오는 코드이다.
        #     self.test_target_human_path = self.conf.get_string('test.target_human.path')
        #     self.test_target_human_epoch_num = self.conf.get_string('test.target_human.epoch_num')
        #     self.test_target_human_conf = ConfigFactory.parse_file(self.conf.get_string('test.target_human.conf'))
        #     self.test_target_voting_table_path = self.conf.get_string('test.target_human.voting_table_path')
        #     self.test_target_human_voting_threshold = self.conf.get_float('test.target_human.voting_threshold')

        #     target_human_path = os.path.join(self.test_target_human_path, find_checkpoint_file(self.test_target_human_path, self.test_target_human_epoch_num, 'lightning'))
        #     saved_model_state = torch.load(target_human_path, map_location=self.device)

        #     th_pcd_init = {}
        #     th_pcd_init['n_init_points'] = saved_model_state["state_dict"]['model.pc.points'].shape[0]
        #     th_pcd_init['init_radius'] = saved_model_state["state_dict"]['model.pc.radius']

        #     self.test_target_human_model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.test_target_human_conf, 
        #                                                                                                         shape_params=self.train_dataset.shape_params,       # NOTE 어차피 의미없다.
        #                                                                                                         img_res=self.train_dataset.img_res,
        #                                                                                                         canonical_expression=self.train_dataset.mean_expression,
        #                                                                                                         canonical_pose=self.conf.get_float('dataset.canonical_pose', default=0.2),
        #                                                                                                         use_background=self.use_background,
        #                                                                                                         checkpoint_path=None,
        #                                                                                                         pcd_init=th_pcd_init)
        #     new_state_dict = {}
        #     for key, value in saved_model_state['state_dict'].items():
        #         if 'model.' in key:
        #             new_key = key.replace('model.', '')
        #             new_state_dict[new_key] = value
        #     self.test_target_human_model.load_state_dict(new_state_dict, strict=False)

        #     self.test_target_scene_latent_codes = torch.nn.Embedding(1, self.scene_latent_dim)
        #     self.test_target_scene_latent_category_codes = torch.nn.Embedding(len(self.target_category_dict.keys()), self.scene_latent_dim)
        #     self.test_target_zero_latent_codes = torch.nn.Embedding(1, self.scene_latent_dim)
            
        #     self.test_target_scene_latent_codes.weight.data.copy_(saved_model_state["state_dict"]['target_scene_latent_codes.weight'])
        #     self.test_target_scene_latent_category_codes.weight.data.copy_(saved_model_state["state_dict"]['target_scene_latent_category_codes.weight'])
        #     self.test_target_zero_latent_codes.weight.data.copy_(saved_model_state["state_dict"]['zero_latent_codes.weight'])

        #     for filename in natsort.natsorted(os.listdir(self.test_target_voting_table_path)):
        #         if self.test_target_human_epoch_num == filename.split('-')[0].split('=')[1]:
        #             test_target_human_voting_table_ckpt_filename = filename
        #             break         

        #     print('[INFO] voting table is loaded from {}'.format(os.path.join(self.test_target_voting_table_path, test_target_human_voting_table_ckpt_filename)))       
        #     self.test_target_human_model.voting_table = torch.load(os.path.join(self.test_target_voting_table_path, test_target_human_voting_table_ckpt_filename), map_location=self.device)

        #     self.test_target_human_masked_point_cloud_indices = {}

        #     for i in range(self.test_target_human_model.voting_table.shape[0]):
        #         row_sum = self.test_target_human_model.voting_table[i, :, :].sum(dim=-1)
        #         mask = (row_sum > self.test_target_human_voting_threshold).float()
        #         count = torch.sum(mask == 1).item()
        #         test_target_human_sub_dir = self.test_target_human_conf.get_list('dataset.train.sub_dir')[i]
        #         self.test_target_human_masked_point_cloud_indices[test_target_human_sub_dir] = mask
        #         print('[INFO] sub_dir {} | count {}'.format(test_target_human_sub_dir, count))

        #     self.model_free_memory = copy.copy(self.model)
        #     self.test_target_human_model_free_memory = copy.copy(self.test_target_human_model)
        
        if self.test_target_human_blending:
            # NOTE target human이 torch로 만든 경우에 불러오는 코드이다.
            self.test_target_human_path = self.conf.get_string('test.target_human.path')
            self.test_target_human_epoch_num = self.conf.get_string('test.target_human.epoch_num')
            self.test_target_human_conf = ConfigFactory.parse_file(self.conf.get_string('test.target_human.conf'))
            # self.test_target_voting_table_path = self.conf.get_string('test.target_human.voting_table_path')
            # self.test_target_human_voting_threshold = self.conf.get_float('test.target_human.voting_threshold')

            target_human_path = os.path.join(self.test_target_human_path, 'ModelParameters', '{}.pth'.format(self.test_target_human_epoch_num))
            saved_model_state = torch.load(target_human_path, map_location=self.device)

            th_pcd_init = {}
            th_pcd_init['n_init_points'] = saved_model_state["model_state_dict"]['pc.points'].shape[0]
            th_pcd_init['init_radius'] = saved_model_state['radius']

            self.test_target_human_model = utils.get_class(self.test_target_human_conf.get_string('train.model_class'))(conf=self.test_target_human_conf, 
                                                                                                                shape_params=self.train_dataset.shape_params,       # NOTE 어차피 의미없다.
                                                                                                                img_res=self.train_dataset.img_res,
                                                                                                                canonical_expression=self.train_dataset.mean_expression,
                                                                                                                canonical_pose=self.conf.get_float('dataset.canonical_pose', default=0.2),
                                                                                                                use_background=self.use_background,
                                                                                                                checkpoint_path=None,
                                                                                                                pcd_init=th_pcd_init)
            self.test_target_human_model.load_state_dict(saved_model_state["model_state_dict"], strict=False)

            saved_input_state = torch.load(os.path.join(self.test_target_human_path, 'InputParameters', '{}.pth'.format(self.test_target_human_epoch_num)), map_location=self.device)
            self.test_target_scene_latent_codes = torch.nn.Embedding(1, self.scene_latent_dim, sparse=False)
            self.test_target_scene_latent_codes.load_state_dict(saved_input_state["target_scene_latent_codes_state_dict"])
            self.test_target_scene_latent_category_codes = torch.nn.Embedding(len(self.target_category_dict.keys()) * 1, self.scene_latent_dim)
            self.test_target_scene_latent_category_codes.load_state_dict(saved_input_state['target_scene_latent_category_codes_state_dict'])
            self.test_target_zero_latent_codes = torch.nn.Embedding(1, self.scene_latent_dim)
            self.test_target_zero_latent_codes.load_state_dict(saved_input_state['zero_latent_codes_state_dict'])

            # for filename in natsort.natsorted(os.listdir(self.test_target_voting_table_path)):
            #     if self.test_target_human_epoch_num == filename.split('-')[0].split('=')[1]:
            #         test_target_human_voting_table_ckpt_filename = filename
            #         break         
            # print('[INFO] voting table is loaded from {}'.format(os.path.join(self.test_target_voting_table_path, test_target_human_voting_table_ckpt_filename)))       
            # self.test_target_human_model.voting_table = torch.load(os.path.join(self.test_target_voting_table_path, test_target_human_voting_table_ckpt_filename), map_location=self.device)

            self.test_target_human_masked_point_cloud_indices = {}

            # for i in range(self.test_target_human_model.voting_table.shape[0]):
            #     row_sum = self.test_target_human_model.voting_table[i, :, :].sum(dim=-1)
            #     mask = (row_sum > self.test_target_human_voting_threshold).float()
            #     count = torch.sum(mask == 1).item()
            #     test_target_human_sub_dir = self.test_target_human_conf.get_list('dataset.train.sub_dir')[i]
            #     self.test_target_human_masked_point_cloud_indices[test_target_human_sub_dir] = mask
            #     print('[INFO] sub_dir {} | count {}'.format(test_target_human_sub_dir, count))

            self.model_free_memory = copy.copy(self.model)
            self.test_target_human_model_free_memory = copy.copy(self.test_target_human_model)
        #########################################################

    def configure_optimizers(self):
        params = list(self.model.parameters())
        if self.optimize_inputs:
            params += list(self.input_params)
        grouped_parameters = [
            {"params": params},
        ]
        optimizer = torch.optim.Adam(grouped_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.sched_milestones, gamma=self.sched_factor)
        self.last_lr_scheduler = scheduler
        return [optimizer], [scheduler]

    # def update_optimizer_params(self):
    #     params = list(self.model.parameters())
    #     if self.optimize_inputs:
    #         params += list(self.input_params)
    #     # grouped_parameters = [{"params": params}]
    #     # optimizer = torch.optim.Adam(grouped_parameters, lr=self.lr)
    #     # self.optimizers = [optimizer]
    #     self.optimizers().param_groups = []
    #     self.optimizers().add_param_group({'params': params})
    
    # def on_load_checkpoint(self, checkpoint: Dict) -> None:
    #     checkpoint.pop("optimizer_states", None)
    
    def upsample_points(self):
        current_radius = self.model.raster_settings.radius
        points = self.model.pc.points.data
        if self.model.pc.points.shape[0] <= self.model.pc.max_points / 2:           # 2배씩 증가
            noise = (torch.rand(*points.shape, device=points.device) - 0.5) * current_radius
            new_points = noise + points
        else:
            new_points_shape = (self.model.pc.max_points - points.shape[0], points.shape[1])
            noise = (torch.rand(new_points_shape, device=points.device) - 0.5) * current_radius     # NOTE cuda 삭제

            indices = torch.randperm(points.size(0))
            selected_points = points[indices[:new_points_shape[0]]]
            new_points = noise + selected_points
        self.model.pc.upsample_points(new_points)                                           # NOTE 기존에서 new_points를 합침.
        if self.model.pc.min_radius < current_radius:
            new_radius = 0.75 * current_radius
            self.model.pc.register_radius(new_radius)
            self.model.raster_settings.radius = new_radius
        print("***************************************************")
        print("old radius: {}, new radius: {}".format(current_radius, self.model.raster_settings.radius))
        # print("old points: {}, new points: {}".format(self.model.pc.points.data.shape[0]/2, self.model.pc.points.data.shape[0]))      # NOTE original code
        print("old points: {}, new points: {}".format(points.shape[0], self.model.pc.points.data.shape[0]))                             # NOTE custom code. 더 정확하게 표시하기 위해서. 정말 upsample이 된게 맞나.     
        print("***************************************************") 
        # NOTE 원래 코드에서는 여기서 self.optimizer를 다시 선언하지만, 여기서는 그럴 수 없다.
        # self.optimizer = torch.optim.Adam([
        #     {'params': list(self.model.parameters())},
        # ], lr=self.lr)
        
    
    def file_backup(self, path_conf):
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y%m%d_%H%M')

        # NOTE backup python file. 기존 코드
        dir_lis = ['./model', './scripts', './utils', './flame', './datasets']
        recording_path = os.path.join(self.train_dir, '{}_recording'.format(formatted_time))
        os.makedirs(recording_path, exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(recording_path, dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
        
        # NOTE backup config file. 추가하였음
        source_path = os.path.join(os.getcwd(), path_conf)
        filename = "{}_{}".format(formatted_time, source_path.split("/")[-1])
        destination_path = os.path.join(self.train_dir, filename)
        shutil.copy(source_path, destination_path)

    def voting_based_tensor(self, tensors, frequency_num=7):
        # Create a dictionary to count the number of tensors each number appears in
        count_dict = {}
        # Iterate through all tensors and count the numbers
        for tensor in tensors:
            unique_nums = torch.unique(tensor).tolist()
            for num in unique_nums:
                count_dict[num] = count_dict.get(num, 0) + 1
        # Create a list with numbers that appear in 7 or more tensors
        result_numbers = [num for num, count in count_dict.items() if count >= frequency_num]
        # Convert the list to a tensor
        result_tensor = torch.tensor(result_numbers)
        
        return result_tensor

    def on_train_start(self):
        print('\n**************** Training Start ****************')
        epoch = self.current_epoch
        print('[INFO] Starting epoch {}'.format(epoch))
        print('[INFO] current number of points: {}, radius: {}'.format(self.model.pc.points.data.shape[0], self.model.raster_settings.radius))
        print('************************************************\n')

    def on_train_epoch_start(self):     # NOTE callback -> on_train_epoch_start
        # For geometry network annealing frequency band
        epoch = self.current_epoch
        if epoch in self.GT_lbs_milestones:
            self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor
        # if len(self.GT_lbs_milestones) > 0 and epoch >= self.GT_lbs_milestones[-1]:
        #   self.loss.lbs_weight = 0.

    def training_step(self, batch, batch_idx):
        indices = batch[0]
        model_input = batch[1]
        ground_truth = batch[2]
        data_index = batch_idx       # the index starts from 0
        epoch = self.current_epoch
        device = model_input["idx"].device

        if self.optimize_inputs:
            if self.optimize_expression:
                model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
            if self.optimize_pose:
                model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)
            # hyunsoo added
            if self.optimize_latent_code:
                model_input['latent_code'] = self.latent_codes(model_input["idx"]).squeeze(1)
            if self.optimize_scene_latent_code:
                if not self.target_training:
                    # NOTE 각 비디오별로 latent를 부여함 (1, 28)
                    # Convert the names to indices
                    indices = [self.dataset_train_subdir.index(name) for name in model_input['sub_dir']]
                    # Convert indices to a tensor
                    indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                    # Fetch the corresponding latent codes using the Embedding layer
                    scene_latent_codes_tensor = self.scene_latent_codes(indices_tensor)

                    model_input['indices_tensor'] = indices_tensor
                    
                    # NOTE category별로 latent를 부여함 (1, 4)
                    category_latent_codes = []  # List to collect the transformed tensors
                    # Loop over each item in model_input['sub_dir']
                    index_list = []
                    for sub_dir_item in model_input['sub_dir']:
                        for i, v in self.category_dict.items():
                            if i in sub_dir_item:
                                index_list.append(v)
                                tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                                category_latent_codes.append(tensor)
                    category_latent_codes = torch.cat(category_latent_codes, dim=0).to(device).detach()
                    scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

                    # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. 
                    scene_category_latent_code = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))

                    # NOTE zero scene latent로 일단 다 초기화한다.
                    for i, v in self.category_dict.items():
                        category_latent_code = int_to_tensor(v, device=self.device)
                        start_idx = v*scene_latent_codes_tensor.shape[1]
                        end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                        scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.zero_latent_codes(torch.tensor(0).to(device))), dim=0)
                    # NOTE source latent code도 부여함. (1, 4+28)
                    
                    # NOTE multicomposition을 위해 추가한 부분.
                    source_index_list = []
                    source_category_latent_codes = []
                    for i, v in self.source_category_dict.items():
                        source_index_list.append(v)
                        tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                        source_category_latent_codes.append(tensor)
                    source_category_latent_codes = torch.cat(source_category_latent_codes, dim=0).to(device).detach()
                    #######################################

                    source_start_idx = 0
                    source_end_idx = scene_latent_codes_tensor.shape[1]
                    source_category_code = int_to_tensor(0, device=self.device).detach()

                    scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_code, self.source_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

                    # NOTE multicomposition을 위해 추가한 부분.
                    for i, v in enumerate(source_index_list):
                        source_start_idx = v*scene_latent_codes_tensor.shape[1]
                        source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                        scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_latent_codes[i], self.source_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)
                    #######################################

                    start_idx = index_list[0]*scene_latent_codes_tensor.shape[1]
                    end_idx = (index_list[0]+1)*scene_latent_codes_tensor.shape[1]
                    scene_category_latent_code[0, start_idx:end_idx] = scene_latent_codes_tensor[0]
                    
                else:
                    scene_category_latent_code = torch.zeros(1, 32*len(self.category_dict))
                    # NOTE zero scene latent로 일단 다 초기화한다.
                    for i, v in self.category_dict.items():
                        category_latent_code = int_to_tensor(v, device=self.device)
                        start_idx = v*32
                        end_idx = (v+1)*32
                        scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.zero_latent_codes(torch.tensor(0).to(device))), dim=0)

                    target_index_list = []
                    target_category_latent_codes = []
                    for i, v in self.target_category_dict.items():
                        target_index_list.append(v)
                        tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                        target_category_latent_codes.append(tensor)
                    target_category_latent_codes = torch.cat(target_category_latent_codes, dim=0).to(device).detach()

                    target_start_idx = 0
                    target_end_idx = 32
                    target_category_code = int_to_tensor(0, device=self.device).detach()

                    scene_category_latent_code[0, target_start_idx:target_end_idx] = torch.cat((target_category_code, self.target_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

                    for i, v in enumerate(target_index_list):
                        target_start_idx = v*32
                        target_end_idx = (v+1)*32
                        scene_category_latent_code[0, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.target_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)

                model_input['scene_latent_code'] = scene_category_latent_code.to(device)

        model_input['rank'] = self.trainer.global_rank

        model_outputs = self.model(model_input)

        loss_output = self.loss(model_outputs, ground_truth, model_input)

        loss = loss_output['loss']

        # NOTE manual optimization.
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        for k, v in loss_output.items():
            if 'loss' in k:
                log_name = 'Loss/{}'.format(k)
            else:
                log_name = k
            self.log(log_name, v.detach().item())
            loss_output[k] = v.detach().item()
            if k not in self.acc_loss:
                self.acc_loss[k] = [v]
            else:
                self.acc_loss[k].append(v)

        # self.total_loss = self.acc_loss['loss'][0].detach().item()

        if self.enable_prune:
            self.acc_loss['visible_percentage'] = (torch.sum(self.model.visible_points)/self.model.pc.points.shape[0]).unsqueeze(0)

        if data_index % self.print_freq == 0:
            for k, v in self.acc_loss.items():
                self.acc_loss[k] = sum(v) / len(v)
            # print_str = '{0} [{1}] ({2}/{3}): '.format(self.methodname, epoch, data_index, self.n_batches)
            print_str = '{0} [{1}] ({2}/{3}): '.format(self.methodname, epoch, data_index, len(self.trainer.train_dataloader))
            for k, v in self.acc_loss.items():
                print_str += '{}: {:.3g} '.format(k, v)
            print_str += 'num_points: {} radius: {}'.format(self.model.pc.points.shape[0], self.model.raster_settings.radius)
            print(print_str)
            self.acc_loss['num_points'] = self.model.pc.points.shape[0]
            self.acc_loss['radius'] = self.model.raster_settings.radius

            self.acc_loss['lr'] = self.last_lr_scheduler.get_last_lr()[0]
            # wandb.log(self.acc_loss, step=epoch * len(self.train_dataset) + data_index * self.batch_size)
            self.log('num_points', self.acc_loss['num_points'])
            self.log('radius', self.acc_loss['radius'])
            self.log('lr', self.acc_loss['lr'])
            self.log('batch_size', self.batch_size)

            self.acc_loss = {}
        
        # if data_index % 500 == 0:
        #     self.log('num_points_save_pole', self.model.pc.points.shape[0])

        return loss

    def on_train_epoch_end(self):           # NOTE callback -> on_train_epoch_end
        epoch = self.current_epoch

        # NOTE voting
        if self.trainer.global_rank == 0:
            # self.masked_point_cloud_indices = {}
            # for i in range(self.model.voting_table.shape[0]):
            #     row_sum = self.model.voting_table[i, :, :].sum(dim=-1)
            #     mask = (row_sum > self.voting_threshold).float()
            #     count = torch.sum(mask == 1).item()
            #     self.masked_point_cloud_indices[self.dataset_train_subdir[i]] = mask
            #     print('[INFO] sub_dir {} | count {}'.format(self.dataset_train_subdir[i], count))

            # torch.save(self.masked_point_cloud_indices, 
            #             os.path.join(self.voting_save_path, 
            #                         'epoch={}-step={}-points={}-radius={}-threshold={}.pth'.format(epoch, 
            #                                                                                         self.global_step, 
            #                                                                                         self.model.pc.points.shape[0], 
            #                                                                                         round(self.model.raster_settings.radius, 6),
            #                                                                                         self.voting_threshold)))

            torch.save(self.model.voting_table, 
                        os.path.join(self.voting_table_save_path,
                                    'epoch={}-step={}-points={}-radius={}-voting_table.pth'.format(epoch, 
                                                                                                    self.global_step, 
                                                                                                    self.model.pc.points.shape[0], 
                                                                                                    round(self.model.raster_settings.radius, 6))))

        self.start_time.record()
        
        # NOTE 2023.10.10. 23:47 upsample_freq에 맞게 epoch가 끝나고 난 이후에 저장하고 그다음 init에 적용될 수 있도록 해본다.
        # Pruning
        visible_percentage = (torch.sum(self.model.visible_points)/self.model.pc.points.shape[0]).unsqueeze(0)

        # if self.enable_prune and epoch != self.start_epoch and self.model.raster_settings.radius >= 0.006:
        if self.enable_prune and (self.model.raster_settings.radius >= 0.006) and (visible_percentage >= 0.8) and (self.model.pc.points.shape[0] != self.model.pc.max_points):
            if self.train_from_scratch and epoch == 0:
                print('[INFO] Skip prune at the epoch {}'.format(epoch))
            else:
                self.model.pc.prune(self.model.visible_points)
            # self.optimizers().optimizer = torch.optim.Adam([{'params': list(self.model.parameters()) + list(self.input_params)}], lr=self.lr)
        
        # Upsampling
        if self.enable_upsample and epoch % self.upsample_freq == 0:
            # if epoch != 0 and self.model.pc.points.shape[0] <= self.model.pc.max_points: #self.model.pc.max_points / 2:
            # NOTE epoch가 0이더라도 할 수 있게 한다. 어차피 epoch가 끝나고 upsample을 하기 때문에 굳이 못하게 할 이유는 없다.
            if self.model.pc.points.shape[0] < self.model.pc.max_points: #self.model.pc.max_points / 2:
                if self.train_from_scratch and epoch == 0:
                    print('[INFO] Skip upsample at the epoch {}'.format(epoch))
                else:
                    self.upsample_points()
                batch_size = min(int(self.conf.get_int('train.max_points_training') / self.model.pc.points.shape[0]), self.max_batch)
                if batch_size != self.batch_size:
                    self.batch_size = batch_size
                # self.optimizers().optimizer = torch.optim.Adam([{'params': list(self.model.parameters()) + list(self.input_params)}], lr=self.lr)
                    # NOTE original code
                    # self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                    #                                                     batch_size=self.batch_size,
                    #                                                     shuffle=True,
                    #                                                     collate_fn=self.train_dataset.collate_fn,
                    #                                                     num_workers=4,
                    #                                                     )
                    # self.n_batches = len(self.train_dataloader)
                    # NOTE 다음 epoch때 바뀐다. 순서는 train_dataloader -> callback (if exists) -> here이다.
                    # self.datamodule.batch_size = batch_size
            elif self.model.pc.points.shape[0] == self.model.pc.max_points:
                current_radius = self.model.raster_settings.radius
                # points = self.model.pc.points.data
                # if self.model.pc.points.shape[0] <= self.model.pc.max_points / 2:           # 2배씩 증가
                #     noise = (torch.rand(*points.shape, device=points.device) - 0.5) * current_radius
                #     new_points = noise + points
                # else:
                #     new_points_shape = (self.model.pc.max_points - points.shape[0], points.shape[1])
                #     noise = (torch.rand(new_points_shape, device=points.device) - 0.5) * current_radius     # NOTE cuda 삭제

                #     indices = torch.randperm(points.size(0))
                #     selected_points = points[indices[:new_points_shape[0]]]
                #     new_points = noise + selected_points
                # self.model.pc.upsample_points(new_points)                                           # NOTE 기존에서 new_points를 합침.
                if self.model.pc.min_radius < 0.75 * current_radius:
                    new_radius = 0.75 * current_radius
                    self.model.pc.register_radius(new_radius)
                    self.model.raster_settings.radius = new_radius
                print("***************************************************")
                print("old radius: {}, new radius: {}".format(current_radius, self.model.raster_settings.radius))

        # re-init visible point tensor each epoch
        if self.enable_prune:
            self.model.visible_points = torch.zeros(self.model.pc.points.shape[0], device=self.model.visible_points.device).bool()     # .cuda() 삭제

        # self.log('total_loss', self.total_loss)
        # print('[INFO] total_loss: {}'.format(self.total_loss))
        # self.log('num_points_save_pole', self.model.pc.points.shape[0])
        
        self.lr_schedulers().step()             # NOTE 이건 IMavatar 시절에도 했던거라 무리는 없다.
        self.end_time.record()
        torch.cuda.synchronize()
        # wandb.log({"timing_epoch": self.start_time.elapsed_time(self.end_time)}, step=(epoch+1) * len(self.train_dataset))
        self.log('timing_epoch', self.start_time.elapsed_time(self.end_time))           # NOTE for save checkpoints

        if self.enable_prune or self.enable_upsample or self.nepochs == epoch:
            self.trainer.should_stop = True
            self.trainer.limit_val_batches = 0

    # def on_train_batch_start(self, batch, batch_idx):       # NOTE on_train_start() -> on_train_epoch_start() -> on_train_batch_start()
    #     epoch = self.current_epoch
    #     if self.enable_prune and epoch != self.start_epoch and self.model.raster_settings.radius >= 0.006:
    #         return -1
    #     if self.enable_upsample and epoch % self.upsample_freq == 0:
    #         if epoch != 0 and self.model.pc.points.shape[0] <= self.model.pc.max_points / 2:
    #             return -1


    def on_train_end(self):     # NOTE 여기서는 self.log를 사용할 수 없다고 한다..
        print('\n***************** Train End *****************')
        epoch = self.current_epoch
        print('[INFO] Finished epoch {}'.format(epoch))
        print('[INFO] current number of points: {}, radius: {}'.format(self.model.pc.points.data.shape[0], self.model.raster_settings.radius))
        print('************************************************\n')

    def on_validation_epoch_start(self):
        self.start_time.record()

        epoch = self.current_epoch
        if epoch in self.GT_lbs_milestones:
            self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor

    def validation_step(self, batch, batch_idx):
        indices = batch[0]
        model_input = batch[1]
        ground_truth = batch[2]
        data_index = batch_idx
        epoch = self.current_epoch
        device = model_input["idx"].device
        if not self.target_training:
            batch_index, train_subdir = next(iter(enumerate(self.dataset_train_subdir)))
        else:
            batch_index, train_subdir = next(iter(enumerate(self.dataset_target_train_subdir)))

        if self.optimize_inputs:
            if self.optimize_scene_latent_code:
                if self.target_training:
                    scene_category_latent_code = torch.zeros(1, 32*len(self.category_dict))
                    # NOTE zero scene latent로 일단 다 초기화한다.
                    for i, v in self.category_dict.items():
                        category_latent_code = int_to_tensor(v, device=self.device)
                        start_idx = v*32
                        end_idx = (v+1)*32
                        scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.zero_latent_codes(torch.tensor(0).to(device))), dim=0)

                    target_index_list = []
                    target_category_latent_codes = []
                    for i, v in self.target_category_dict.items():
                        target_index_list.append(v)
                        tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                        target_category_latent_codes.append(tensor)
                    target_category_latent_codes = torch.cat(target_category_latent_codes, dim=0).to(device).detach()

                    target_start_idx = 0
                    target_end_idx = 32
                    target_category_code = int_to_tensor(0, device=self.device).detach()

                    scene_category_latent_code[0, target_start_idx:target_end_idx] = torch.cat((target_category_code, self.target_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

                    for i, v in enumerate(target_index_list):
                        target_start_idx = v*32
                        target_end_idx = (v+1)*32
                        scene_category_latent_code[0, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.target_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)

                    # Add to the model_input dictionary
                    # model_input['scene_latent_code'] = scene_latent_codes_tensor
                    model_input['scene_latent_code'] = scene_category_latent_code.to(device)
                else:
                    # validation_index = batch_index
                    scene_latent_codes_tensor = self.scene_latent_codes(torch.LongTensor([batch_index]).to(device)).squeeze(1).detach()     # [1, 28]

                    # NOTE category별로 latent를 부여함 (1, 4)
                    category_latent_codes = []  # List to collect the transformed tensors
                    # Loop over each item in model_input['sub_dir']
                    index_list = []
                    for i, v in self.category_dict.items():
                        if i in train_subdir:
                            index_list.append(v)
                            tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                            category_latent_codes.append(tensor)
                    category_latent_codes = torch.cat(category_latent_codes, dim=0).to(device).detach()
                    scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)
                    
                    # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. 
                    scene_category_latent_code = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))

                    # NOTE zero scene latent로 일단 다 초기화한다.
                    for i, v in self.category_dict.items():
                        category_latent_code = int_to_tensor(v, device=self.device)
                        start_idx = v*scene_latent_codes_tensor.shape[1]
                        end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                        scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.zero_latent_codes(torch.tensor(0).to(device))), dim=0)

                    # NOTE multicomposition을 위해 추가한 부분.
                    source_index_list = []
                    source_category_latent_codes = []
                    for i, v in self.source_category_dict.items():
                        source_index_list.append(v)
                        tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                        source_category_latent_codes.append(tensor)
                    source_category_latent_codes = torch.cat(source_category_latent_codes, dim=0).to(device).detach()
                    #######################################
                    
                    # NOTE source latent code도 부여함. (1, 4+28)
                    source_start_idx = 0
                    source_end_idx = scene_latent_codes_tensor.shape[1]
                    source_category_code = int_to_tensor(0, device=self.device).detach()
                    scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_code, self.source_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

                    # NOTE multicomposition을 위해 추가한 부분.
                    for i, v in enumerate(source_index_list):
                        source_start_idx = v*scene_latent_codes_tensor.shape[1]
                        source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                        scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_latent_codes[i], self.source_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)
                    #######################################

                    start_idx = index_list[0]*scene_latent_codes_tensor.shape[1]
                    end_idx = (index_list[0]+1)*scene_latent_codes_tensor.shape[1]
                    scene_category_latent_code[0, start_idx:end_idx] = scene_latent_codes_tensor[0]

                    # Add to the model_input dictionary
                    model_input['scene_latent_code'] = scene_category_latent_code.to(device)

        with torch.set_grad_enabled(True):
            model_outputs = self.model(model_input)
            
        for k, v in model_outputs.items():
            try:
                model_outputs[k] = v.detach()
            except:
                model_outputs[k] = v

        novel_view_type = 'validation_{}'.format(train_subdir)
        # plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_'+str(epoch)) for i in range(len(model_input['sub_dir']))]
        plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(str(epoch), novel_view_type)) for i in range(len(model_input['sub_dir']))]
        img_names = model_input['img_name'][:, 0].cpu().numpy()
        # print("Plotting images: {}".format(img_names))
        print("Plotting images: {}".format(os.path.join(plot_dir[0], '{}.png'.format(img_names[0]))))
        utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))

        plt.plot(img_names,
                model_outputs,
                ground_truth,
                plot_dir,
                epoch,
                self.img_res,
                is_eval=False,
                first=(data_index==0),
                custom_settings={'wandb_logger': self.logger, 'global_step': self.global_step, 'novel_view': novel_view_type})
        

        del model_outputs
        del ground_truth
        torch.cuda.empty_cache()

    
    def on_validation_epoch_end(self):
        self.end_time.record()
        torch.cuda.synchronize()
        print("Plot time per image: {} ms".format(self.start_time.elapsed_time(self.end_time) / len(self.test_dataset)))

    def test_step(self, batch, batch_idx):
        indices = batch[0]
        model_input = batch[1]
        ground_truth = batch[2]
        data_index = batch_idx

        device = model_input["idx"].device

        # if indices.item() < self.test_start_frame:
        #     return
        # if indices.item() > self.test_end_frame:
        #     return

        eval_all = True
        if self.optimize_inputs:
            if self.optimize_expression:
                model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
            if self.optimize_pose:
                model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)

        is_first_batch = True

        if self.test_default_rendering:
            for batch_index, train_subdir in enumerate(self.dataset_train_subdir):
                novel_view_type = 'default_rendering_{}'.format(train_subdir)
                plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                img_names = model_input['img_name'][:,0].cpu().numpy()[0]
                scene_latent_codes_tensor = self.scene_latent_codes(torch.LongTensor([batch_index]).to(device)).squeeze(1).detach()     # [1, 28]

                # NOTE category별로 latent를 부여함 (1, 4)
                category_latent_codes = []  # List to collect the transformed tensors
                # Loop over each item in model_input['sub_dir']
                index_list = []
                for i, v in self.category_dict.items():
                    if i in train_subdir:
                        index_list.append(v)
                        tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                        category_latent_codes.append(tensor)
                category_latent_codes = torch.cat(category_latent_codes, dim=0).to(device).detach()
                scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)
                
                # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. 
                scene_category_latent_code = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))

                # NOTE zero scene latent로 일단 다 초기화한다.
                for i, v in self.category_dict.items():
                    category_latent_code = int_to_tensor(v, device=self.device)
                    start_idx = v*scene_latent_codes_tensor.shape[1]
                    end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                    scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.zero_latent_codes(torch.tensor(0).to(device))), dim=0)

                # NOTE multicomposition을 위해 추가한 부분.
                source_index_list = []
                source_category_latent_codes = []
                for i, v in self.source_category_dict.items():
                    source_index_list.append(v)
                    tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                    source_category_latent_codes.append(tensor)
                source_category_latent_codes = torch.cat(source_category_latent_codes, dim=0).to(device).detach()
                #######################################

                # NOTE source latent code도 부여함. (1, 4+28)
                source_start_idx = 0
                source_end_idx = scene_latent_codes_tensor.shape[1]
                source_category_code = int_to_tensor(0, device=self.device).detach()
                scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_code, self.source_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

                # NOTE multicomposition을 위해 추가한 부분.
                for i, v in enumerate(source_index_list):
                    source_start_idx = v*scene_latent_codes_tensor.shape[1]
                    source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                    scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_latent_codes[i], self.source_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)
                #######################################

                start_idx = index_list[0]*scene_latent_codes_tensor.shape[1]
                end_idx = (index_list[0]+1)*scene_latent_codes_tensor.shape[1]
                scene_category_latent_code[0, start_idx:end_idx] = scene_latent_codes_tensor[0]

                # Add to the model_input dictionary
                # model_input['scene_latent_code'] = scene_latent_codes_tensor
                model_input['scene_latent_code'] = scene_category_latent_code.to(device)

                with torch.set_grad_enabled(True):
                    model_outputs = self.model(model_input)

                for k, v in model_outputs.items():
                    try:
                        model_outputs[k] = v.detach()
                    except:
                        model_outputs[k] = v
                
                print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_names))))
                utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
                if eval_all:
                    for dir in plot_dir:
                        utils.mkdir_ifnotexists(dir)
                plt.plot(img_names,
                        model_outputs,
                        ground_truth,
                        plot_dir,
                        self.test_epoch,
                        self.img_res,
                        is_eval=eval_all,
                        first=is_first_batch,
                        custom_settings={'novel_view': novel_view_type})
                
                del model_outputs
        
        if self.test_mask_rendering:
            # NOTE generate mask images for target human.
            for batch_index, train_subdir in enumerate(self.dataset_train_subdir):
                # if train_subdir == 'hat_Syuka_foxhat':
                novel_view_type = 'mask_rendering_{}'.format(train_subdir)
                plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                img_names = model_input['img_name'][:,0].cpu().numpy()[0]
                scene_latent_codes_tensor = self.scene_latent_codes(torch.LongTensor([batch_index]).to(device)).squeeze(1).detach()     # [1, 28]

                # NOTE category별로 latent를 부여함 (1, 4)
                category_latent_codes = []  # List to collect the transformed tensors
                # Loop over each item in model_input['sub_dir']
                index_list = []
                for i, v in self.category_dict.items():
                    if i in train_subdir:
                        index_list.append(v)
                        tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                        category_latent_codes.append(tensor)
                category_latent_codes = torch.cat(category_latent_codes, dim=0).to(device).detach()
                scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)
                
                # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. 
                scene_category_latent_code = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))

                # NOTE zero scene latent로 일단 다 초기화한다.
                for i, v in self.category_dict.items():
                    category_latent_code = int_to_tensor(v, device=self.device)
                    start_idx = v*scene_latent_codes_tensor.shape[1]
                    end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                    scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.zero_latent_codes(torch.tensor(0).to(device))), dim=0)

                # NOTE multicomposition을 위해 추가한 부분.
                source_index_list = []
                source_category_latent_codes = []
                for i, v in self.source_category_dict.items():
                    source_index_list.append(v)
                    tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                    source_category_latent_codes.append(tensor)
                source_category_latent_codes = torch.cat(source_category_latent_codes, dim=0).to(device).detach()
                #######################################

                # NOTE source latent code도 부여함. (1, 4+28)
                source_start_idx = 0
                source_end_idx = scene_latent_codes_tensor.shape[1]
                source_category_code = int_to_tensor(0, device=self.device).detach()
                scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_code, self.source_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

                # NOTE multicomposition을 위해 추가한 부분.
                for i, v in enumerate(source_index_list):
                    source_start_idx = v*scene_latent_codes_tensor.shape[1]
                    source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                    scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_latent_codes[i], self.source_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)
                #######################################

                start_idx = index_list[0]*scene_latent_codes_tensor.shape[1]
                end_idx = (index_list[0]+1)*scene_latent_codes_tensor.shape[1]
                scene_category_latent_code[0, start_idx:end_idx] = scene_latent_codes_tensor[0]

                # Add to the model_input dictionary
                # model_input['scene_latent_code'] = scene_latent_codes_tensor
                model_input['indices_tensor'] = batch_index
                model_input['masked_point_cloud_indices'] = self.masked_point_cloud_indices[train_subdir]

                model_input['scene_latent_code'] = scene_category_latent_code.to(device)

                with torch.set_grad_enabled(True):
                    model_outputs = self.model(model_input)

                for k, v in model_outputs.items():
                    try:
                        model_outputs[k] = v.detach()
                    except:
                        model_outputs[k] = v
                
                print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_names))))
                utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
                if eval_all:
                    for dir in plot_dir:
                        utils.mkdir_ifnotexists(dir)
                plt.plot(img_names,
                        model_outputs,
                        ground_truth,
                        plot_dir,
                        self.test_epoch,
                        self.img_res,
                        is_eval=eval_all,
                        first=is_first_batch,
                        custom_settings={'novel_view': novel_view_type})
                
                del model_outputs

        if self.test_target_interpolation:
            for batch_index, train_subdir in enumerate(self.dataset_target_train_subdir):
                novel_view_type = 'target_default_rendering_{}'.format(train_subdir)
                plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                img_names = model_input['img_name'][:,0].cpu().numpy()[0]

                scene_category_latent_code = torch.zeros(1, 32*len(self.category_dict))
                # NOTE zero scene latent로 일단 다 초기화한다.
                for i, v in self.category_dict.items():
                    category_latent_code = int_to_tensor(v, device=self.device)
                    start_idx = v*32
                    end_idx = (v+1)*32
                    scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.zero_latent_codes(torch.tensor(0).to(device))), dim=0)

                target_index_list = []
                target_category_latent_codes = []
                for i, v in self.target_category_dict.items():
                    target_index_list.append(v)
                    tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                    target_category_latent_codes.append(tensor)
                target_category_latent_codes = torch.cat(target_category_latent_codes, dim=0).to(device).detach()

                target_start_idx = 0
                target_end_idx = 32
                target_category_code = int_to_tensor(0, device=self.device).detach()

                scene_category_latent_code[0, target_start_idx:target_end_idx] = torch.cat((target_category_code, self.target_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

                for i, v in enumerate(target_index_list):
                    target_start_idx = v*32
                    target_end_idx = (v+1)*32
                    scene_category_latent_code[0, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.target_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)

                # Add to the model_input dictionary
                # model_input['scene_latent_code'] = scene_latent_codes_tensor
                model_input['scene_latent_code'] = scene_category_latent_code.to(device)

                with torch.set_grad_enabled(True):
                    model_outputs = self.model(model_input)

                for k, v in model_outputs.items():
                    try:
                        model_outputs[k] = v.detach()
                    except:
                        model_outputs[k] = v
                
                print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_names))))
                utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
                if eval_all:
                    for dir in plot_dir:
                        utils.mkdir_ifnotexists(dir)
                plt.plot(img_names,
                        model_outputs,
                        ground_truth,
                        plot_dir,
                        self.test_epoch,
                        self.img_res,
                        is_eval=eval_all,
                        first=is_first_batch,
                        custom_settings={'novel_view': novel_view_type})
                
                del model_outputs


        
            # NOTE 현재로서는 checkpoint 3으로 하는게 제일 잘나온다.
            scene_category_latent_code = torch.zeros(1, 32*len(self.category_dict))
            # NOTE zero scene latent로 일단 다 초기화한다.
            for i, v in self.category_dict.items():
                category_latent_code = int_to_tensor(v, device=self.device)
                start_idx = v*32
                end_idx = (v+1)*32
                scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.test_target_zero_latent_codes(torch.tensor(0).to(device))), dim=0)

            target_index_list = []
            target_category_latent_codes = []
            for i, v in self.target_category_dict.items():
                target_index_list.append(v)
                tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                target_category_latent_codes.append(tensor)
            target_category_latent_codes = torch.cat(target_category_latent_codes, dim=0).to(device).detach()

            target_start_idx = 0
            target_end_idx = 32
            target_category_code = int_to_tensor(0, device=self.device).detach()

            scene_category_latent_code[0, target_start_idx:target_end_idx] = torch.cat((target_category_code, self.test_target_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

            for i, v in enumerate(target_index_list):
                target_start_idx = v*32
                target_end_idx = (v+1)*32
                scene_category_latent_code[0, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.test_target_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)

            # Add to the model_input dictionary
            # model_input['scene_latent_code'] = scene_latent_codes_tensor
            model_input['scene_latent_code'] = scene_category_latent_code.to(device)
            
            # model_input['scene_latent_code'] = self.test_target_scene_latent_codes(torch.tensor([0]).to(model_input['idx'].device)).squeeze(1).detach() 
            model_input['indices_tensor'] = 0
            # model_input['masked_point_cloud_indices'] = self.test_target_human_masked_point_cloud_indices['target_Dave']
            model_input['blending_middle_inference'] = True
            
            with torch.set_grad_enabled(True):
                middle_inference = self.test_target_human_model(model_input)

            for k, v in middle_inference.items():
                try:
                    middle_inference[k] = v.detach().cpu()
                except:
                    middle_inference[k] = v

            model_input['middle_inference'] = middle_inference
            del model_input['scene_latent_code'], model_input['indices_tensor'], model_input['blending_middle_inference'] #  model_input['masked_point_cloud_indices'], 
            del self.test_target_human_model
            self.test_target_human_model = copy.copy(self.test_target_human_model_free_memory)
            gc.collect()
            torch.cuda.empty_cache()


            test_category = self.test_scene_latent_interpolation_category
            
            category_subdir_index = [index for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
            category_subdir = [element for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]

            img_name = model_input['img_name'].item()
            j = category_subdir_index[(img_name-1)%len(category_subdir_index)]
            k = category_subdir_index[img_name%len(category_subdir_index)]
            scene_latent_codes_tensor1 = self.scene_latent_codes(torch.tensor([j]).to(model_input['idx'].device)).squeeze(1).detach() 
            scene_latent_codes_tensor2 = self.scene_latent_codes(torch.tensor([k]).to(model_input['idx'].device)).squeeze(1).detach() 

            train_subdir1 = self.dataset_train_subdir[j] # 'hair_Chloe_Grace_Moretz'
            train_subdir2 = self.dataset_train_subdir[k] #'hair_John_Krasinski'
            batch_index1 = j # self.dataset_train_subdir.index(train_subdir1)
            batch_index2 = k # self.dataset_train_subdir.index(train_subdir2)

            # interpolated_latent = lambda ratio: self.linear_interpolation(latent_a, latent_b, ratio)


            # NOTE generate mask images for target human.
            # train_subdir1 = 'hair_Chloe_Grace_Moretz'
            # train_subdir2 = 'hair_John_Krasinski'
            # batch_index1 = self.dataset_train_subdir.index(train_subdir1)
            # batch_index2 = self.dataset_train_subdir.index(train_subdir2)

            # novel_view_type = 'mask_interpolation_{}_2_{}'.format(train_subdir1, train_subdir2)
            novel_view_type = 'mask_interpolation_{}'.format(test_category)
            plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
            
            # scene_latent_codes_tensor1 = self.scene_latent_codes(torch.LongTensor([batch_index1]).to(device)).squeeze(1).detach()     # [1, 28]
            # scene_latent_codes_tensor2 = self.scene_latent_codes(torch.LongTensor([batch_index2]).to(device)).squeeze(1).detach()     # [1, 28]
            
            def latent_factory(batch_index, train_subdir, scene_latent_codes_tensor):
                category_latent_codes = []  # List to collect the transformed tensors
                # Loop over each item in model_input['sub_dir']
                index_list = []
                for i, v in self.category_dict.items():
                    if i in train_subdir:
                        index_list.append(v)
                        tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                        category_latent_codes.append(tensor)
                category_latent_codes = torch.cat(category_latent_codes, dim=0).to(device).detach()
                scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)
                
                # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. 
                scene_category_latent_code = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))

                # NOTE zero scene latent로 일단 다 초기화한다.
                for i, v in self.category_dict.items():
                    category_latent_code = int_to_tensor(v, device=self.device)
                    start_idx = v*scene_latent_codes_tensor.shape[1]
                    end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                    scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.zero_latent_codes(torch.tensor(0).to(device))), dim=0)

                # NOTE multicomposition을 위해 추가한 부분.
                source_index_list = []
                source_category_latent_codes = []
                for i, v in self.source_category_dict.items():
                    source_index_list.append(v)
                    tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                    source_category_latent_codes.append(tensor)
                source_category_latent_codes = torch.cat(source_category_latent_codes, dim=0).to(device).detach()
                #######################################

                # NOTE source latent code도 부여함. (1, 4+28)
                source_start_idx = 0
                source_end_idx = scene_latent_codes_tensor.shape[1]
                source_category_code = int_to_tensor(0, device=self.device).detach()
                scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_code, self.source_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

                # NOTE multicomposition을 위해 추가한 부분.
                for i, v in enumerate(source_index_list):
                    source_start_idx = v*scene_latent_codes_tensor.shape[1]
                    source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                    scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_latent_codes[i], self.source_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)
                #######################################

                start_idx = index_list[0]*scene_latent_codes_tensor.shape[1]
                end_idx = (index_list[0]+1)*scene_latent_codes_tensor.shape[1]
                scene_category_latent_code[0, start_idx:end_idx] = scene_latent_codes_tensor[0]

                # Add to the model_input dictionary
                # model_input['scene_latent_code'] = scene_latent_codes_tensor
                model_input['indices_tensor'] = batch_index
                model_input['masked_point_cloud_indices'] = self.masked_point_cloud_indices[train_subdir]

                model_input['scene_latent_code'] = scene_category_latent_code.to(device)
                return scene_category_latent_code

            scene_category_latent_code1 = latent_factory(batch_index1, train_subdir1, scene_latent_codes_tensor1)
            scene_category_latent_code2 = latent_factory(batch_index2, train_subdir2, scene_latent_codes_tensor2)           
            
            interpolated_latent = lambda ratio: self.linear_interpolation(scene_category_latent_code1, scene_category_latent_code2, ratio)

            interpolation_frames = 10
            for frame in range(interpolation_frames):
                img_names = model_input['img_name'][:,0].cpu().numpy()[0]
                img_names = np.array('{}-{}'.format(img_names, frame))

                model_input['indices_tensor'] = batch_index1
                model_input['masked_point_cloud_indices'] = self.masked_point_cloud_indices[train_subdir1]
                model_input['scene_latent_code'] = interpolated_latent(frame/interpolation_frames).to(device)

                with torch.set_grad_enabled(True):
                    model_outputs = self.model(model_input)

                for k, v in model_outputs.items():
                    try:
                        model_outputs[k] = v.detach().cpu()
                    except:
                        model_outputs[k] = v
                
                for k, v in ground_truth.items():
                    try:
                        ground_truth[k] = v.detach().cpu()
                    except:
                        ground_truth[k] = v
                
                print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_names))))
                utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
                if eval_all:
                    for dir in plot_dir:
                        utils.mkdir_ifnotexists(dir)
                plt.plot(img_names,
                        model_outputs,
                        ground_truth,
                        plot_dir,
                        self.test_epoch,
                        self.img_res,
                        is_eval=eval_all,
                        first=is_first_batch,
                        custom_settings={'novel_view': novel_view_type})
                
                # for k, v in model_input.items():
                #     try:
                #         model_input[k] = v.detach().cpu()
                #     except:
                #         model_input[k] = v

                del model_outputs # , model_input
                del self.model
                self.model = copy.copy(self.model_free_memory)
                gc.collect()
                torch.cuda.empty_cache()

        if self.test_target_human_blending:
            # NOTE generate mask images for target human.
            for batch_index, train_subdir in enumerate(self.dataset_train_subdir):
                if train_subdir == 'hat_Syuka_foxhat':
                    novel_view_type = 'mask_rendering_{}'.format(train_subdir)
                    plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                    img_names = model_input['img_name'][:,0].cpu().numpy()[0]
                    scene_latent_codes_tensor = self.scene_latent_codes(torch.LongTensor([batch_index]).to(device)).squeeze(1).detach()     # [1, 28]

                    # NOTE category별로 latent를 부여함 (1, 4)
                    category_latent_codes = []  # List to collect the transformed tensors
                    # Loop over each item in model_input['sub_dir']
                    index_list = []
                    for i, v in self.category_dict.items():
                        if i in train_subdir:
                            index_list.append(v)
                            tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                            category_latent_codes.append(tensor)
                    category_latent_codes = torch.cat(category_latent_codes, dim=0).to(device).detach()
                    scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)
                    
                    # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. 
                    scene_category_latent_code = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))

                    # NOTE zero scene latent로 일단 다 초기화한다.
                    for i, v in self.category_dict.items():
                        category_latent_code = int_to_tensor(v, device=self.device)
                        start_idx = v*scene_latent_codes_tensor.shape[1]
                        end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                        scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.zero_latent_codes(torch.tensor(0).to(device))), dim=0)

                    # NOTE multicomposition을 위해 추가한 부분.
                    source_index_list = []
                    source_category_latent_codes = []
                    for i, v in self.source_category_dict.items():
                        source_index_list.append(v)
                        tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                        source_category_latent_codes.append(tensor)
                    source_category_latent_codes = torch.cat(source_category_latent_codes, dim=0).to(device).detach()
                    #######################################

                    # NOTE source latent code도 부여함. (1, 4+28)
                    source_start_idx = 0
                    source_end_idx = scene_latent_codes_tensor.shape[1]
                    source_category_code = int_to_tensor(0, device=self.device).detach()
                    scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_code, self.source_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

                    # NOTE multicomposition을 위해 추가한 부분.
                    for i, v in enumerate(source_index_list):
                        source_start_idx = v*scene_latent_codes_tensor.shape[1]
                        source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                        scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_latent_codes[i], self.source_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)
                    #######################################

                    start_idx = index_list[0]*scene_latent_codes_tensor.shape[1]
                    end_idx = (index_list[0]+1)*scene_latent_codes_tensor.shape[1]
                    scene_category_latent_code[0, start_idx:end_idx] = scene_latent_codes_tensor[0]

                    # Add to the model_input dictionary
                    # model_input['scene_latent_code'] = scene_latent_codes_tensor
                    model_input['indices_tensor'] = batch_index
                    model_input['masked_point_cloud_indices'] = self.masked_point_cloud_indices[train_subdir]

                    model_input['scene_latent_code'] = scene_category_latent_code.to(device)

                    with torch.set_grad_enabled(True):
                        model_outputs = self.model(model_input)

                    for k, v in model_outputs.items():
                        try:
                            model_outputs[k] = v.detach().cpu()
                        except:
                            model_outputs[k] = v
                    
                    for k, v in ground_truth.items():
                        try:
                            ground_truth[k] = v.detach().cpu()
                        except:
                            ground_truth[k] = v
                    
                    print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_names))))
                    utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
                    if eval_all:
                        for dir in plot_dir:
                            utils.mkdir_ifnotexists(dir)
                    plt.plot(img_names,
                            model_outputs,
                            ground_truth,
                            plot_dir,
                            self.test_epoch,
                            self.img_res,
                            is_eval=eval_all,
                            first=is_first_batch,
                            custom_settings={'novel_view': novel_view_type})
                    
                    for k, v in model_input.items():
                        try:
                            model_input[k] = v.detach().cpu()
                        except:
                            model_input[k] = v

                    del model_outputs, model_input
                    del self.model
                    self.model = copy.copy(self.model_free_memory)
                    gc.collect()
                    torch.cuda.empty_cache()
        

        if self.test_target_human_chamfer_blending:
            # NOTE target human과 source human + DB를 합성해준다.
            scene_category_latent_code = torch.zeros(1, 32*len(self.category_dict))
            # NOTE zero scene latent로 일단 다 초기화한다.
            for i, v in self.category_dict.items():
                category_latent_code = int_to_tensor(v, device=self.device)
                start_idx = v*32
                end_idx = (v+1)*32
                scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.test_target_zero_latent_codes(torch.tensor(0).to(device))), dim=0)

            target_index_list = []
            target_category_latent_codes = []
            for i, v in self.target_category_dict.items():
                target_index_list.append(v)
                tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                target_category_latent_codes.append(tensor)
            target_category_latent_codes = torch.cat(target_category_latent_codes, dim=0).to(device).detach()

            target_start_idx = 0
            target_end_idx = 32
            target_category_code = int_to_tensor(0, device=self.device).detach()

            scene_category_latent_code[0, target_start_idx:target_end_idx] = torch.cat((target_category_code, self.test_target_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

            for i, v in enumerate(target_index_list):
                target_start_idx = v*32
                target_end_idx = (v+1)*32
                scene_category_latent_code[0, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.test_target_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)

            # Add to the model_input dictionary
            # model_input['scene_latent_code'] = scene_latent_codes_tensor
            model_input['scene_latent_code'] = scene_category_latent_code.to(device)
            
            # model_input['scene_latent_code'] = self.test_target_scene_latent_codes(torch.tensor([0]).to(model_input['idx'].device)).squeeze(1).detach() 
            model_input['indices_tensor'] = 0
            # model_input['masked_point_cloud_indices'] = self.test_target_human_masked_point_cloud_indices['target_Dave']      # NOTE TH는 특별히 해주는 것 없이 전부 다 들고와준다.
            model_input['blending_middle_inference'] = True
            
            with torch.set_grad_enabled(True):
                middle_inference = self.test_target_human_model(model_input)

            for k, v in middle_inference.items():
                try:
                    middle_inference[k] = v.detach().cpu()
                except:
                    middle_inference[k] = v

            model_input['middle_inference'] = middle_inference
            del model_input['scene_latent_code'], model_input['indices_tensor'], model_input['blending_middle_inference']   # model_input['masked_point_cloud_indices'], 
            del self.test_target_human_model
            self.test_target_human_model = copy.copy(self.test_target_human_model_free_memory)
            gc.collect()
            torch.cuda.empty_cache()

            # NOTE generate mask images for target human.
            for batch_index, train_subdir in enumerate(self.dataset_train_subdir):
                if train_subdir == 'hair_Chloe_Grace_Moretz':
                    novel_view_type = 'mask_rendering_{}'.format(train_subdir)
                    plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                    img_names = model_input['img_name'][:,0].cpu().numpy()[0]
                    scene_latent_codes_tensor = self.scene_latent_codes(torch.LongTensor([batch_index]).to(device)).squeeze(1).detach()     # [1, 28]

                    # NOTE category별로 latent를 부여함 (1, 4)
                    category_latent_codes = []  # List to collect the transformed tensors
                    # Loop over each item in model_input['sub_dir']
                    index_list = []
                    for i, v in self.category_dict.items():
                        if i in train_subdir:
                            index_list.append(v)
                            tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                            category_latent_codes.append(tensor)
                    category_latent_codes = torch.cat(category_latent_codes, dim=0).to(device).detach()
                    scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)
                    
                    # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. 
                    scene_category_latent_code = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))

                    # NOTE zero scene latent로 일단 다 초기화한다.
                    for i, v in self.category_dict.items():
                        category_latent_code = int_to_tensor(v, device=self.device)
                        start_idx = v*scene_latent_codes_tensor.shape[1]
                        end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                        scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.zero_latent_codes(torch.tensor(0).to(device))), dim=0)

                    # NOTE multicomposition을 위해 추가한 부분.
                    source_index_list = []
                    source_category_latent_codes = []
                    for i, v in self.source_category_dict.items():
                        source_index_list.append(v)
                        tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                        source_category_latent_codes.append(tensor)
                    source_category_latent_codes = torch.cat(source_category_latent_codes, dim=0).to(device).detach()
                    #######################################

                    # NOTE source latent code도 부여함. (1, 4+28)
                    source_start_idx = 0
                    source_end_idx = scene_latent_codes_tensor.shape[1]
                    source_category_code = int_to_tensor(0, device=self.device).detach()
                    scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_code, self.source_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

                    # NOTE multicomposition을 위해 추가한 부분.
                    for i, v in enumerate(source_index_list):
                        source_start_idx = v*scene_latent_codes_tensor.shape[1]
                        source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                        scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_latent_codes[i], self.source_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)
                    #######################################

                    start_idx = index_list[0]*scene_latent_codes_tensor.shape[1]
                    end_idx = (index_list[0]+1)*scene_latent_codes_tensor.shape[1]
                    scene_category_latent_code[0, start_idx:end_idx] = scene_latent_codes_tensor[0]

                    # Add to the model_input dictionary
                    # model_input['scene_latent_code'] = scene_latent_codes_tensor
                    model_input['indices_tensor'] = batch_index
                    model_input['masked_point_cloud_indices'] = self.masked_point_cloud_indices[train_subdir]

                    model_input['scene_latent_code'] = scene_category_latent_code.to(device)

                    with torch.set_grad_enabled(True):
                        model_outputs = self.model(model_input)

                    for k, v in model_outputs.items():
                        try:
                            model_outputs[k] = v.detach().cpu()
                        except:
                            model_outputs[k] = v
                    
                    for k, v in ground_truth.items():
                        try:
                            ground_truth[k] = v.detach().cpu()
                        except:
                            ground_truth[k] = v
                    
                    print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_names))))
                    utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
                    if eval_all:
                        for dir in plot_dir:
                            utils.mkdir_ifnotexists(dir)
                    plt.plot(img_names,
                            model_outputs,
                            ground_truth,
                            plot_dir,
                            self.test_epoch,
                            self.img_res,
                            is_eval=eval_all,
                            first=is_first_batch,
                            custom_settings={'novel_view': novel_view_type})
                    
                    for k, v in model_input.items():
                        try:
                            model_input[k] = v.detach().cpu()
                        except:
                            model_input[k] = v

                    del model_outputs, model_input
                    del self.model
                    self.model = copy.copy(self.model_free_memory)
                    gc.collect()
                    torch.cuda.empty_cache()




        # if self.test_target_human_delta_blending:
        #     model_input['scene_latent_code'] = self.test_target_scene_latent_codes(torch.tensor([0]).to(model_input['idx'].device)).squeeze(1).detach() 
        #     model_input['indices_tensor'] = 0
        #     model_input['masked_point_cloud_indices'] = self.test_target_human_masked_point_cloud_indices['Dave']
        #     model_input['blending_middle_inference'] = True
            
        #     with torch.set_grad_enabled(True):
        #         middle_inference = self.test_target_human_model(model_input)

        #     for k, v in middle_inference.items():
        #         try:
        #             middle_inference[k] = v.detach().cpu()
        #         except:
        #             middle_inference[k] = v

        #     model_input['middle_inference'] = middle_inference
        #     del model_input['scene_latent_code'], model_input['indices_tensor'], model_input['masked_point_cloud_indices'], model_input['blending_middle_inference']
        #     del self.test_target_human_model
        #     self.test_target_human_model = copy.copy(self.test_target_human_model_free_memory)
        #     gc.collect()
        #     torch.cuda.empty_cache()

        #     # NOTE generate mask images for target human.
        #     for batch_index, train_subdir in enumerate(self.dataset_train_subdir):
        #         if train_subdir == 'hat_Syuka_foxhat':
        #             novel_view_type = 'mask_rendering_{}'.format(train_subdir)
        #             plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
        #             img_names = model_input['img_name'][:,0].cpu().numpy()[0]
        #             scene_latent_codes_tensor = self.scene_latent_codes(torch.LongTensor([batch_index]).to(device)).squeeze(1).detach()     # [1, 28]

        #             # NOTE category별로 latent를 부여함 (1, 4)
        #             category_latent_codes = []  # List to collect the transformed tensors
        #             # Loop over each item in model_input['sub_dir']
        #             index_list = []
        #             for i, v in self.category_dict.items():
        #                 if i in train_subdir:
        #                     index_list.append(v)
        #                     tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
        #                     category_latent_codes.append(tensor)
        #             category_latent_codes = torch.cat(category_latent_codes, dim=0).to(device).detach()
        #             scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)
                    
        #             # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. 
        #             scene_category_latent_code = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))

        #             # NOTE zero scene latent로 일단 다 초기화한다.
        #             for i, v in self.category_dict.items():
        #                 category_latent_code = int_to_tensor(v, device=self.device)
        #                 start_idx = v*scene_latent_codes_tensor.shape[1]
        #                 end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
        #                 scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.zero_latent_codes(torch.tensor(0).to(device))), dim=0)

        #             # NOTE multicomposition을 위해 추가한 부분.
        #             source_index_list = []
        #             source_category_latent_codes = []
        #             for i, v in self.source_category_dict.items():
        #                 source_index_list.append(v)
        #                 tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
        #                 source_category_latent_codes.append(tensor)
        #             source_category_latent_codes = torch.cat(source_category_latent_codes, dim=0).to(device).detach()
        #             #######################################

        #             # NOTE source latent code도 부여함. (1, 4+28)
        #             source_start_idx = 0
        #             source_end_idx = scene_latent_codes_tensor.shape[1]
        #             source_category_code = int_to_tensor(0, device=self.device).detach()
        #             scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_code, self.source_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

        #             # NOTE multicomposition을 위해 추가한 부분.
        #             for i, v in enumerate(source_index_list):
        #                 source_start_idx = v*scene_latent_codes_tensor.shape[1]
        #                 source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
        #                 scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_latent_codes[i], self.source_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)
        #             #######################################

        #             start_idx = index_list[0]*scene_latent_codes_tensor.shape[1]
        #             end_idx = (index_list[0]+1)*scene_latent_codes_tensor.shape[1]
        #             scene_category_latent_code[0, start_idx:end_idx] = scene_latent_codes_tensor[0]

        #             # Add to the model_input dictionary
        #             # model_input['scene_latent_code'] = scene_latent_codes_tensor
        #             model_input['indices_tensor'] = batch_index
        #             model_input['masked_point_cloud_indices'] = self.masked_point_cloud_indices[train_subdir]

        #             model_input['scene_latent_code'] = scene_category_latent_code.to(device)

        #             with torch.set_grad_enabled(True):
        #                 model_outputs = self.model(model_input)

        #             for k, v in model_outputs.items():
        #                 try:
        #                     model_outputs[k] = v.detach().cpu()
        #                 except:
        #                     model_outputs[k] = v
                    
        #             for k, v in ground_truth.items():
        #                 try:
        #                     ground_truth[k] = v.detach().cpu()
        #                 except:
        #                     ground_truth[k] = v
                    
        #             print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_names))))
        #             utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
        #             if eval_all:
        #                 for dir in plot_dir:
        #                     utils.mkdir_ifnotexists(dir)
        #             plt.plot(img_names,
        #                     model_outputs,
        #                     ground_truth,
        #                     plot_dir,
        #                     self.test_epoch,
        #                     self.img_res,
        #                     is_eval=eval_all,
        #                     first=is_first_batch,
        #                     custom_settings={'novel_view': novel_view_type})
                    
        #             for k, v in model_input.items():
        #                 try:
        #                     model_input[k] = v.detach().cpu()
        #                 except:
        #                     model_input[k] = v

        #             del model_outputs, model_input
        #             del self.model
        #             self.model = copy.copy(self.model_free_memory)
        #             gc.collect()
        #             torch.cuda.empty_cache()
        

        if self.test_multi_composition:
            # multi_composition = [0, 17]
            compositional_part, scene_latent_codes_tensors = [], []
            # for mc in multi_composition:
            for mc in self.test_multi_composition_list:
                compositional_part.append(mc)
                scene_latent_codes_tensors.append(self.scene_latent_codes(torch.LongTensor([self.dataset_train_subdir.index(mc)]).to(device)).squeeze(1).detach())     # [1, 28]

            novel_view_type = 'multi_composition_{}'.format('__'.join(compositional_part))
            plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
            img_names = model_input['img_name'][:,0].cpu().numpy()[0]

            # NOTE category별로 latent를 부여함 (1, 4)
            category_latent_codes = []  # List to collect the transformed tensors
            # Loop over each item in model_input['sub_dir']
            index_list = []
            for i, v in self.category_dict.items():
                for compart in compositional_part:
                    if i in compart:
                        assert v not in index_list, 'multi composition should not be the same category'
                        index_list.append(v)
                        tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                        category_latent_codes.append(tensor)
            
            scene_latent_codes_tensors_list = []
            for category, scene in zip(category_latent_codes, scene_latent_codes_tensors):
                scene_latent_code_final = torch.cat((category.to(device).detach(), scene), dim=1)
                scene_latent_codes_tensors_list.append(scene_latent_code_final)
            # category_latent_codes = torch.cat(category_latent_codes, dim=0).type_as(scene_latent_codes_tensor).detach()
            # scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)
            
            # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. 
            scene_category_latent_code = torch.zeros(scene_latent_codes_tensors_list[0].shape[0], scene_latent_codes_tensors_list[0].shape[1]*len(self.category_dict))

            # NOTE zero scene latent로 일단 다 초기화한다.
            for i, v in self.category_dict.items():
                category_latent_code = int_to_tensor(v, device=self.device)
                start_idx = v*scene_latent_codes_tensor.shape[1]
                end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.zero_latent_codes(torch.tensor(0).to(device))), dim=0)

            # NOTE multicomposition을 위해 추가한 부분.
            source_index_list = []
            source_category_latent_codes = []
            for i, v in self.source_category_dict.items():
                source_index_list.append(v)
                tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                source_category_latent_codes.append(tensor)
            source_category_latent_codes = torch.cat(source_category_latent_codes, dim=0).to(device).detach()
            #######################################

            # NOTE source latent code도 부여함. (1, 4+28)
            source_start_idx = 0
            source_end_idx = scene_latent_codes_tensors_list[0].shape[1]
            source_category_code = int_to_tensor(0, device=self.device).detach()
            scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_code, self.source_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

            # NOTE multicomposition을 위해 추가한 부분.
            for i, v in enumerate(source_index_list):
                source_start_idx = v*scene_latent_codes_tensors_list[0].shape[1]
                source_end_idx = (v+1)*scene_latent_codes_tensors_list[0].shape[1]
                scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_latent_codes[i], self.source_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)
            #######################################

            for idx, value in enumerate(scene_latent_codes_tensors_list):
                start_idx = index_list[idx]*value.shape[1]
                end_idx = (index_list[idx]+1)*value.shape[1]
                scene_category_latent_code[0, start_idx:end_idx] = value[0]

            # Add to the model_input dictionary
            model_input['scene_latent_code'] = scene_category_latent_code.to(device)

            with torch.set_grad_enabled(True):
                model_outputs = self.model(model_input)

            for k, v in model_outputs.items():
                try:
                    model_outputs[k] = v.detach()
                except:
                    model_outputs[k] = v
            
            print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_names))))
            utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
            if eval_all:
                for dir in plot_dir:
                    utils.mkdir_ifnotexists(dir)
            plt.plot(img_names,
                    model_outputs,
                    ground_truth,
                    plot_dir,
                    self.test_epoch,
                    self.img_res,
                    is_eval=eval_all,
                    first=is_first_batch,
                    custom_settings={'novel_view': novel_view_type})
            
            del model_outputs

        if self.test_scene_latent_interpolation:
            test_category = self.test_scene_latent_interpolation_category
            
            category_subdir_index = [index for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
            category_subdir = [element for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]

            img_name = model_input['img_name'].item()
            j = category_subdir_index[(img_name-1)%len(category_subdir_index)]
            k = category_subdir_index[img_name%len(category_subdir_index)]
            latent_a = self.scene_latent_codes(torch.tensor([j]).to(model_input['idx'].device)).squeeze(1).detach() 
            latent_b = self.scene_latent_codes(torch.tensor([k]).to(model_input['idx'].device)).squeeze(1).detach() 
            interpolated_latent = lambda ratio: self.linear_interpolation(latent_a, latent_b, ratio)

            scene_category_latent_code = torch.zeros(1, 32*len(self.category_dict)).to(device)

            # NOTE zero scene latent로 일단 다 초기화한다.
            for i, v in self.category_dict.items():
                category_latent_code = int_to_tensor(v, device=self.device)
                start_idx = v*scene_latent_codes_tensor.shape[1]
                end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code.to(device), self.zero_latent_codes(torch.tensor(0).to(device))), dim=0)

            source_index_list = []
            source_category_latent_codes = []
            for i, v in self.source_category_dict.items():
                source_index_list.append(v)
                tensor = int_to_tensor(v, device=self.device).unsqueeze(0)
                source_category_latent_codes.append(tensor)
            source_category_latent_codes = torch.cat(source_category_latent_codes, dim=0).to(device).detach()

            # NOTE source latent code도 부여함. (1, 4+28)
            source_start_idx = 0
            source_end_idx = 32
            source_category_code = int_to_tensor(0, device=self.device).detach()
            scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_code, self.source_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

            # NOTE multicomposition을 위해 추가한 부분.
            for i, v in enumerate(source_index_list):
                source_start_idx = v*32
                source_end_idx = (v+1)*32
                scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_latent_codes[i], self.source_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)
            #######################################

            # NOTE category별로 latent를 부여함 (1, 4)
            category_value = self.category_dict[test_category]
            category_latent_code = int_to_tensor(category_value, device=self.device).unsqueeze(0)
            
            for l in range(self.test_interpolation_step_size):
                # NOTE interpolation을 하는 코드.
                novel_view_type = '{}_interpolation'.format(test_category)
                plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                img_name = model_input['img_name'].item()
                img_name = np.array('{}-{}'.format(img_name, l))

                scene_latent_code = interpolated_latent(l/self.test_interpolation_step_size)
                scene_latent_code = torch.cat((category_latent_code.to(device).detach(), scene_latent_code), dim=1)

                start_idx = category_value*32
                end_idx = (category_value+1)*32
                scene_category_latent_code[0, start_idx:end_idx] = scene_latent_code[0]

                model_input['scene_latent_code'] = scene_category_latent_code

                with torch.set_grad_enabled(True):
                    model_outputs = self.model(model_input)

                for k, v in model_outputs.items():
                    try:
                        model_outputs[k] = v.detach()
                    except:
                        model_outputs[k] = v
                
                print("[INFO] Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_name))))
                utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
                if eval_all:
                    for dir in plot_dir:
                        utils.mkdir_ifnotexists(dir)
                plt.plot(img_name,
                        model_outputs,
                        ground_truth,
                        plot_dir,
                        self.test_epoch,
                        self.img_res,
                        is_eval=eval_all,
                        first=is_first_batch,
                        custom_settings={'novel_view': novel_view_type})
                
                del model_outputs

        # if self.test_target_finetuning:
        #     # NOTE target human에 대해서 latent code를 만듦
        #     target_scene_cateogry_latent_code = torch.zeros(1, 32*len(self.category_dict), device=device)

        #     # NOTE zero scene latent로 일단 다 초기화한다.
        #     for i, v in self.category_dict.items():
        #         category_latent_code = int_to_tensor(v, device=device)
        #         start_idx = v*32
        #         end_idx = (v+1)*32
        #         target_scene_cateogry_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code, self.zero_latent_codes(torch.tensor(0, device=device))), dim=0)

        #     target_index_list = []
        #     target_category_latent_codes = []
        #     for i, v in self.target_category_dict.items():
        #         target_index_list.append(v)
        #         tensor = int_to_tensor(v, device=device).unsqueeze(0)
        #         target_category_latent_codes.append(tensor)
        #     target_category_latent_codes = torch.cat(target_category_latent_codes, dim=0).detach()

        #     target_start_idx = 0
        #     target_end_idx = 32
        #     target_category_code = int_to_tensor(0, device=device).detach()

        #     target_scene_cateogry_latent_code[0, target_start_idx:target_end_idx] = torch.cat((target_category_code, self.target_scene_latent_codes(torch.tensor(0, device=device))), dim=0)

        #     for i, v in enumerate(target_index_list):
        #         target_start_idx = v*32
        #         target_end_idx = (v+1)*32
        #         target_scene_cateogry_latent_code[0, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.target_scene_latent_category_codes(torch.tensor(i, device=device))), dim=0)

        #     # Add to the model_input dictionary
        #     model_input['target_scene_latent_code'] = target_scene_cateogry_latent_code

        #     # NOTE source human에 대해서 latent code를 만듦.
        #     scene_latent_codes_tensor = self.scene_latent_codes(torch.tensor([self.dataset_train_subdir.index(self.test_target_finetuning_subdir)], device=device, dtype=torch.long)).squeeze(1).detach()     # [1, 28]

        #     # NOTE category별로 latent를 부여함 (1, 4)
        #     category_latent_codes = []  # List to collect the transformed tensors
        #     index_list = []
        #     target_category = self.test_target_finetuning_subdir.split('_')[0]
        #     model_input['target_category'] = target_category
        #     for i, v in self.category_dict.items():
        #         if i in target_category:
        #             index_list.append(v)
        #             tensor = int_to_tensor(v, device=device).unsqueeze(0)
        #             category_latent_codes.append(tensor)
        #     category_latent_codes = torch.cat(category_latent_codes, dim=0).detach()
        #     scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

        #     # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. 
        #     source_scene_category_latent_code = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict), device=device)

        #     # NOTE zero scene latent로 일단 다 초기화한다.
        #     for i, v in self.category_dict.items():
        #         category_latent_code = int_to_tensor(v, device=device)
        #         start_idx = v*scene_latent_codes_tensor.shape[1]
        #         end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
        #         source_scene_category_latent_code[:, start_idx:end_idx] = torch.cat((category_latent_code, self.zero_latent_codes(torch.tensor(0, device=device))), dim=0)

        #     # NOTE source human의 category latent를 추가하는 부분이다.
        #     source_index_list = []
        #     source_category_latent_codes = []
        #     for i, v in self.source_category_dict.items():
        #         source_index_list.append(v)
        #         tensor = int_to_tensor(v, device=device).unsqueeze(0)
        #         source_category_latent_codes.append(tensor)
        #     source_category_latent_codes = torch.cat(source_category_latent_codes, dim=0).detach()
        #     #######################################

        #     # NOTE source latent code도 부여함. (1, 4+28)
        #     source_start_idx = 0
        #     source_end_idx = scene_latent_codes_tensor.shape[1]
        #     source_category_code = int_to_tensor(0, device=device).detach()
        #     source_scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_code, self.source_scene_latent_codes(torch.tensor(0, device=device))), dim=0)

        #     # NOTE category latent랑 scene latent를 합쳐준다.
        #     for i, v in enumerate(source_index_list):
        #         source_start_idx = v*scene_latent_codes_tensor.shape[1]
        #         source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
        #         source_scene_category_latent_code[0, source_start_idx:source_end_idx] = torch.cat((source_category_latent_codes[i], self.source_scene_latent_category_codes(torch.tensor(i, device=device))), dim=0)
        #     #######################################

        #     start_idx = index_list[0]*scene_latent_codes_tensor.shape[1]
        #     end_idx = (index_list[0]+1)*scene_latent_codes_tensor.shape[1]
        #     source_scene_category_latent_code[0, start_idx:end_idx] = scene_latent_codes_tensor[0]

        #     # Add to the model_input dictionary
        #     model_input['source_scene_latent_code'] = source_scene_category_latent_code

        #     novel_view_type = 'target_finetuning_{}'.format(self.test_target_finetuning_subdir)
        #     plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]

        #     with torch.set_grad_enabled(True):
        #         # segment_mask = self.model.deformer_mask_generator(model_input, 'source')
        #         # model_input['segment_mask'] = segment_mask
        #         # del segment_mask
        #         # torch.cuda.empty_cache()
        #         model_input['segment_mask'] = self.segment_masks[self.test_target_finetuning_subdir]
        #         model_outputs = self.model(model_input)

        #     for k, v in model_outputs.items():
        #         try:
        #             model_outputs[k] = v.detach()
        #         except:
        #             model_outputs[k] = v
            
            
        #     img_name = model_input['img_name'][:,0].cpu().numpy()[0]

        #     print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_name))))
        #     utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
        #     if eval_all:
        #         for dir in plot_dir:
        #             utils.mkdir_ifnotexists(dir)
        #     plt.plot(img_name,
        #             model_outputs,
        #             ground_truth,
        #             plot_dir,
        #             self.test_epoch,
        #             self.img_res,
        #             is_eval=eval_all,
        #             first=is_first_batch,
        #             custom_settings={'novel_view': novel_view_type})
            
        #     del model_outputs

        del ground_truth
        gc.collect()
        torch.cuda.empty_cache()

    def novelView(self, cam_pose, euler_angle, translation):
        ##############################################
        # euler_angle should be tuple type
        x_euler, y_euler, z_euler = euler_angle
        ##############################################

        ## for novel view synthesis
        cam_R = cam_pose[:, :3, :3]
        cam_t = cam_pose[:, :3, -1]
        cam_t = cam_t.reshape(-1, 1).unsqueeze(0)
        world_R, world_t = camtoworld_to_worldtocam_Rt(cam_R, cam_t)
        
        rotate_world_R = quaternion_to_rotation_matrix(rotation_matrix_to_quaternion(world_R)+torch.cat(quaternion_from_euler(x_euler, y_euler, z_euler)).unsqueeze(0))
        trans_world_t = world_t + translation.reshape(-1, 1).unsqueeze(0)

        rotate_cam_R, rotate_cam_t = worldtocam_to_camtoworld_Rt(rotate_world_R, trans_world_t)
        cam_pose = Rt_to_matrix4x4(rotate_cam_R, rotate_cam_t)
        ##############################################
        return cam_pose

    def linear_interpolation(self, x1, x2, ratio):
        return x1 + ratio * (x2 - x1)