import os
from pyhocon import ConfigFactory
import sys
import torch
import utils.general as utils
import utils.plots as plt
from functools import partial
import numpy as np
from kornia.geometry.conversions import Rt_to_matrix4x4
from utils.rotation_converter import *
import copy
from tqdm import tqdm
import gc
import natsort

print = partial(print, flush=True)


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
  
    
class TestRunnerPEGASUS():
    # TestRunnerSceneLatentCategoryMultiComposition
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        # self.conf.put('dataset.test.subsample', 1)
        # self.conf.put('dataset.test.load_images', False)

        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        # NOTE custom
        # self.category_specific = self.conf.get_bool('dataset.category_specific')
        # self.num_category = self.conf.get_int('dataset.num_category')
        # self.category_latent_dim = self.conf.get_int('dataset.category_latent_dim')

        self.optimize_latent_code = self.conf.get_bool('train.optimize_latent_code')
        self.optimize_scene_latent_code = self.conf.get_bool('train.optimize_scene_latent_code')
        self.target_training = self.conf.get_bool('train.target_training', default=False)
        
        self.multi_source_training = self.conf.get_bool('train.multi_source_training', default=False)

        self.dataset_train_subdir = self.conf.get_list('dataset.train.sub_dir')
        self.dataset_test_subdir = self.conf.get_list('dataset.test.sub_dir')

        if self.optimize_latent_code:
            self.latent_dim = 32
        else:
            self.latent_dim = 0
        if self.optimize_scene_latent_code:
            self.scene_latent_dim = 32
        else:
            self.scene_latent_dim = 0
        
        self.category_dict = {
            'identity': 0,
            'ears': 1,
            'eyebrows': 2,
            'eyes': 3,
            'hair': 4,
            'hat': 5,
            'mouth': 6,
            'nose': 7,
            'eyeglasses': 8
        }
        self.category_latent_dim = len(self.category_dict)
        # assert len(self.category_dict) == self.num_category
        self.num_category = len(self.category_dict)
        self.source_category_dict = {
            'identity': 0,
            'ears': 1,
            'eyebrows': 2,
            'eyes': 3,
            'hair': 4,
            'mouth': 6,
            'nose': 7,
        }
        self.target_category_dict = {
            'identity': 0,
            'ears': 1,
            'eyebrows': 2,
            'eyes': 3,
            'hair': 4,
            'mouth': 6,
            'nose': 7,
            'eyeglasses': 8
        }
        if self.multi_source_training:
            self.without_hat_category_dict = {
                'identity': 0,
                'ears': 1,
                'eyebrows': 2,
                'eyes': 3,
                'hair': 4,
                'mouth': 6,
                'nose': 7
            }
            self.with_hat_category_dict = {
                'identity': 0,
                'ears': 1,
                'eyebrows': 2,
                'eyes': 3,
                'hair': 4,
                'hat': 5,
                'mouth': 6,
                'nose': 7
            }
            self.source_datasets = ['Guy']
            self.hat_dataset_train_subdir = [
                'Danny_DeVito', 'Ice_Cube', 'Jeff_Staple', 'Sommelier', 'Syuka_beigehat', 
                'Syuka_brownhat', 'Syuka_cap', 'Syuka_crimsonhat', 'Syuka_orangehat', 'Syuka_foxhat', 
                'Syuka_pinkhat', 'Syuka_redhat', 'Syuka_santahat'
            ]
            self.no_hat_dataset_train_subdir = [item for item in self.dataset_train_subdir if (item not in self.hat_dataset_train_subdir) and (item not in self.source_datasets)]
        ####################
        
        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.dataset_train_subdir)

        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        # if kwargs['load_path'] != '':
        #     load_path = kwargs['load_path']
        # else:
        #     load_path = self.train_dir
        # assert os.path.exists(load_path)

        utils.mkdir_ifnotexists(self.eval_dir)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.use_background = self.conf.get_bool('dataset.use_background', default=False)

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(conf=self.conf,
                                                                                          data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                          subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                          json_name=self.conf.get_string('dataset.json_name'),
                                                                                          use_mean_expression=self.conf.get_bool('dataset.use_mean_expression', default=False),
                                                                                          use_background=self.use_background,
                                                                                          only_json=True,
                                                                                          mode='train',
                                                                                          **self.conf.get_config('dataset.train'))

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(conf=self.conf,
                                                                                         data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         only_json=kwargs['only_json'],
                                                                                         use_background=self.use_background,
                                                                                         mode='test',
                                                                                         **self.conf.get_config('dataset.test'))

        self.val_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(conf=self.conf,
                                                                                         data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         only_json=kwargs['only_json'],
                                                                                         use_background=self.use_background,
                                                                                         mode='val',
                                                                                         **self.conf.get_config('dataset.val'))
        
        print('Finish loading data ...')

        pcd_init = {}
        saved_model_state = torch.load(
            os.path.join(self.train_dir, 'checkpoints', 'ModelParameters', '{}.pth'.format(str(kwargs['checkpoint']))), map_location=lambda storage, loc: storage)

        pcd_init['n_init_points'] = saved_model_state["model_state_dict"]['pc.points'].shape[0]
        pcd_init['init_radius'] = saved_model_state['radius']

        latent_code_dim = (self.scene_latent_dim + self.category_latent_dim) * len(self.category_dict)
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf,
                                      shape_params=self.plot_dataset.shape_params,
                                      img_res=self.plot_dataset.img_res,
                                      canonical_expression=self.train_dataset.mean_expression,
                                      canonical_pose=self.conf.get_float(
                                          'dataset.canonical_pose',
                                          default=0.2),
                                      use_background=self.use_background,
                                      checkpoint_path=None,
                                      latent_code_dim=latent_code_dim,
                                      pcd_init=pcd_init)
        
        if torch.cuda.is_available():
            self.model.cuda()
        old_checkpnts_dir = os.path.join(self.train_dir, 'checkpoints')
        self.checkpoints_path = old_checkpnts_dir
        assert os.path.exists(old_checkpnts_dir)
        saved_model_state = torch.load(
            os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"), map_location=lambda storage, loc: storage)
        n_points = saved_model_state["model_state_dict"]['pc.points'].shape[0]
        self.model.pc.init(n_points)
        self.model.pc = self.model.pc.cuda()

        self.model.raster_settings.radius = saved_model_state['radius']

        self.model.load_state_dict(saved_model_state["model_state_dict"]) #, strict=False)
        self.start_epoch = saved_model_state['epoch']

        self.optimize_expression = self.conf.get_bool('train.optimize_expression') # hyunsoo added, originally, train
        self.optimize_pose = self.conf.get_bool('train.optimize_camera') # hyunsoo added, originally, train
        # hyunsoo added
        self.optimize_latent_code = self.conf.get_bool('train.optimize_latent_code')
        self.optimize_scene_latent_code = self.conf.get_bool('train.optimize_scene_latent_code')

        self.optimize_inputs = self.optimize_expression or self.optimize_pose

        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=1,# min(int(self.conf.get_int('train.max_points_training') /self.model.pc.points.shape[0]),self.conf.get_int('train.max_batch',default='10')),
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn
                                                           )
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                                           batch_size=1,# min(int(self.conf.get_int('train.max_points_training') /self.model.pc.points.shape[0]),self.conf.get_int('train.max_batch',default='10')),
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn
                                                           )
        
        self.optimize_tracking = False
        if self.optimize_inputs:
            print('[DEBUG] Optimizing input parameters ...')
            self.input_params_subdir = "TestInputParameters"
            test_input_params = []
            if self.optimize_expression:
                init_expression = self.plot_dataset.data["expressions"]

                self.expression = torch.nn.Embedding(len(self.plot_dataset), self.model.deformer_network.num_exp, _weight=init_expression, sparse=True).cuda()
                test_input_params += list(self.expression.parameters())

            if self.optimize_pose:
                self.flame_pose = torch.nn.Embedding(len(self.plot_dataset), 15,
                                                     _weight=self.plot_dataset.data["flame_pose"],
                                                     sparse=True).cuda()
                self.camera_pose = torch.nn.Embedding(len(self.plot_dataset), 3,
                                                      _weight=self.plot_dataset.data["world_mats"][:, :3, 3],
                                                      sparse=True).cuda()
                test_input_params += list(self.flame_pose.parameters()) + list(self.camera_pose.parameters())
            
            self.optimizer_cam = torch.optim.SparseAdam(test_input_params,
                                                        self.conf.get_float('train.learning_rate_cam'))
            
            try:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.input_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                if self.optimize_expression:
                    self.expression.load_state_dict(data["expression_state_dict"])
                if self.optimize_pose:
                    self.flame_pose.load_state_dict(data["flame_pose_state_dict"])
                    self.camera_pose.load_state_dict(data["camera_pose_state_dict"])
                print('Using pre-tracked test expressions')

            except:
                self.optimize_tracking = True
                # from model.loss import Loss, Loss_lightning_singeGPU
                # # self.loss = Loss(mask_weight=0.0)
                # self.loss = Loss_lightning_singeGPU(mask_weight=0.0)
                # print('Optimizing test expressions')

        # self.optimize_shape_blendshapes = True
        # if self.optimize_shape_blendshapes:
        #     # self.loss = Loss_lightning_singeGPU(mask_weight=0.0)
        #     self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(mask_weight=0.0, lbs_weight=10.0)
        #     print('Optimizing test shape blendshapes')
        #     self.lr = self.conf.get_float('train.learning_rate')
        #     self.optimizer = torch.optim.Adam([{'params': list(self.model.deformer_network.parameters())},], lr=self.lr)

        # hyunsoo added
        if self.optimize_latent_code:
            if self.latent_dim == 0:
                raise ValueError('The latent dimension cannot be 0. Please double check the config file.')
            data = torch.load(
                os.path.join(old_checkpnts_dir, "InputParameters", str(kwargs['checkpoint']) + ".pth"))
            self.latent_codes = torch.nn.Embedding(data["latent_codes_state_dict"]['weight'].shape[0], self.latent_dim, sparse=True).cuda()
            self.latent_codes.load_state_dict(data["latent_codes_state_dict"])
            print('[DEBUG] Latent code is used. The latent dimension is {0}x{1}.'.format(data["latent_codes_state_dict"]['weight'].shape[0], self.latent_dim))
        
        if self.optimize_scene_latent_code:
            data = torch.load(
                os.path.join(old_checkpnts_dir, "InputParameters", str(kwargs['checkpoint']) + ".pth"), map_location=lambda storage, loc: storage)
            self.zero_latent_codes = torch.nn.Embedding(1, self.scene_latent_dim, sparse=True).cuda()
            self.zero_latent_codes.load_state_dict(data["zero_latent_codes_state_dict"])

            if self.target_training:
                self.target_scene_latent_codes = torch.nn.Embedding(len(self.target_category_dict.keys()), self.scene_latent_dim, sparse=True).cuda()
                try:
                    self.target_scene_latent_codes.load_state_dict(data["target_scene_latent_codes_state_dict"])  
                except:
                    self.target_scene_latent_codes.load_state_dict(data["target_scene_latent_category_codes_state_dict"])
            
            elif self.multi_source_training:
                # NOTE Source Human (Guy)를 위한 latent code. [7, 32]
                self.source_scene_latent_codes = torch.nn.Embedding(len(self.source_category_dict.keys()), self.scene_latent_dim, sparse=True).cuda()
                self.source_scene_latent_codes.load_state_dict(data["source_scene_latent_codes_state_dict"])

                # NOTE Multi Source를 위한 latent code. [7, 32]*21
                self.multi_source_without_hat_scene_latent_codes = []
                for idx, subdir in enumerate(self.no_hat_dataset_train_subdir):
                    multi_source_without_hat_scene_latent_codes = torch.nn.Embedding(len(self.without_hat_category_dict.keys()), self.scene_latent_dim, sparse=True).cuda()
                    multi_source_without_hat_scene_latent_codes.load_state_dict(data["multi_source_without_hat_scene_latent_codes_{}_state_dict".format(subdir)])
                    self.multi_source_without_hat_scene_latent_codes.append(multi_source_without_hat_scene_latent_codes)

                self.multi_source_with_hat_scene_latent_codes = []
                for idx, subdir in enumerate(self.hat_dataset_train_subdir):
                    multi_source_with_hat_scene_latent_codes = torch.nn.Embedding(len(self.with_hat_category_dict.keys()), self.scene_latent_dim, sparse=True).cuda()
                    multi_source_with_hat_scene_latent_codes.load_state_dict(data["multi_source_with_hat_scene_latent_codes_{}_state_dict".format(subdir)])
                    self.multi_source_with_hat_scene_latent_codes.append(multi_source_with_hat_scene_latent_codes)

            else:
                self.scene_latent_codes = torch.nn.Embedding(data["scene_latent_codes_state_dict"]['weight'].shape[0], self.scene_latent_dim, sparse=True).cuda()
                self.scene_latent_codes.load_state_dict(data["scene_latent_codes_state_dict"])
                self.source_scene_latent_codes = torch.nn.Embedding(data["source_scene_latent_codes_state_dict"]['weight'].shape[0], self.scene_latent_dim, sparse=True).cuda()
                self.source_scene_latent_codes.load_state_dict(data["source_scene_latent_codes_state_dict"])
            
        self.img_res = self.plot_dataset.img_res

        # NOTE For Test Settings ################################
        def extract_category(s):
            return s.split('_')[0]
        
        self.conf.put('dataset.test.subsample', 1)
        self.conf.put('dataset.test.load_images', False)
        self.dataset_test_subdir = self.conf.get_list('dataset.test.sub_dir')
        self.acc_loss = {}

        self.test_epoch = saved_model_state['epoch']
        print('[INFO] Loading checkpoint from {0} epoch'.format(self.test_epoch))
        
        # self.test_normal_only = self.conf.get_bool('test.normal_only')

        self.test_default_rendering = self.conf.get_bool('test.default_rendering')
        self.test_random_sampling = self.conf.get_bool('test.random_sampling')
        self.test_target_default_rendering = self.conf.get_bool('test.target_default_rendering')
        self.test_scene_latent_interpolation = self.conf.get_bool('test.scene_latent_interpolation')
        self.test_scene_latent_interpolation_category = self.conf.get_string('test.scene_latent_interpolation_category')
        self.test_multi_composition = self.conf.get_bool('test.multi_composition')
        self.test_multi_composition_list = sorted(self.conf.get_list('test.multi_composition_list'), key=lambda x: self.category_dict[extract_category(x)])
        self.test_target_human_blending = self.conf.get_bool('test.target_human.blending')
        self.test_target_interpolation = self.conf.get_bool('test.target_human.target_interpolation')
        self.test_target_blending_default_rendering = self.conf.get_bool('test.target_human.blending_default_rendering')
        if not self.test_target_human_blending:
            self.test_target_blending_default_rendering = False
        self.dataset_target_train_subdir = self.conf.get_list('dataset.train.sub_dir')
        if self.test_target_human_blending:
            # NOTE target human이 torch로 만든 경우에 불러오는 코드이다.
            self.test_target_human_conf = ConfigFactory.parse_file(self.conf.get_string('test.target_human.conf'))
            self.test_target_category = self.conf.get_string('test.target_human.target_category')
            test_exps_folder_name = self.test_target_human_conf.get_string('train.exps_folder')
            test_subject = self.test_target_human_conf.get_string('dataset.subject_name')
            test_methodname = self.test_target_human_conf.get_string('train.methodname')
            test_expdir = os.path.join(test_exps_folder_name, test_subject, test_methodname)
            self.test_train_split_name = utils.get_split_name(self.test_target_human_conf.get_list('dataset.train.sub_dir'))
            self.test_target_human_path = os.path.join(test_expdir, self.test_train_split_name, 'train', 'checkpoints')

            self.test_target_human_epoch_num = self.conf.get_string('test.target_human.epoch_num')

            target_human_path = os.path.join(self.test_target_human_path, 'ModelParameters', '{}.pth'.format(self.test_target_human_epoch_num))
            saved_model_state = torch.load(target_human_path, map_location=lambda storage, loc: storage)

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
                                                                                                                latent_code_dim=latent_code_dim,
                                                                                                                pcd_init=th_pcd_init).cuda()
            n_points = saved_model_state["model_state_dict"]['pc.points'].shape[0]
            self.test_target_human_model.pc.init(n_points)
            self.test_target_human_model.pc = self.test_target_human_model.pc.cuda()
            self.test_target_human_model.raster_settings.radius = saved_model_state['radius']
            self.test_target_human_model.load_state_dict(saved_model_state["model_state_dict"])
            self.test_target_human_model.training = False           # NOTE 이거 안하면 visible문제가 생김.

            saved_input_state = torch.load(os.path.join(self.test_target_human_path, 'InputParameters', '{}.pth'.format(self.test_target_human_epoch_num)), map_location=lambda storage, loc: storage)

            self.test_target_zero_latent_codes = torch.nn.Embedding(1, self.scene_latent_dim, sparse=True).cuda()
            self.test_target_zero_latent_codes.load_state_dict(saved_input_state["zero_latent_codes_state_dict"])

            self.test_target_scene_latent_codes = torch.nn.Embedding(len(self.target_category_dict.keys()), self.scene_latent_dim, sparse=True).cuda()
            try:
                self.test_target_scene_latent_codes.load_state_dict(saved_input_state["target_scene_latent_codes_state_dict"])
            except:
                self.test_target_scene_latent_codes.load_state_dict(saved_input_state["target_scene_latent_category_codes_state_dict"])

            # try:
            #     self.test_target_scene_latent_codes = torch.nn.Embedding(1, self.scene_latent_dim, sparse=True).cuda()
            #     self.test_target_scene_latent_codes.load_state_dict(saved_input_state["target_scene_latent_codes_state_dict"])
            # except:
            #     self.test_target_scene_latent_codes = torch.nn.Embedding(len(self.target_category_dict.keys()), self.scene_latent_dim, sparse=True).cuda()
            #     self.test_target_scene_latent_codes.load_state_dict(saved_input_state["target_scene_latent_codes_state_dict"])
                
            # try:
            #     self.test_target_scene_latent_category_codes = torch.nn.Embedding(len(self.target_category_dict.keys()), self.scene_latent_dim, sparse=True).cuda()
            #     self.test_target_scene_latent_category_codes.load_state_dict(saved_input_state["target_scene_latent_category_codes_state_dict"])  #  target_scene_latent_category_codes_state_dict 개정 전에는 이걸 썼었다. 이게 지금 pretrained model마다 다르네 ㅠㅠ
            # except:
            #     print('[INFO] no target_scene_latent_category_codes_state_dict')

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

            self.test_target_human_model_free_memory = copy.copy(self.test_target_human_model)

        self.model_free_memory = copy.copy(self.model)
        # self.test_target_finetuning = self.conf.get_bool('test.target_finetuning')
        # self.test_target_finetuning_list = sorted(self.conf.get_list('test.target_finetuning_list'), key=lambda x: self.category_dict[extract_category(x)])
        # self.test_db_sub_dir = self.conf.get_list('test.db_sub_dir')
        
        self.test_interpolation_step_size = 45
        #########################################################

    
    def save_test_optimization(self, epoch):
        if not os.path.exists(os.path.join(self.checkpoints_path, "TestInputParameters")):
            os.mkdir(os.path.join(self.checkpoints_path, "TestInputParameters"))
        
        dict_to_save = {}
        dict_to_save["epoch"] = epoch
        dict_to_save["chamfer_translation_state_dict"] = self.chamfer_translation.state_dict()
        torch.save(dict_to_save, os.path.join(self.checkpoints_path, "TestInputParameters", str(epoch) + ".pth"))
        torch.save(dict_to_save, os.path.join(self.checkpoints_path, "TestInputParameters", "latest.pth"))
        dict_to_save = {}
        dict_to_save["epoch"] = epoch
        dict_to_save["model_state_dict"] = self.model.state_dict()
        torch.save(dict_to_save, os.path.join(self.checkpoints_path, "TestInputParameters", str(epoch) + ".pth"))
        torch.save(dict_to_save, os.path.join(self.checkpoints_path, "TestInputParameters", "latest.pth"))


    def save_test_tracking(self, epoch):
        if not os.path.exists(os.path.join(self.checkpoints_path, "TestInputParameters")):
            os.mkdir(os.path.join(self.checkpoints_path, "TestInputParameters"))
        if self.optimize_inputs:
            dict_to_save = {}
            dict_to_save["epoch"] = epoch
            if self.optimize_expression:
                dict_to_save["expression_state_dict"] = self.expression.state_dict()
            if self.optimize_pose:
                dict_to_save["flame_pose_state_dict"] = self.flame_pose.state_dict()
                dict_to_save["camera_pose_state_dict"] = self.camera_pose.state_dict()
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, "TestInputParameters", str(epoch) + ".pth"))
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, "TestInputParameters", "latest.pth"))
        dict_to_save = {}
        dict_to_save["epoch"] = epoch
        dict_to_save["model_state_dict"] = self.model.state_dict()
        torch.save(dict_to_save, os.path.join(self.checkpoints_path, "TestInputParameters", str(epoch) + ".pth"))
        torch.save(dict_to_save, os.path.join(self.checkpoints_path, "TestInputParameters", "latest.pth"))
        

    # hyunsoo added
    def novelView(self, cam_pose, euler_angle, translation):
        ##############################################
        # euler_angle should be tuple type
        x_euler, y_euler, z_euler = euler_angle
        ##############################################

        ## for novel view synthesis
        cam_R = cam_pose[:, :3, :3]
        cam_t = cam_pose[:, :3, -1]
        cam_t = cam_t.reshape(-1, 1).unsqueeze(0)
        # world_R, world_t = camtoworld_to_worldtocam_Rt(cam_R, cam_t) # IMavatar legacy code
        world_R, world_t = cam_R, cam_t
        
        rotate_world_R = quaternion_to_rotation_matrix(rotation_matrix_to_quaternion(world_R)+torch.cat(quaternion_from_euler(x_euler, y_euler, z_euler)).unsqueeze(0))
        trans_world_t = world_t + translation.reshape(-1, 1).unsqueeze(0)

        # rotate_cam_R, rotate_cam_t = worldtocam_to_camtoworld_Rt(rotate_world_R, trans_world_t) # IMavatar legacy code
        rotate_cam_R, rotate_cam_t = rotate_world_R, trans_world_t
        cam_pose = Rt_to_matrix4x4(rotate_cam_R, rotate_cam_t)
        ##############################################
        return cam_pose

    def linear_interpolation(self, x1, x2, ratio):
        return x1 + ratio * (x2 - x1)

    def latent_factory(self, model_input, train_subdir, scene_latent_codes_tensor):
        learnable_tensor_size = self.category_latent_dim + self.scene_latent_dim
        input_latent_codes = torch.zeros(len(model_input['sub_dir']), learnable_tensor_size*len(self.category_dict))

        # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
        for i, v in self.category_dict.items():
            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
            start_idx = v*learnable_tensor_size
            end_idx = (v+1)*learnable_tensor_size
            input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                
        # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
        for i, v in enumerate(self.source_category_dict.values()):
            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
            source_start_idx = v*learnable_tensor_size
            source_end_idx = (v+1)*learnable_tensor_size
            input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

        index_list = []
        for sub_dir_item in [train_subdir]:
            category_idx = self.category_dict[sub_dir_item.split('_')[0]]
            index_list.append(category_idx)
        category_latent_codes = F.one_hot(torch.tensor(index_list), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(index_list).cuda())
        scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

        # NOTE 마지막으로 db human의 latent code를 넣어준다.
        for i in range(len(index_list)):
            start_idx = index_list[i]*learnable_tensor_size
            end_idx = (index_list[i]+1)*learnable_tensor_size
            input_latent_codes[i, start_idx:end_idx] = scene_latent_codes_tensor[i]

        return input_latent_codes


    def run(self):
        self.model.eval()
        self.model.training = False

        eval_all = True
        is_first_batch = True
        
        # if self.test_target_blending_default_rendering:
        if self.test_target_human_blending:
            # NOTE pnts_c_flame에서 deformation을 시키고 나머지 sequence에 대해서는 그걸 바탕으로 쓰도록 수정.
            for batch_idx, (indices, model_input, ground_truth) in enumerate(self.plot_dataloader):
                if model_input['img_name'].item() >= 1:
                    break
                
                if model_input['img_name'].item() >= 68:
                    break
                
            device = 'cuda'
            batch_size = model_input['expression'].shape[0]
            for k, v in model_input.items():
                try:
                    model_input[k] = v.to(device)
                except:
                    model_input[k] = v

            for k, v in ground_truth.items():
                try:
                    ground_truth[k] = v.to(device)
                except:
                    ground_truth[k] = v

            if self.optimize_inputs:
                if self.optimize_expression:
                    model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                if self.optimize_pose:
                    model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                    model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)
            
            is_first_batch = True

            # shape_learnable = torch.nn.Embedding(model_input['shape'].shape[0], model_input['shape'].shape[1], sparse=False).cuda()
            # shape_learnable.weight.data = model_input['shape'].cuda()
            # param = []
            # param += list(shape_learnable.parameters())
            # shape_bak = model_input['shape'].clone()

            translation_learnable = torch.nn.Embedding(1, 3, sparse=False).cuda()
            torch.nn.init.zeros_(translation_learnable.weight.data)
            param = []
            param += list(translation_learnable.parameters())

            rotation_learnable = torch.nn.Embedding(1, 3, sparse=False).cuda() # angle axis
            torch.nn.init.zeros_(rotation_learnable.weight.data)
            param += list(rotation_learnable.parameters())
        
            optimizer = torch.optim.Adam(param, lr=0.01) # 0.01은 매 iteration마다 landmark를 잡았을 때 잘 되었던 경우이다. 하지만 아예 다른곳으로 가도 잘되는 현상이 있었다. 

            learnable_tensor_size = self.category_latent_dim + self.scene_latent_dim
            input_latent_codes = torch.zeros(len(model_input['sub_dir']), learnable_tensor_size*len(self.category_dict))

            # NOTE zero scene latent로 일단 다 초기화한다.
            for i, v in self.category_dict.items():
                category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).detach() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                start_idx = v*learnable_tensor_size
                end_idx = (v+1)*learnable_tensor_size
                input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.test_target_zero_latent_codes(torch.tensor(0).cuda())), dim=0)

            target_index_list = []
            target_category_latent_codes = []
            for i, v in self.target_category_dict.items():
                target_index_list.append(v)
                tensor = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).unsqueeze(0) # self.category_latent_codes(torch.tensor(v).cuda()).detach().unsqueeze(0) # int_to_tensor(v).unsqueeze(0)
                target_category_latent_codes.append(tensor)
            target_category_latent_codes = torch.cat(target_category_latent_codes, dim=0).cuda().detach()

            for i, v in enumerate(target_index_list):
                target_start_idx = v*learnable_tensor_size
                target_end_idx = (v+1)*learnable_tensor_size
                # input_latent_codes[:, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.test_target_scene_latent_category_codes(torch.tensor(i).cuda())), dim=0)
                input_latent_codes[:, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.test_target_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

            # Add to the model_input dictionary
            model_input['scene_latent_code'] = input_latent_codes.cuda()
            # model_input['indices_tensor'] = 0
            model_input['target_human_values'] = True
            model_input['category'] = self.test_target_category
            model_input['rotation'] = None
            model_input['translation'] = None
            
            with torch.set_grad_enabled(True):
                middle_inference = self.test_target_human_model(model_input)

            for k, v in middle_inference.items():
                try:
                    middle_inference[k] = v.detach().cpu()
                except:
                    middle_inference[k] = v

            model_input['middle_inference'] = middle_inference
            del model_input['scene_latent_code'], model_input['target_human_values'] #  model_input['masked_point_cloud_indices'], model_input['indices_tensor'],
            del self.test_target_human_model
            self.test_target_human_model = copy.copy(self.test_target_human_model_free_memory)
            gc.collect()
            torch.cuda.empty_cache()


            # NOTE sh와 th를 optimization을 통해 서로 붙여준다.
            test_category = self.test_target_category
            
            category_subdir_index = [index for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
            category_subdir = [element for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]

            index_nn_ = None
            idx = 11    # 11이 좀 잘된다.            # 뭘로 돌리는지는 사실 중요하지 않고 어떤 sequence로 확인하냐의 뜻이다.
            iteration = 100
            for epoch in range(iteration):
                scene_latent_codes_tensor = self.scene_latent_codes(torch.tensor([idx]).to(model_input['idx'].device)).squeeze(1).detach() 
                train_subdir = self.dataset_train_subdir[idx]

                model_input['indices_tensor'] = idx
                model_input['masked_point_cloud_indices'] = {}
                model_input['scene_latent_code'] = self.latent_factory(model_input, train_subdir, scene_latent_codes_tensor).to(device)  # interpolated_latent(frame/interpolation_frames).to(device)
                model_input['chamfer_loss'] = True
                # model_input['shape'] = shape_learnable(torch.tensor([0]).to(model_input['idx'].device))
                model_input['rotation'] = rotation_learnable(torch.tensor([0]).to(model_input['idx'].device))
                model_input['translation'] = translation_learnable(torch.tensor([0]).to(model_input['idx'].device))
                model_input['index_nn'] = index_nn_
                model_input['epoch'] = epoch

                # with torch.set_grad_enabled(True):
                self.model.training = False
                loss, index_nn = self.model(model_input)

                index_nn_ = index_nn

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('[INFO] epoch: {}, loss: {}'.format(epoch, loss))

            del self.model, model_input, indices, ground_truth
            self.model = copy.copy(self.model_free_memory)
            gc.collect()
            torch.cuda.empty_cache()
    

        eval_iterator = iter(self.plot_dataloader)

        for img_index in tqdm(range(len(self.plot_dataloader)), desc='[INFO] Rendering...'):
            indices, model_input, ground_truth = next(eval_iterator)

            device = 'cuda'

            batch_size = model_input['expression'].shape[0]
            for k, v in model_input.items():
                try:
                    model_input[k] = v.to(device)
                except:
                    model_input[k] = v

            for k, v in ground_truth.items():
                try:
                    ground_truth[k] = v.to(device)
                except:
                    ground_truth[k] = v

            if self.optimize_inputs:
                if self.optimize_expression:
                    model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                if self.optimize_pose:
                    model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                    model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)
            
            is_first_batch = True

            # if model_input['sub_dir'][0] == 'target_Yufeng':
            #     model_input['cam_pose'][-1, -1, -1] += 1.5

            if False:
                front_camera_matrix = model_input['cam_pose'].squeeze(0)
                # Rotation matrices for 90 and 180 degrees around the Y-axis
                rotation_90 = torch.tensor([
                    [0, 0, 1],
                    [0, 1, 0],
                    [-1, 0, 0]
                ]).to(front_camera_matrix.device).float()

                rotation_180 = torch.tensor([
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1]
                ]).to(front_camera_matrix.device).float()
                # Compute back, left, and right camera matrices
                back_camera_matrix = torch.mm(rotation_180, front_camera_matrix[:, :3])
                left_camera_matrix = torch.mm(rotation_90, front_camera_matrix[:, :3])
                right_camera_matrix = torch.mm(rotation_90.T, front_camera_matrix[:, :3])  # Transpose for -90 degrees

                # Add the translation component back
                back_camera_matrix = torch.cat((back_camera_matrix, front_camera_matrix[:, 3:]), 1)
                left_camera_matrix = torch.cat((left_camera_matrix, front_camera_matrix[:, 3:]), 1)
                right_camera_matrix = torch.cat((right_camera_matrix, front_camera_matrix[:, 3:]), 1)

                model_input['cam_pose'] = back_camera_matrix.unsqueeze(0)

            # NOTE camera 변경을 하였다.

            if False:
                for batch_index, train_subdir in enumerate(self.dataset_train_subdir):
                    novel_view_type = 'default_rendering_{}'.format(train_subdir)
                    plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                    img_names = model_input['img_name'][:,0].cpu().numpy()[0]

                    scene_latent_codes_tensor = self.scene_latent_codes(torch.LongTensor([batch_index]).to(device)).squeeze(1).detach()     # [1, 28]

                    index_list = []
                    for sub_dir_item in [train_subdir]:
                        category_idx = self.category_dict[sub_dir_item.split('_')[0]]
                        index_list.append(category_idx)
                    category_latent_codes = F.one_hot(torch.tensor(index_list), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(index_list).cuda())
                    scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

                    # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. [B, 320]
                    input_latent_codes = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))
                    
                    # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                    for i, v in self.category_dict.items():
                        category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                        start_idx = v*scene_latent_codes_tensor.shape[1]
                        end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                        input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                            
                    # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                    for i, v in enumerate(self.source_category_dict.values()):
                        category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                        source_start_idx = v*scene_latent_codes_tensor.shape[1]
                        source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                        input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                    # NOTE 마지막으로 db human의 latent code를 넣어준다.
                    # for i in range(len(index_list)):
                    #     start_idx = index_list[i]*scene_latent_codes_tensor.shape[1]
                    #     end_idx = (index_list[i]+1)*scene_latent_codes_tensor.shape[1]
                    #     input_latent_codes[i, start_idx:end_idx] = scene_latent_codes_tensor[i]
                    
                    model_input['scene_latent_code'] = input_latent_codes.cuda()

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


            if self.test_default_rendering:
                for batch_index, train_subdir in enumerate(self.dataset_train_subdir):
                    # if train_subdir == 'hat_Syuka_crimsonhat' or train_subdir == 'hat_Syuka_foxhat' or train_subdir == 'hat_Syuka_pinkhat' or train_subdir == 'hat_Syuka_santahat':
                    # if train_subdir in ['hair_Chloe_Grace_Moretz', 'hair_John_Krasinski', 'hair_LaurDIY', 'hair_LILHUDDY', 'hair_Malcolm_Gladwell']:
                    if train_subdir in 'hair_{}'.format(model_input['sub_dir'][0]):
                        novel_view_type = 'default_rendering_{}'.format(train_subdir)
                        plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                        img_names = model_input['img_name'][:,0].cpu().numpy()[0]

                        # if img_names <= 100:
                        #     break

                        scene_latent_codes_tensor = self.scene_latent_codes(torch.LongTensor([batch_index]).to(device)).squeeze(1).detach()     # [1, 28]

                        index_list = []
                        for sub_dir_item in [train_subdir]:
                            category_idx = self.category_dict[sub_dir_item.split('_')[0]]
                            index_list.append(category_idx)
                        category_latent_codes = F.one_hot(torch.tensor(index_list), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(index_list).cuda())
                        scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

                        # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. [B, 320]
                        input_latent_codes = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))
                        
                        # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                        for i, v in self.category_dict.items():
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                            start_idx = v*scene_latent_codes_tensor.shape[1]
                            end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                            input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                                
                        # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                        for i, v in enumerate(self.source_category_dict.values()):
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                            source_start_idx = v*scene_latent_codes_tensor.shape[1]
                            source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                            input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                        # NOTE 마지막으로 db human의 latent code를 넣어준다.
                        for i in range(len(index_list)):
                            start_idx = index_list[i]*scene_latent_codes_tensor.shape[1]
                            end_idx = (index_list[i]+1)*scene_latent_codes_tensor.shape[1]
                            input_latent_codes[i, start_idx:end_idx] = scene_latent_codes_tensor[i]
                        
                        model_input['scene_latent_code'] = input_latent_codes.cuda()

                        with torch.set_grad_enabled(True):
                            model_outputs = self.model(model_input)

                        if model_outputs is not None:
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
                        if img_names >= 1000:
                            break

            
            if self.test_random_sampling:
                def find_indexes(my_list, element):
                    indexes = []
                    for i, item in enumerate(my_list):
                        if element in item:
                            indexes.append(i)
                    return indexes

                how_many_sample = 1
                for batch_index, train_subdir in enumerate(self.dataset_train_subdir):
                    # if train_subdir == 'nose_LILHUDDY' and model_input['img_name'].item() in [610]:
                    if model_input['img_name'].item() < 51:
                        for j in range(how_many_sample):
                            # novel_view_type = 'random_sampling_{}'.format(train_subdir)
                            novel_view_type = 'random_sampling_{}'.format('dirs')
                            plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                            # img_names = np.array('{}_{}'.format(model_input['img_name'][:,0].cpu().numpy()[0], j))
                            img_names = np.array('{}_{}'.format(model_input['img_name'][:,0].cpu().numpy()[0], train_subdir))

                            scene_latent_codes_tensor = self.scene_latent_codes(torch.LongTensor([batch_index]).to(device)).squeeze(1).detach()     # [1, 28]

                            if False: # NOTE zero mean and std
                                scene_latent_codes_tensor = torch.zeros(scene_latent_codes_tensor.size()).cuda()

                            mask_identity_rendering = False

                            sample_mean_std = False
                            total_mean_std = True
                            standard_normal_tensor = torch.randn(scene_latent_codes_tensor.size()).cuda()    # torch.Size([1, 32])

                            if sample_mean_std: # NOTE sample mean and std
                                scene_latent_codes_mean = scene_latent_codes_tensor.mean()
                                scene_latent_codes_std = scene_latent_codes_tensor.std()
                                scene_latent_codes_tensor = torch.normal(scene_latent_codes_mean, scene_latent_codes_std, size=scene_latent_codes_tensor.size()).cuda()

                            if total_mean_std:
                                category = train_subdir.split('_')[0]
                                indexes = find_indexes(self.dataset_train_subdir, category)
                                scene_latent_codes_tensor = self.scene_latent_codes(torch.LongTensor(indexes).to(device)).squeeze(1).detach()

                                scene_latent_codes_mean = scene_latent_codes_tensor.mean(dim=0)              # torch.Size([32])
                                scene_latent_codes_std = scene_latent_codes_tensor.std(dim=0)              # torch.Size([32])
                                scene_latent_codes_tensor = standard_normal_tensor * scene_latent_codes_std + scene_latent_codes_mean

                            index_list = []
                            for sub_dir_item in [train_subdir]:
                                category_idx = self.category_dict[sub_dir_item.split('_')[0]]
                                index_list.append(category_idx)
                            category_latent_codes = F.one_hot(torch.tensor(index_list), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(index_list).cuda())
                            scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

                            # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. [B, 320]
                            input_latent_codes = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))
                            
                            # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                            for i, v in self.category_dict.items():
                                category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                                start_idx = v*scene_latent_codes_tensor.shape[1]
                                end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                                input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                                    
                            # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                            for i, v in enumerate(self.source_category_dict.values()):
                                category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                                source_start_idx = v*scene_latent_codes_tensor.shape[1]
                                source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                                input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)
                            
                            if mask_identity_rendering:
                                scene_latent_code_default = input_latent_codes.clone()

                            # NOTE 마지막으로 db human의 latent code를 넣어준다.
                            for i in range(len(index_list)):
                                start_idx = index_list[i]*scene_latent_codes_tensor.shape[1]
                                end_idx = (index_list[i]+1)*scene_latent_codes_tensor.shape[1]
                                input_latent_codes[i, start_idx:end_idx] = scene_latent_codes_tensor[i]
                            
                            #####################################################################
                            model_input['scene_latent_code'] = input_latent_codes.cuda()
                            if mask_identity_rendering:
                                model_input['scene_latent_code_default'] = scene_latent_code_default.cuda()
                                # model_input['scene_latent_code'] = scene_latent_code_default.cuda()
                                model_input['generate_mask_identity'] = True

                                with torch.set_grad_enabled(True):
                                    mask_identity = self.model(model_input)
                                
                                del self.model
                                self.model = copy.copy(self.model_free_memory)
                                self.model.training = False
                                gc.collect()
                                torch.cuda.empty_cache()

                                del model_input['generate_mask_identity']

                                model_input['mask_identity'] = mask_identity
                            #####################################################################
                            # model_input['scene_latent_code'] = input_latent_codes.cuda()
                            # model_input['scene_latent_code_default'] = scene_latent_code_default.cuda()

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
            

            if self.test_default_rendering and (1 <= model_input['img_name'].item() <= 45) and False:
                for batch_index, train_subdir in enumerate(self.dataset_train_subdir):
                    if train_subdir == 'eyebrows_Chloe_Grace_Moretz':
                        novel_view_type = 'default_rendering_{}'.format(train_subdir)
                        plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                        img_names = model_input['img_name'][:,0].cpu().numpy()[0]

                        scene_latent_codes_tensor = self.scene_latent_codes(torch.LongTensor([batch_index]).to(device)).squeeze(1).detach()     # [1, 28]

                        index_list = []
                        for sub_dir_item in [train_subdir]:
                            category_idx = self.category_dict[sub_dir_item.split('_')[0]]
                            index_list.append(category_idx)
                        category_latent_codes = F.one_hot(torch.tensor(index_list), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(index_list).cuda())
                        scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

                        # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. [B, 320]
                        input_latent_codes = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))
                        
                        # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                        for i, v in self.category_dict.items():
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                            start_idx = v*scene_latent_codes_tensor.shape[1]
                            end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                            input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                                
                        # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                        for i, v in enumerate(self.source_category_dict.values()):
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                            source_start_idx = v*scene_latent_codes_tensor.shape[1]
                            source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                            input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                        # NOTE 마지막으로 db human의 latent code를 넣어준다.
                        for i in range(len(index_list)):
                            start_idx = index_list[i]*scene_latent_codes_tensor.shape[1]
                            end_idx = (index_list[i]+1)*scene_latent_codes_tensor.shape[1]
                            input_latent_codes[i, start_idx:end_idx] = scene_latent_codes_tensor[i]
                        
                        model_input['scene_latent_code'] = input_latent_codes.cuda()

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

                        plt.plot(90-img_names,
                                model_outputs,
                                ground_truth,
                                plot_dir,
                                self.test_epoch,
                                self.img_res,
                                is_eval=eval_all,
                                first=is_first_batch,
                                custom_settings={'novel_view': novel_view_type})
                        
                        del model_outputs



            if self.multi_source_training and False:
                for batch_index, train_subdir in enumerate(self.dataset_train_subdir):
                    novel_view_type = 'default_multi_source_rendering_{}'.format(train_subdir)
                    plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                    img_names = model_input['img_name'][:,0].cpu().numpy()[0]

                    learnable_tensor_size = self.category_latent_dim + self.scene_latent_dim

                    input_latent_codes = torch.zeros(len(model_input['sub_dir']), learnable_tensor_size*len(self.category_dict))

                    if train_subdir in self.no_hat_dataset_train_subdir:
                        # NOTE latent combination
                        for i, v in self.category_dict.items():
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                            start_idx = v* learnable_tensor_size
                            end_idx = (v+1)* learnable_tensor_size
                            input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                        
                        for i, v in enumerate(self.source_category_dict.values()):
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                            source_start_idx = v* learnable_tensor_size
                            source_end_idx = (v+1)* learnable_tensor_size
                            input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                        # NOTE hair에 대해서만 찍어보기.
                        indices = [self.no_hat_dataset_train_subdir.index(name) for name in [train_subdir]]
                        multi_source_without_hat_scene_latent_codes = self.multi_source_without_hat_scene_latent_codes[indices[0]]

                        for i, v in enumerate(self.without_hat_category_dict.values()):
                            if i == 4:      # NOTE hair
                                category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                                source_start_idx = v* learnable_tensor_size
                                source_end_idx = (v+1)* learnable_tensor_size
                                input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, multi_source_without_hat_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                        model_input['scene_latent_code'] = input_latent_codes.cuda()

                    elif train_subdir in self.hat_dataset_train_subdir:
                        # NOTE latent combination
                        for i, v in self.category_dict.items():
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                            start_idx = v* learnable_tensor_size
                            end_idx = (v+1)* learnable_tensor_size
                            input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                        
                        for i, v in enumerate(self.source_category_dict.values()):
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                            source_start_idx = v* learnable_tensor_size
                            source_end_idx = (v+1)* learnable_tensor_size
                            input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                        indices = [self.hat_dataset_train_subdir.index(name) for name in [train_subdir]]
                        multi_source_with_hat_scene_latent_codes = self.multi_source_with_hat_scene_latent_codes[indices[0]]

                        for i, v in enumerate(self.without_hat_category_dict.values()):
                            if i == 4:
                                category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                                source_start_idx = v* learnable_tensor_size
                                source_end_idx = (v+1)* learnable_tensor_size
                                input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, multi_source_without_hat_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                        model_input['scene_latent_code'] = input_latent_codes.cuda()

                        # # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                        # for i, v in self.category_dict.items():
                        #     category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                        #     start_idx = v* learnable_tensor_size
                        #     end_idx = (v+1)* learnable_tensor_size
                        #     input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                            
                        # # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                        # for i, v in enumerate(self.with_hat_category_dict.values()):
                        #     category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                        #     source_start_idx = v* learnable_tensor_size
                        #     source_end_idx = (v+1)* learnable_tensor_size
                        #     input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, multi_source_with_hat_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                        # model_input['scene_latent_code'] = input_latent_codes.cuda()

                    else:
                        continue

                    # indices = [self.hat_dataset_train_subdir.index(name) for name in [train_subdir]]
                    # multi_source_with_hat_scene_latent_codes = self.multi_source_with_hat_scene_latent_codes[indices[0]]

                    # for i, v in enumerate(self.with_hat_category_dict.values()):
                    #     category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                    #     source_start_idx = v* learnable_tensor_size
                    #     source_end_idx = (v+1)* learnable_tensor_size
                    #     input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, multi_source_with_hat_scene_latent_codes(torch.tensor(i).cuda())), dim=0)


                    # # NOTE default rendering for synthesis experiments.
                    # if train_subdir == self.source_datasets[0]:
                    #     # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                    #     for i, v in self.category_dict.items():
                    #         category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                    #         start_idx = v* learnable_tensor_size
                    #         end_idx = (v+1)* learnable_tensor_size
                    #         input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                            
                    #     # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                    #     for i, v in enumerate(self.source_category_dict.values()):
                    #         category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                    #         source_start_idx = v* learnable_tensor_size
                    #         source_end_idx = (v+1)* learnable_tensor_size
                    #         input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)
                        
                    #     # Add to the model_input dictionary
                    #     model_input['scene_latent_code'] = input_latent_codes.cuda()


                    # elif train_subdir in self.hat_dataset_train_subdir:            # NOTE hat가 있는 dataset.
                    #     indices = [self.hat_dataset_train_subdir.index(name) for name in [train_subdir]]
                    #     multi_source_with_hat_scene_latent_codes = self.multi_source_with_hat_scene_latent_codes[indices[0]]

                    #     # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                    #     for i, v in self.category_dict.items():
                    #         category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                    #         start_idx = v* learnable_tensor_size
                    #         end_idx = (v+1)* learnable_tensor_size
                    #         input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                            
                    #     # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                    #     for i, v in enumerate(self.with_hat_category_dict.values()):
                    #         category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                    #         source_start_idx = v* learnable_tensor_size
                    #         source_end_idx = (v+1)* learnable_tensor_size
                    #         input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, multi_source_with_hat_scene_latent_codes(torch.tensor(i).cuda())), dim=0)
                        
                    #     model_input['scene_latent_code'] = input_latent_codes.cuda()

                    # else:       # NOTE 모자가 없는 dataset.
                    #     indices = [self.no_hat_dataset_train_subdir.index(name) for name in [train_subdir]]
                    #     multi_source_without_hat_scene_latent_codes = self.multi_source_without_hat_scene_latent_codes[indices[0]]

                    #     # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                    #     for i, v in self.category_dict.items():
                    #         category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                    #         start_idx = v* learnable_tensor_size
                    #         end_idx = (v+1)* learnable_tensor_size
                    #         input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                            
                    #     # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                    #     for i, v in enumerate(self.without_hat_category_dict.values()):
                    #         category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                    #         source_start_idx = v* learnable_tensor_size
                    #         source_end_idx = (v+1)* learnable_tensor_size
                    #         input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, multi_source_without_hat_scene_latent_codes(torch.tensor(i).cuda())), dim=0)
                            
                    #     model_input['scene_latent_code'] = input_latent_codes.cuda()
                    

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

            if self.multi_source_training and False:          # NOTE latent code combination으로 만들어주는거다. a.k.a. Latent swapping
                for batch_index, train_subdir in enumerate(self.dataset_train_subdir):
                    for category, idx in self.category_dict.items():
                        novel_view_type = 'default_multi_source_rendering_{}'.format(train_subdir)
                        plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                        img_names = model_input['img_name'][:,0].cpu().numpy()[0]
                        img_names = np.array('{}-{}'.format(img_names, category)) # category를 나타낸다.

                        learnable_tensor_size = self.category_latent_dim + self.scene_latent_dim

                        input_latent_codes = torch.zeros(len(model_input['sub_dir']), learnable_tensor_size*len(self.category_dict))

                        if train_subdir in self.no_hat_dataset_train_subdir:
                            # NOTE latent combination
                            for i, v in self.category_dict.items():
                                category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                                start_idx = v* learnable_tensor_size
                                end_idx = (v+1)* learnable_tensor_size
                                input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                            
                            for i, v in enumerate(self.source_category_dict.values()):
                                category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                                source_start_idx = v* learnable_tensor_size
                                source_end_idx = (v+1)* learnable_tensor_size
                                input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                            # NOTE hair에 대해서만 찍어보기.
                            indices = [self.no_hat_dataset_train_subdir.index(name) for name in [train_subdir]]
                            multi_source_without_hat_scene_latent_codes = self.multi_source_without_hat_scene_latent_codes[indices[0]]

                            for i, v in enumerate(self.without_hat_category_dict.values()):
                                if i == idx:      # NOTE hair
                                    category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                                    source_start_idx = v* learnable_tensor_size
                                    source_end_idx = (v+1)* learnable_tensor_size
                                    input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, multi_source_without_hat_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                            model_input['scene_latent_code'] = input_latent_codes.cuda()

                        elif train_subdir in self.hat_dataset_train_subdir:
                            # NOTE latent combination
                            for i, v in self.category_dict.items():
                                category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                                start_idx = v* learnable_tensor_size
                                end_idx = (v+1)* learnable_tensor_size
                                input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                            
                            for i, v in enumerate(self.source_category_dict.values()):
                                category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                                source_start_idx = v* learnable_tensor_size
                                source_end_idx = (v+1)* learnable_tensor_size
                                input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                            indices = [self.hat_dataset_train_subdir.index(name) for name in [train_subdir]]
                            multi_source_with_hat_scene_latent_codes = self.multi_source_with_hat_scene_latent_codes[indices[0]]

                            for i, v in enumerate(self.with_hat_category_dict.values()):
                                if i == idx:
                                    category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                                    source_start_idx = v* learnable_tensor_size
                                    source_end_idx = (v+1)* learnable_tensor_size
                                    input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, multi_source_with_hat_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                            model_input['scene_latent_code'] = input_latent_codes.cuda()

                            
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


            # NOTE interpolation하려고 만들었다.
            if self.multi_source_training: # a.k.a. Latent interpolation
                for batch_index, train_subdir in enumerate(self.dataset_train_subdir):
                    novel_view_type = 'interpolation_multi_source_rendering_{}'.format(train_subdir)
                    plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                    img_names = model_input['img_name'][:,0].cpu().numpy()[0]
                    # img_names = np.array('{}-{}'.format(img_names, batch_index)) # 1-0, 1-1, 이런식으로해서 뒷부분은 dataset을 가르키도록 해보겠다.

                    learnable_tensor_size = self.category_latent_dim + self.scene_latent_dim
                    input_latent_codes = torch.zeros(len(model_input['sub_dir']), learnable_tensor_size*len(self.category_dict))

                    # NOTE source에 대해서 latent code를 만들어줌.
                    for i, v in self.category_dict.items():
                        category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                        start_idx = v* learnable_tensor_size
                        end_idx = (v+1)* learnable_tensor_size
                        input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                    
                    for i, v in enumerate(self.source_category_dict.values()):
                        category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                        source_start_idx = v* learnable_tensor_size
                        source_end_idx = (v+1)* learnable_tensor_size
                        input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                    source_latent_codes = input_latent_codes.cuda()

                    if train_subdir in self.hat_dataset_train_subdir:
                        indices = [self.hat_dataset_train_subdir.index(name) for name in [train_subdir]]
                        multi_source_with_hat_scene_latent_codes = self.multi_source_with_hat_scene_latent_codes[indices[0]]
                        
                        # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                        for i, v in self.category_dict.items():
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                            start_idx = v* learnable_tensor_size
                            end_idx = (v+1)* learnable_tensor_size
                            input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                            
                        # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                        for i, v in enumerate(self.with_hat_category_dict.values()):
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                            source_start_idx = v* learnable_tensor_size
                            source_end_idx = (v+1)* learnable_tensor_size
                            input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, multi_source_with_hat_scene_latent_codes(torch.tensor(i).cuda())), dim=0)
                    
                    elif train_subdir in self.no_hat_dataset_train_subdir:
                        indices = [self.no_hat_dataset_train_subdir.index(name) for name in [train_subdir]]
                        multi_source_without_hat_scene_latent_codes = self.multi_source_without_hat_scene_latent_codes[indices[0]]

                        # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                        for i, v in self.category_dict.items():
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                            start_idx = v* learnable_tensor_size
                            end_idx = (v+1)* learnable_tensor_size
                            input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                            
                        # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                        for i, v in enumerate(self.without_hat_category_dict.values()):
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                            source_start_idx = v* learnable_tensor_size
                            source_end_idx = (v+1)* learnable_tensor_size
                            input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, multi_source_without_hat_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                    # Add to the model_input dictionary
                    model_input['scene_latent_code'] = self.linear_interpolation(source_latent_codes, input_latent_codes.cuda(), 0.4)

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

            
            if self.test_target_default_rendering:
                novel_view_type = 'default_rendering_{}'.format(model_input['sub_dir'][0])
                learnable_tensor_size = self.category_latent_dim + self.scene_latent_dim
                input_latent_codes = torch.zeros(len(model_input['sub_dir']), learnable_tensor_size*len(self.category_dict))

                # NOTE zero scene latent로 일단 다 초기화한다.
                for i, v in self.category_dict.items():
                    category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).detach() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                    start_idx = v*learnable_tensor_size
                    end_idx = (v+1)*learnable_tensor_size
                    input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)

                target_index_list = []
                target_category_latent_codes = []
                for i, v in self.target_category_dict.items():
                    target_index_list.append(v)
                    tensor = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).unsqueeze(0) # self.category_latent_codes(torch.tensor(v).cuda()).detach().unsqueeze(0) # int_to_tensor(v).unsqueeze(0)
                    target_category_latent_codes.append(tensor)
                target_category_latent_codes = torch.cat(target_category_latent_codes, dim=0).cuda().detach()

                # target_start_idx = 0
                # target_end_idx = 32
                # target_category_code = int_to_tensor(0).cuda().detach()

                # input_latent_codes[:, target_start_idx:target_end_idx] = torch.cat((target_category_code, self.target_scene_latent_codes(torch.tensor(0).cuda())), dim=0)

                for i, v in enumerate(target_index_list):
                    target_start_idx = v*learnable_tensor_size
                    target_end_idx = (v+1)*learnable_tensor_size
                    # input_latent_codes[:, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.target_scene_latent_category_codes(torch.tensor(i).cuda())), dim=0)
                    input_latent_codes[:, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.target_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                # Add to the model_input dictionary
                model_input['scene_latent_code'] = input_latent_codes.cuda()

                model_outputs = self.model(model_input)
                for k, v in model_outputs.items():
                    try:
                        model_outputs[k] = v.detach()
                    except:
                        model_outputs[k] = v
                plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                # img_names = model_input['img_name'][:, 0].cpu().numpy()
                img_name = str(model_input['img_name'].item())

                # tqdm.write("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_names[0]))))
                if img_index == 0:
                    tqdm.write("[INFO] Saving path: {}".format(plot_dir[0]))
                utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
                if eval_all:
                    for dir in plot_dir:
                        utils.mkdir_ifnotexists(dir)
                
                rendering_select = {
                    'rendering_grid': False,
                    'rendering_rgb': False,
                    'rendering_rgb_dilate_erode': True,
                    'rendering_normal': False,
                    'rendering_normal_dilate_erode': True,
                    'rendering_albedo': False,
                    'rendering_shading': False,
                    'rendering_segment': True,
                    'rendering_mask_hole': True
                }
                # plt.plot(img_names,
                plt.plot(img_name,
                        model_outputs,
                        ground_truth,
                        plot_dir,
                        self.test_epoch,
                        self.img_res,
                        # is_eval=True,
                        first=True,
                        custom_settings={'novel_view': novel_view_type, 'rendering_select': rendering_select})


            if self.test_multi_composition:
                # NOTE sh와 zero latent를 넣는 거까지는 똑같다. 다만, 두개 이상의 composition을 넣어주는 것이다. 즉, default rendering과 크게 다르지 않다.
                # multi_composition = [0, 17]
                # compositional_part, scene_latent_codes_tensors = [], []
                # # for mc in multi_composition:
                # for mc in self.test_multi_composition_list:
                #     compositional_part.append(mc)
                #     scene_latent_codes_tensors.append(self.scene_latent_codes(torch.LongTensor([self.dataset_train_subdir.index(mc)]).to(device)).squeeze(1).detach())     # [1, 28]

                novel_view_type = 'multi_composition_{}'.format('__'.join(self.test_multi_composition_list))
                plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                img_names = model_input['img_name'][:,0].cpu().numpy()[0]


                scene_latent_codes_tensor_list = []
                index_list = []
                for comp_part in self.test_multi_composition_list:
                    # NOTE category latent와 변하는 부분에 대한 latent code를 만들어주는 단계이다.
                    batch_index = self.dataset_train_subdir.index(comp_part)
                    scene_latent_codes_tensor = self.scene_latent_codes(torch.LongTensor([batch_index]).to(device)).squeeze(1).detach()     # [1, 28]
                    # for sub_dir_item in [comp_part]:
                    #     category_idx = self.category_dict[sub_dir_item.split('_')[0]]
                    #     index_list.append(category_idx)
                    category_idx = self.category_dict[comp_part.split('_')[0]]
                    index_list.append(category_idx)
                    category_latent_codes = F.one_hot(torch.tensor([category_idx]), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(index_list).cuda())
                    scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)
                    scene_latent_codes_tensor_list.append(scene_latent_codes_tensor)

                # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. [B, 320]
                input_latent_codes = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))
                
                # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                for i, v in self.category_dict.items():
                    category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                    start_idx = v*scene_latent_codes_tensor.shape[1]
                    end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                    input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                        
                # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                for i, v in enumerate(self.source_category_dict.values()):
                    category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                    source_start_idx = v*scene_latent_codes_tensor.shape[1]
                    source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                    input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                # NOTE 마지막으로 db human의 latent code를 넣어준다.
                for i in range(len(index_list)):
                    start_idx = index_list[i]*scene_latent_codes_tensor.shape[1]
                    end_idx = (index_list[i]+1)*scene_latent_codes_tensor.shape[1]
                    input_latent_codes[:, start_idx:end_idx] = scene_latent_codes_tensor_list[i]



                # # NOTE category별로 latent를 부여함 (1, 4)
                # category_latent_codes = []  # List to collect the transformed tensors
                # # Loop over each item in model_input['sub_dir']
                # index_list = []
                # for i, v in self.category_dict.items():
                #     for compart in compositional_part:
                #         if i in compart:
                #             assert v not in index_list, 'multi composition should not be the same category'
                #             index_list.append(v)
                #             tensor = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # int_to_tensor(v).unsqueeze(0)
                #             category_latent_codes.append(tensor)
                
                # scene_latent_codes_tensors_list = []
                # for category, scene in zip(category_latent_codes, scene_latent_codes_tensors):
                #     scene_latent_code_final = torch.cat((category.to(device).detach(), scene), dim=1)
                #     scene_latent_codes_tensors_list.append(scene_latent_code_final)
                # # category_latent_codes = torch.cat(category_latent_codes, dim=0).to(device).detach()
                # # scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)
                
                # # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. 
                # input_latent_codes = torch.zeros(scene_latent_codes_tensors_list[0].shape[0], scene_latent_codes_tensors_list[0].shape[1]*len(self.category_dict))

                # # NOTE multicomposition을 위해 추가한 부분.
                # source_index_list = []
                # source_category_latent_codes = []
                # for i, v in self.source_category_dict.items():
                #     source_index_list.append(v)
                #     tensor = int_to_tensor(v).unsqueeze(0)
                #     source_category_latent_codes.append(tensor)
                # source_category_latent_codes = torch.cat(source_category_latent_codes, dim=0).to(device).detach()
                # #######################################

                # # NOTE source latent code도 부여함. (1, 4+28)
                # source_start_idx = 0
                # source_end_idx = scene_latent_codes_tensors_list[0].shape[1]
                # source_category_code = int_to_tensor(0).to(device).detach()
                # input_latent_codes[0, source_start_idx:source_end_idx] = torch.cat((source_category_code, self.source_scene_latent_codes(torch.tensor(0).to(device))), dim=0)

                # # NOTE multicomposition을 위해 추가한 부분.
                # for i, v in enumerate(source_index_list):
                #     source_start_idx = v*scene_latent_codes_tensors_list[0].shape[1]
                #     source_end_idx = (v+1)*scene_latent_codes_tensors_list[0].shape[1]
                #     input_latent_codes[0, source_start_idx:source_end_idx] = torch.cat((source_category_latent_codes[i], self.source_scene_latent_category_codes(torch.tensor(i).to(device))), dim=0)
                # #######################################

                # for idx, value in enumerate(scene_latent_codes_tensors_list):
                #     start_idx = index_list[idx]*value.shape[1]
                #     end_idx = (index_list[idx]+1)*value.shape[1]
                #     input_latent_codes[0, start_idx:end_idx] = value[0]

                # Add to the model_input dictionary
                model_input['scene_latent_code'] = input_latent_codes.to(device)

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

            if self.test_scene_latent_interpolation and model_input['img_name'].item() == 1:
                # NOTE 특정 부위에 대해서만 latent interpolation을 해주는 코드이다. 
                test_category = self.test_scene_latent_interpolation_category
                
                category_subdir_index = [index for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
                # category_subdir = [element for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
                category_subdir_index = [self.dataset_train_subdir.index('hat_Syuka_foxhat'), self.dataset_train_subdir.index('hat_Syuka_santahat')]
                test_category = 'hat'
                img_name = model_input['img_name'].item()
                j = category_subdir_index[(img_name-1)%len(category_subdir_index)]
                k = category_subdir_index[img_name%len(category_subdir_index)]
                latent_a = self.scene_latent_codes(torch.tensor([j]).to(model_input['idx'].device)).squeeze(1).detach() 
                latent_b = self.scene_latent_codes(torch.tensor([k]).to(model_input['idx'].device)).squeeze(1).detach() 
                interpolated_latent = lambda ratio: self.linear_interpolation(latent_a, latent_b, ratio)

                scene_latent_dim = len(self.category_dict) + self.scene_latent_dim
                input_latent_codes = torch.zeros(1, scene_latent_dim * len(self.category_dict)).to(device)

                # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                for i, v in self.category_dict.items():
                    category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                    start_idx = v*scene_latent_dim
                    end_idx = (v+1)*scene_latent_dim
                    input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                        
                # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                for i, v in enumerate(self.source_category_dict.values()):
                    category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                    source_start_idx = v*scene_latent_dim
                    source_end_idx = (v+1)*scene_latent_dim
                    input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                # NOTE 마지막으로 db human의 latent code를 넣어준다.
                # for i in range(len(index_list)):
                #     start_idx = index_list[i]*scene_latent_dim
                #     end_idx = (index_list[i]+1)*scene_latent_dim
                #     input_latent_codes[i, start_idx:end_idx] = scene_latent_codes_tensor[i]

                # NOTE category별로 latent를 부여함 (1, 4)
                category_value = self.category_dict[test_category]
                category_latent_code = F.one_hot(torch.tensor(category_value), num_classes=len(self.category_dict)).unsqueeze(0)     # int_to_tensor(category_value).unsqueeze(0)       

                for l in range(self.test_interpolation_step_size):
                    # NOTE interpolation을 하는 코드.
                    novel_view_type = '{}_interpolation'.format(test_category)
                    plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                    img_name = model_input['img_name'].item()
                    img_name = np.array('{}-{}'.format(img_name, l))

                    scene_latent_code = interpolated_latent(l/self.test_interpolation_step_size)
                    scene_latent_code = torch.cat((category_latent_code.to(device).detach(), scene_latent_code), dim=1)

                    start_idx = category_value*scene_latent_dim
                    end_idx = (category_value+1)*scene_latent_dim
                    input_latent_codes[0, start_idx:end_idx] = scene_latent_code[0]

                    model_input['scene_latent_code'] = input_latent_codes.cuda()

                    with torch.set_grad_enabled(True):
                        model_outputs = self.model(model_input)

                    for k, v in model_outputs.items():
                        try:
                            model_outputs[k] = v.detach()
                        except:
                            model_outputs[k] = v

                    print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_name))))
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


                    img_name = model_input['img_name'].item()
                    img_name = np.array('{}-{}'.format(img_name, 89-l))

                    scene_latent_code = interpolated_latent(l/self.test_interpolation_step_size)
                    scene_latent_code = torch.cat((category_latent_code.to(device).detach(), scene_latent_code), dim=1)

                    start_idx = category_value*scene_latent_dim
                    end_idx = (category_value+1)*scene_latent_dim
                    input_latent_codes[0, start_idx:end_idx] = scene_latent_code[0]

                    model_input['scene_latent_code'] = input_latent_codes.cuda()

                    with torch.set_grad_enabled(True):
                        model_outputs = self.model(model_input)

                    for k, v in model_outputs.items():
                        try:
                            model_outputs[k] = v.detach()
                        except:
                            model_outputs[k] = v

                    print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_name))))
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
            

            if self.test_scene_latent_interpolation and model_input['img_name'].item() == 1 and False:
                # NOTE suppl.video.에서 쓰는 코드이다. 원본에서 특정 부위만 바꾸었다가 다시 돌려놓는걸 만든다. 
                test_category = self.test_scene_latent_interpolation_category
                
                category_subdir_index = [index for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
                # category_subdir = [element for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
                category_subdir_index = [self.dataset_train_subdir.index('nose_LILHUDDY')]
                test_category = 'nose'
                img_name = model_input['img_name'].item()

                # j = category_subdir_index[(img_name-1)%len(category_subdir_index)]
                j = list(self.source_category_dict.keys()).index(test_category)
                k = category_subdir_index[(img_name-1)%len(category_subdir_index)]
                # latent_a = self.scene_latent_codes(torch.tensor([j]).to(model_input['idx'].device)).squeeze(1).detach() 
                latent_a = self.source_scene_latent_codes(torch.tensor([j]).cuda()).detach()
                latent_b = self.scene_latent_codes(torch.tensor([k]).to(model_input['idx'].device)).detach() 
                interpolated_latent = lambda ratio: self.linear_interpolation(latent_a, latent_b, ratio)

                scene_latent_dim = len(self.category_dict) + self.scene_latent_dim
                input_latent_codes = torch.zeros(1, scene_latent_dim * len(self.category_dict)).to(device)

                # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                for i, v in self.category_dict.items():
                    category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                    start_idx = v*scene_latent_dim
                    end_idx = (v+1)*scene_latent_dim
                    input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                        
                # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                for i, v in enumerate(self.source_category_dict.values()):
                    category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                    source_start_idx = v*scene_latent_dim
                    source_end_idx = (v+1)*scene_latent_dim
                    input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                # NOTE 마지막으로 db human의 latent code를 넣어준다.
                # for i in range(len(index_list)):
                #     start_idx = index_list[i]*scene_latent_dim
                #     end_idx = (index_list[i]+1)*scene_latent_dim
                #     input_latent_codes[i, start_idx:end_idx] = scene_latent_codes_tensor[i]

                # NOTE category별로 latent를 부여함 (1, 4)
                category_value = self.category_dict[test_category]
                category_latent_code = F.one_hot(torch.tensor(category_value), num_classes=len(self.category_dict)).unsqueeze(0)     # int_to_tensor(category_value).unsqueeze(0)       

                for l in range(self.test_interpolation_step_size):
                    # NOTE interpolation을 하는 코드.
                    novel_view_type = '{}_interpolation'.format(test_category)
                    plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                    img_name = model_input['img_name'].item()

                    img_name = np.array('{}-{}'.format(img_name, l))

                    scene_latent_code = interpolated_latent(l/self.test_interpolation_step_size)
                    scene_latent_code = torch.cat((category_latent_code.to(device).detach(), scene_latent_code), dim=1)

                    start_idx = category_value*scene_latent_dim
                    end_idx = (category_value+1)*scene_latent_dim
                    input_latent_codes[0, start_idx:end_idx] = scene_latent_code[0]

                    model_input['scene_latent_code'] = input_latent_codes.cuda()

                    with torch.set_grad_enabled(True):
                        model_outputs = self.model(model_input)

                    for k, v in model_outputs.items():
                        try:
                            model_outputs[k] = v.detach()
                        except:
                            model_outputs[k] = v

                    print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_name))))
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

                    # NOTE for reverse play
                    img_name = model_input['img_name'].item()
                    img_name = np.array('{}-{}'.format(img_name, 89-l))

                    scene_latent_code = interpolated_latent(l/self.test_interpolation_step_size)
                    scene_latent_code = torch.cat((category_latent_code.to(device).detach(), scene_latent_code), dim=1)

                    start_idx = category_value*scene_latent_dim
                    end_idx = (category_value+1)*scene_latent_dim
                    input_latent_codes[0, start_idx:end_idx] = scene_latent_code[0]

                    model_input['scene_latent_code'] = input_latent_codes.cuda()

                    with torch.set_grad_enabled(True):
                        model_outputs = self.model(model_input)

                    for k, v in model_outputs.items():
                        try:
                            model_outputs[k] = v.detach()
                        except:
                            model_outputs[k] = v

                    print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_name))))
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
            

            if self.test_scene_latent_interpolation and False: # 모든 category에 대해 category별로 해주는 코드이다. Generative Performance Comparison에 사용한다.
                # test_category = self.test_scene_latent_interpolation_category
                for test_category, v in self.category_dict.items():
                    # if test_category != 'eyebrows':
                    #     continue
                    category_subdir_index = [index for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
                    # category_subdir = [element for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
                    # category_subdir_index = [34, 35]
                    # test_category = 'nose'
                    # img_name = model_input['img_name'].item()
                    for idx in category_subdir_index:
                        # j = category_subdir_index[(img_name-1)%len(category_subdir_index)]
                        # k = category_subdir_index[img_name%len(category_subdir_index)]
                        j = idx
                        k = idx+1
                        if k >= len(category_subdir_index):
                            k = category_subdir_index[0]
                        latent_a = self.scene_latent_codes(torch.tensor([j]).to(model_input['idx'].device)).squeeze(1).detach() 
                        latent_b = self.scene_latent_codes(torch.tensor([k]).to(model_input['idx'].device)).squeeze(1).detach() 
                        interpolated_latent = lambda ratio: self.linear_interpolation(latent_a, latent_b, ratio)

                        scene_latent_dim = len(self.category_dict) + self.scene_latent_dim
                        input_latent_codes = torch.zeros(1, scene_latent_dim * len(self.category_dict)).to(device)

                        # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                        for i, v in self.category_dict.items():
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                            start_idx = v*scene_latent_dim
                            end_idx = (v+1)*scene_latent_dim
                            input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                                
                        # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                        for i, v in enumerate(self.source_category_dict.values()):
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                            source_start_idx = v*scene_latent_dim
                            source_end_idx = (v+1)*scene_latent_dim
                            input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                        # NOTE category별로 latent를 부여함 (1, 4)
                        category_value = self.category_dict[test_category]
                        category_latent_code = F.one_hot(torch.tensor(category_value), num_classes=len(self.category_dict)).unsqueeze(0)     # int_to_tensor(category_value).unsqueeze(0)       

                        # for l in range(self.test_interpolation_step_size):
                            # NOTE interpolation을 하는 코드.
                        novel_view_type = '{}_interpolation'.format(test_category)
                        plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                        img_name = model_input['img_name'].item()
                        img_name = np.array('{}-{}'.format(img_name, idx))

                        scene_latent_code = interpolated_latent(0.5)
                        scene_latent_code = torch.cat((category_latent_code.to(device).detach(), scene_latent_code), dim=1)

                        start_idx = category_value*scene_latent_dim
                        end_idx = (category_value+1)*scene_latent_dim
                        input_latent_codes[0, start_idx:end_idx] = scene_latent_code[0]

                        model_input['scene_latent_code'] = input_latent_codes.cuda()

                        with torch.set_grad_enabled(True):
                            self.model.training = False
                            model_outputs = self.model(model_input)

                        for k, v in model_outputs.items():
                            try:
                                model_outputs[k] = v.detach()
                            except:
                                model_outputs[k] = v

                        print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_name))))
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
                        
                        if (model_input['img_name'].item() >= 500):
                            break

                        del model_outputs
                        del self.model
                        self.model = copy.copy(self.model_free_memory)
                        gc.collect()
                        torch.cuda.empty_cache()


            if self.test_target_blending_default_rendering: # and model_input['img_name'].item() >= 68:
                # # NOTE 20231120 14:29 먼저 FLAME canonical space에서 alignment를 맞추는 코드.
                # # NOTE shape parameter optimization이다. 이건 해본건데 생각보다 잘 안된다. targeting은 훌륭하나 shape가 너무 많이 바뀐다.
                # # shape_learnable = torch.nn.Embedding(model_input['shape'].shape[0], model_input['shape'].shape[1], sparse=False).cuda()
                # # shape_learnable.weight.data = model_input['shape'].cuda()
                # # param = []
                # # param += list(shape_learnable.parameters())
                # # shape_bak = model_input['shape'].clone()

                # # NOTE translation vector. 모든 점에 대해 동일한 translation을 제공한다.
                # translation_learnable = torch.nn.Embedding(1, 3, sparse=False).cuda()
                # torch.nn.init.zeros_(translation_learnable.weight.data)
                # param = []
                # param += list(translation_learnable.parameters())

                # # NOTE rotation vector. 모든 점에 대해 동일한 rotation을 제공한다.
                # # angle axis
                # rotation_learnable = torch.nn.Embedding(1, 3, sparse=False).cuda()              
                # torch.nn.init.zeros_(rotation_learnable.weight.data)
                # param += list(rotation_learnable.parameters())

                # # NOTE adam optimizer.
                #     # 0.01은 매 iteration마다 landmark를 잡았을 때 잘 되었던 경우이다. 하지만 아예 다른곳으로 가도 잘되는 현상이 있었다. 
                # optimizer = torch.optim.Adam(param, lr=0.01)                                   

                # # NOTE latent code configuration.
                # learnable_tensor_size = self.category_latent_dim + self.scene_latent_dim
                # input_latent_codes = torch.zeros(len(model_input['sub_dir']), learnable_tensor_size*len(self.category_dict))

                # # NOTE zero scene latent로 일단 다 초기화한다.
                # for i, v in self.category_dict.items():
                #     category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).detach() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                #     start_idx = v*learnable_tensor_size
                #     end_idx = (v+1)*learnable_tensor_size
                #     input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.test_target_zero_latent_codes(torch.tensor(0).cuda())), dim=0)

                # # NOTE target의 category latent code.
                # target_index_list = []
                # target_category_latent_codes = []
                # for i, v in self.target_category_dict.items():
                #     target_index_list.append(v)
                #     tensor = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).unsqueeze(0) # self.category_latent_codes(torch.tensor(v).cuda()).detach().unsqueeze(0) # int_to_tensor(v).unsqueeze(0)
                #     target_category_latent_codes.append(tensor)
                # target_category_latent_codes = torch.cat(target_category_latent_codes, dim=0).cuda().detach()

                # # NOTE target의 latent code와 category latent code를 합쳐준다.
                # for i, v in enumerate(target_index_list):
                #     target_start_idx = v*learnable_tensor_size
                #     target_end_idx = (v+1)*learnable_tensor_size
                #     input_latent_codes[:, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.test_target_scene_latent_category_codes(torch.tensor(i).cuda())), dim=0)

                # # Add to the model_input dictionary
                # model_input['scene_latent_code'] = input_latent_codes.cuda()
                # # model_input['indices_tensor'] = 0
                # model_input['target_human_values'] = True
                # model_input['category'] = self.test_scene_latent_interpolation_category
                
                # with torch.set_grad_enabled(True):
                #     middle_inference = self.test_target_human_model(model_input)

                # for k, v in middle_inference.items():
                #     try:
                #         middle_inference[k] = v.detach().cpu()
                #     except:
                #         middle_inference[k] = v

                # model_input['middle_inference'] = middle_inference
                # del model_input['scene_latent_code'], model_input['target_human_values'] #  model_input['masked_point_cloud_indices'], model_input['indices_tensor'],
                # del self.test_target_human_model
                # self.test_target_human_model = copy.copy(self.test_target_human_model_free_memory)
                # gc.collect()
                # torch.cuda.empty_cache()


                # # NOTE 첫번째 frame에 대해서 sh와 th를 서로 맞춰주는 코드이다.
                # test_category = self.test_scene_latent_interpolation_category
                
                # category_subdir_index = [index for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
                # category_subdir = [element for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]

                # img_name = model_input['img_name'].item()
                # novel_view_type = 'target_blending_{}_{}'.format(self.test_train_split_name, test_category)
                # plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]

                # index_nn_ = None
                # idx = 14            # 뭘로 돌리는지는 사실 중요하지 않고 어떤 sequence로 확인하냐의 뜻이다.
                # iteration = 100
                # for epoch in range(iteration):
                #     scene_latent_codes_tensor = self.scene_latent_codes(torch.tensor([idx]).to(model_input['idx'].device)).squeeze(1).detach() 
                #     train_subdir = self.dataset_train_subdir[idx]

                #     # img_names = model_input['img_name'][:,0].cpu().numpy()[0]
                #     # img_names = np.array('{}-{}'.format(img_names, idx))

                #     model_input['indices_tensor'] = idx
                #     model_input['masked_point_cloud_indices'] = {}
                #     model_input['scene_latent_code'] = self.latent_factory(model_input, train_subdir, scene_latent_codes_tensor).to(device)  # interpolated_latent(frame/interpolation_frames).to(device)
                #     model_input['chamfer_loss'] = True
                #     # model_input['shape'] = shape_learnable(torch.tensor([0]).to(model_input['idx'].device))
                #     model_input['translation'] = translation_learnable(torch.tensor([0]).to(model_input['idx'].device))
                #     model_input['rotation'] = rotation_learnable(torch.tensor([0]).to(model_input['idx'].device))
                #     model_input['index_nn'] = index_nn_
                #     model_input['epoch'] = epoch

                #     # with torch.set_grad_enabled(True):
                #     self.model.training = False
                #     loss, index_nn = self.model(model_input)

                #     index_nn_ = index_nn

                #     optimizer.zero_grad()
                #     loss.backward()
                #     optimizer.step()

                #     print('[INFO] epoch: {}, loss: {}'.format(epoch, loss))



                # 여기서부턴 이제 inference.
                learnable_tensor_size = self.category_latent_dim + self.scene_latent_dim
                input_latent_codes = torch.zeros(len(model_input['sub_dir']), learnable_tensor_size*len(self.category_dict))

                # NOTE zero scene latent로 일단 다 초기화한다.
                for i, v in self.category_dict.items():
                    category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).detach() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                    start_idx = v*learnable_tensor_size
                    end_idx = (v+1)*learnable_tensor_size
                    input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.test_target_zero_latent_codes(torch.tensor(0).cuda())), dim=0)

                target_index_list = []
                target_category_latent_codes = []
                for i, v in self.target_category_dict.items():
                    target_index_list.append(v)
                    tensor = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).unsqueeze(0) # self.category_latent_codes(torch.tensor(v).cuda()).detach().unsqueeze(0) # int_to_tensor(v).unsqueeze(0)
                    target_category_latent_codes.append(tensor)
                target_category_latent_codes = torch.cat(target_category_latent_codes, dim=0).cuda().detach()

                for i, v in enumerate(target_index_list):
                    target_start_idx = v*learnable_tensor_size
                    target_end_idx = (v+1)*learnable_tensor_size
                    # input_latent_codes[:, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.test_target_scene_latent_category_codes(torch.tensor(i).cuda())), dim=0)
                    input_latent_codes[:, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.test_target_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                # Add to the model_input dictionary
                model_input['scene_latent_code'] = input_latent_codes.cuda()
                # model_input['indices_tensor'] = 0
                model_input['target_human_values'] = True
                model_input['category'] = self.test_target_category
                model_input['rotation'] = None
                model_input['translation'] = None
                
                with torch.set_grad_enabled(True):
                    middle_inference = self.test_target_human_model(model_input)

                for k, v in middle_inference.items():
                    try:
                        middle_inference[k] = v.detach().cpu()
                    except:
                        middle_inference[k] = v

                model_input['middle_inference'] = middle_inference
                del model_input['scene_latent_code'], model_input['target_human_values'] #  model_input['masked_point_cloud_indices'], model_input['indices_tensor'],
                del self.test_target_human_model
                self.test_target_human_model = copy.copy(self.test_target_human_model_free_memory)
                gc.collect()
                torch.cuda.empty_cache()


                test_category = self.test_target_category
                
                category_subdir_index = [index for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
                category_subdir = [element for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]

                img_name = model_input['img_name'].item()
                novel_view_type = 'target_blending_{}_{}'.format(self.test_train_split_name, test_category)
                plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]

                
                for idx in category_subdir_index:
                    scene_latent_codes_tensor = self.scene_latent_codes(torch.tensor([idx]).to(model_input['idx'].device)).squeeze(1).detach() 
                    train_subdir = self.dataset_train_subdir[idx]

                    if 'hair_{}'.format(model_input['sub_dir'][0]) != train_subdir:
                        continue
                    
                    img_names = model_input['img_name'][:,0].cpu().numpy()[0]
                    img_names = np.array('{}-{}'.format(img_names, idx))

                    model_input['indices_tensor'] = idx
                    model_input['masked_point_cloud_indices'] = {}
                    model_input['chamfer_loss'] = False
                    model_input['scene_latent_code'] = self.latent_factory(model_input, train_subdir, scene_latent_codes_tensor).to(device)  # interpolated_latent(frame/interpolation_frames).to(device)

                    model_input['rotation'] = rotation_learnable(torch.tensor([0]).to(model_input['idx'].device))
                    model_input['translation'] = translation_learnable(torch.tensor([0]).to(model_input['idx'].device))

                    with torch.set_grad_enabled(True):
                        self.model.training = False
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

                    del model_outputs # , model_input
                    del self.model
                    self.model = copy.copy(self.model_free_memory)
                    gc.collect()
                    torch.cuda.empty_cache()


            if self.test_target_interpolation:          # NOTE 여기서 iterpolation 뿐만 아니라 일반 rendering도 해준다.
                # 여기서부턴 이제 inference.
                learnable_tensor_size = self.category_latent_dim + self.scene_latent_dim
                input_latent_codes = torch.zeros(len(model_input['sub_dir']), learnable_tensor_size*len(self.category_dict))

                # NOTE zero scene latent로 일단 다 초기화한다.
                for i, v in self.category_dict.items():
                    category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).detach() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                    start_idx = v*learnable_tensor_size
                    end_idx = (v+1)*learnable_tensor_size
                    input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.test_target_zero_latent_codes(torch.tensor(0).cuda())), dim=0)

                target_index_list = []
                target_category_latent_codes = []
                for i, v in self.target_category_dict.items():
                    target_index_list.append(v)
                    tensor = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).unsqueeze(0) # self.category_latent_codes(torch.tensor(v).cuda()).detach().unsqueeze(0) # int_to_tensor(v).unsqueeze(0)
                    target_category_latent_codes.append(tensor)
                target_category_latent_codes = torch.cat(target_category_latent_codes, dim=0).cuda().detach()

                for i, v in enumerate(target_index_list):
                    target_start_idx = v*learnable_tensor_size
                    target_end_idx = (v+1)*learnable_tensor_size
                    # input_latent_codes[:, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.test_target_scene_latent_category_codes(torch.tensor(i).cuda())), dim=0)
                    input_latent_codes[:, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.test_target_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                # Add to the model_input dictionary
                model_input['scene_latent_code'] = input_latent_codes.cuda()
                # model_input['indices_tensor'] = 0
                model_input['target_human_values'] = True
                model_input['category'] = self.test_target_category
                model_input['rotation'] = None
                model_input['translation'] = None
                
                with torch.set_grad_enabled(True):
                    middle_inference = self.test_target_human_model(model_input)

                for k, v in middle_inference.items():
                    try:
                        middle_inference[k] = v.detach().cpu()
                    except:
                        middle_inference[k] = v

                model_input['middle_inference'] = middle_inference
                del model_input['scene_latent_code'], model_input['target_human_values'] #  model_input['masked_point_cloud_indices'], model_input['indices_tensor'],
                del self.test_target_human_model
                self.test_target_human_model = copy.copy(self.test_target_human_model_free_memory)
                gc.collect()
                torch.cuda.empty_cache()


                # test_category = self.test_scene_latent_interpolation_category
                
                # category_subdir_index = [index for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
                # category_subdir = [element for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]

                img_name = model_input['img_name'].item()
                # novel_view_type = 'target_interpolation_blending_{}_{}'.format(self.test_train_split_name, test_category)
                # plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                
                # category_subdir_index = [index for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
                category_subdir_index = [11, 13] # NOTE [10, 12], [11, 13]
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
                novel_view_type = 'zero_shot_interpolation_{}'.format('hair')
                plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                
                # scene_latent_codes_tensor1 = self.scene_latent_codes(torch.LongTensor([batch_index1]).to(device)).squeeze(1).detach()     # [1, 28]
                # scene_latent_codes_tensor2 = self.scene_latent_codes(torch.LongTensor([batch_index2]).to(device)).squeeze(1).detach()     # [1, 28]

                input_latent_codes1 = self.latent_factory(model_input, train_subdir1, scene_latent_codes_tensor1)
                input_latent_codes2 = self.latent_factory(model_input, train_subdir2, scene_latent_codes_tensor2)           
                
                interpolated_latent = lambda ratio: self.linear_interpolation(input_latent_codes1, input_latent_codes2, ratio)

                interpolation_frames = 60
                for frame in range(interpolation_frames):
                    img_names = model_input['img_name'][:,0].cpu().numpy()[0]
                    img_names = np.array('{}-{}'.format(img_names, frame))

                    model_input['indices_tensor'] = batch_index1
                    model_input['masked_point_cloud_indices'] = {}
                    model_input['scene_latent_code'] = interpolated_latent(frame/interpolation_frames).to(device)

                    model_input['chamfer_loss'] = False
                    # model_input['scene_latent_code'] = latent_factory(train_subdir, scene_latent_codes_tensor).to(device)  # interpolated_latent(frame/interpolation_frames).to(device)
                    model_input['translation'] = translation_learnable(torch.tensor([0]).to(model_input['idx'].device))
                    model_input['rotation'] = rotation_learnable(torch.tensor([0]).to(model_input['idx'].device))

                    with torch.set_grad_enabled(True):
                        self.model.training = False
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

                    del model_outputs # , model_input
                    del self.model
                    self.model = copy.copy(self.model_free_memory)
                    gc.collect()
                    torch.cuda.empty_cache()


            # if self.test_target_interpolation:          # NOTE 여기서 iterpolation 뿐만 아니라 일반 rendering도 해준다.
            #     test_time_optimization = True
            #     if test_time_optimization:
            #         # shape_learnable = torch.nn.Embedding(model_input['shape'].shape[0], model_input['shape'].shape[1], sparse=False).cuda()
            #         # shape_learnable.weight.data = model_input['shape'].cuda()
            #         # param = []
            #         # param += list(shape_learnable.parameters())
            #         # shape_bak = model_input['shape'].clone()

            #         translation_learnable = torch.nn.Embedding(1, 3, sparse=False).cuda()
            #         torch.nn.init.zeros_(translation_learnable.weight.data)
            #         param = []
            #         param += list(translation_learnable.parameters())

            #         rotation_learnable = torch.nn.Embedding(1, 3, sparse=False).cuda() # angle axis
            #         torch.nn.init.zeros_(rotation_learnable.weight.data)
            #         param += list(rotation_learnable.parameters())
                
            #         optimizer = torch.optim.Adam(param, lr=0.01) # 0.01은 매 iteration마다 landmark를 잡았을 때 잘 되었던 경우이다. 하지만 아예 다른곳으로 가도 잘되는 현상이 있었다. 

            #         learnable_tensor_size = self.category_latent_dim + self.scene_latent_dim
            #         input_latent_codes = torch.zeros(len(model_input['sub_dir']), learnable_tensor_size*len(self.category_dict))

            #         # NOTE zero scene latent로 일단 다 초기화한다.
            #         for i, v in self.category_dict.items():
            #             category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).detach() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
            #             start_idx = v*learnable_tensor_size
            #             end_idx = (v+1)*learnable_tensor_size
            #             input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.test_target_zero_latent_codes(torch.tensor(0).cuda())), dim=0)

            #         target_index_list = []
            #         target_category_latent_codes = []
            #         for i, v in self.target_category_dict.items():
            #             target_index_list.append(v)
            #             tensor = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).unsqueeze(0) # self.category_latent_codes(torch.tensor(v).cuda()).detach().unsqueeze(0) # int_to_tensor(v).unsqueeze(0)
            #             target_category_latent_codes.append(tensor)
            #         target_category_latent_codes = torch.cat(target_category_latent_codes, dim=0).cuda().detach()

            #         for i, v in enumerate(target_index_list):
            #             target_start_idx = v*learnable_tensor_size
            #             target_end_idx = (v+1)*learnable_tensor_size
            #             input_latent_codes[:, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.test_target_scene_latent_category_codes(torch.tensor(i).cuda())), dim=0)

            #         # Add to the model_input dictionary
            #         model_input['scene_latent_code'] = input_latent_codes.cuda()
            #         # model_input['indices_tensor'] = 0
            #         model_input['target_human_values'] = True
            #         model_input['category'] = self.test_scene_latent_interpolation_category
                    
            #         with torch.set_grad_enabled(True):
            #             middle_inference = self.test_target_human_model(model_input)

            #         for k, v in middle_inference.items():
            #             try:
            #                 middle_inference[k] = v.detach().cpu()
            #             except:
            #                 middle_inference[k] = v

            #         model_input['middle_inference'] = middle_inference
            #         del model_input['scene_latent_code'], model_input['target_human_values'] #  model_input['masked_point_cloud_indices'], model_input['indices_tensor'],
            #         del self.test_target_human_model
            #         self.test_target_human_model = copy.copy(self.test_target_human_model_free_memory)
            #         gc.collect()
            #         torch.cuda.empty_cache()


            #         test_category = self.test_scene_latent_interpolation_category
                    
            #         category_subdir_index = [index for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
            #         category_subdir = [element for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]

            #         img_name = model_input['img_name'].item()
            #         novel_view_type = 'target_blending_{}_{}'.format(self.test_train_split_name, test_category)
            #         plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]

            #         def latent_factory(train_subdir, scene_latent_codes_tensor):
            #             learnable_tensor_size = self.category_latent_dim + self.scene_latent_dim
            #             input_latent_codes = torch.zeros(len(model_input['sub_dir']), learnable_tensor_size*len(self.category_dict))

            #             # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
            #             for i, v in self.category_dict.items():
            #                 category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
            #                 start_idx = v*learnable_tensor_size
            #                 end_idx = (v+1)*learnable_tensor_size
            #                 input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                                
            #             # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
            #             for i, v in enumerate(self.source_category_dict.values()):
            #                 category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
            #                 source_start_idx = v*learnable_tensor_size
            #                 source_end_idx = (v+1)*learnable_tensor_size
            #                 input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

            #             index_list = []
            #             for sub_dir_item in [train_subdir]:
            #                 category_idx = self.category_dict[sub_dir_item.split('_')[0]]
            #                 index_list.append(category_idx)
            #             category_latent_codes = F.one_hot(torch.tensor(index_list), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(index_list).cuda())
            #             scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

            #             # NOTE 마지막으로 db human의 latent code를 넣어준다.
            #             for i in range(len(index_list)):
            #                 start_idx = index_list[i]*learnable_tensor_size
            #                 end_idx = (index_list[i]+1)*learnable_tensor_size
            #                 input_latent_codes[i, start_idx:end_idx] = scene_latent_codes_tensor[i]

            #             return input_latent_codes
                    
            #         index_nn_ = None
            #         idx = 10
            #         for epoch in range(60):
            #             scene_latent_codes_tensor = self.scene_latent_codes(torch.tensor([idx]).to(model_input['idx'].device)).squeeze(1).detach() 
            #             train_subdir = self.dataset_train_subdir[idx]

            #             # img_names = model_input['img_name'][:,0].cpu().numpy()[0]
            #             # img_names = np.array('{}-{}'.format(img_names, idx))

            #             model_input['indices_tensor'] = idx
            #             model_input['masked_point_cloud_indices'] = {}
            #             model_input['scene_latent_code'] = latent_factory(train_subdir, scene_latent_codes_tensor).to(device)  # interpolated_latent(frame/interpolation_frames).to(device)
            #             model_input['chamfer_loss'] = True
            #             # model_input['shape'] = shape_learnable(torch.tensor([0]).to(model_input['idx'].device))
            #             model_input['translation'] = translation_learnable(torch.tensor([0]).to(model_input['idx'].device))
            #             model_input['rotation'] = rotation_learnable(torch.tensor([0]).to(model_input['idx'].device))
            #             model_input['index_nn'] = index_nn_
            #             model_input['epoch'] = epoch

            #             # with torch.set_grad_enabled(True):
            #             self.model.training = False
            #             loss, index_nn = self.model(model_input)

            #             index_nn_ = index_nn

            #             optimizer.zero_grad()
            #             loss.backward()
            #             optimizer.step()

            #             print('[INFO] epoch: {}, loss: {}'.format(epoch, loss))


            #     # NOTE interpolation step
            #     test_category = self.test_scene_latent_interpolation_category
                
            #     category_subdir_index = [index for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]
            #     category_subdir_index = [10, 12]
            #     category_subdir = [element for index, element in enumerate(self.dataset_train_subdir) if test_category in str(element)]

            #     img_name = model_input['img_name'].item()
            #     j = category_subdir_index[(img_name-1)%len(category_subdir_index)]
            #     k = category_subdir_index[img_name%len(category_subdir_index)]
            #     scene_latent_codes_tensor1 = self.scene_latent_codes(torch.tensor([j]).to(model_input['idx'].device)).squeeze(1).detach() 
            #     scene_latent_codes_tensor2 = self.scene_latent_codes(torch.tensor([k]).to(model_input['idx'].device)).squeeze(1).detach() 

            #     train_subdir1 = self.dataset_train_subdir[j] # 'hair_Chloe_Grace_Moretz'
            #     train_subdir2 = self.dataset_train_subdir[k] #'hair_John_Krasinski'
            #     batch_index1 = j # self.dataset_train_subdir.index(train_subdir1)
            #     batch_index2 = k # self.dataset_train_subdir.index(train_subdir2)

            #     # interpolated_latent = lambda ratio: self.linear_interpolation(latent_a, latent_b, ratio)


            #     # NOTE generate mask images for target human.
            #     # train_subdir1 = 'hair_Chloe_Grace_Moretz'
            #     # train_subdir2 = 'hair_John_Krasinski'
            #     # batch_index1 = self.dataset_train_subdir.index(train_subdir1)
            #     # batch_index2 = self.dataset_train_subdir.index(train_subdir2)

            #     # novel_view_type = 'mask_interpolation_{}_2_{}'.format(train_subdir1, train_subdir2)
            #     novel_view_type = 'mask_interpolation_{}'.format(test_category)
            #     plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                
            #     # scene_latent_codes_tensor1 = self.scene_latent_codes(torch.LongTensor([batch_index1]).to(device)).squeeze(1).detach()     # [1, 28]
            #     # scene_latent_codes_tensor2 = self.scene_latent_codes(torch.LongTensor([batch_index2]).to(device)).squeeze(1).detach()     # [1, 28]

            #     input_latent_codes1 = latent_factory(train_subdir1, scene_latent_codes_tensor1)
            #     input_latent_codes2 = latent_factory(train_subdir2, scene_latent_codes_tensor2)           
                
            #     interpolated_latent = lambda ratio: self.linear_interpolation(input_latent_codes1, input_latent_codes2, ratio)

            #     interpolation_frames = 10
            #     for frame in range(interpolation_frames):
            #         img_names = model_input['img_name'][:,0].cpu().numpy()[0]
            #         img_names = np.array('{}-{}'.format(img_names, frame))

            #         model_input['indices_tensor'] = batch_index1
            #         model_input['masked_point_cloud_indices'] = {}
            #         model_input['scene_latent_code'] = interpolated_latent(frame/interpolation_frames).to(device)

            #         model_input['chamfer_loss'] = False
            #         # model_input['scene_latent_code'] = latent_factory(train_subdir, scene_latent_codes_tensor).to(device)  # interpolated_latent(frame/interpolation_frames).to(device)
            #         model_input['translation'] = translation_learnable(torch.tensor([0]).to(model_input['idx'].device))
            #         model_input['rotation'] = rotation_learnable(torch.tensor([0]).to(model_input['idx'].device))

            #         with torch.set_grad_enabled(True):
            #             self.model.training = False
            #             model_outputs = self.model(model_input)

            #         for k, v in model_outputs.items():
            #             try:
            #                 model_outputs[k] = v.detach().cpu()
            #             except:
            #                 model_outputs[k] = v
                    
            #         for k, v in ground_truth.items():
            #             try:
            #                 ground_truth[k] = v.detach().cpu()
            #             except:
            #                 ground_truth[k] = v
                    
            #         print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_names))))
            #         utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
            #         if eval_all:
            #             for dir in plot_dir:
            #                 utils.mkdir_ifnotexists(dir)
            #         plt.plot(img_names,
            #                 model_outputs,
            #                 ground_truth,
            #                 plot_dir,
            #                 self.test_epoch,
            #                 self.img_res,
            #                 is_eval=eval_all,
            #                 first=is_first_batch,
            #                 custom_settings={'novel_view': novel_view_type})
                    
            #         # for k, v in model_input.items():
            #         #     try:
            #         #         model_input[k] = v.detach().cpu()
            #         #     except:
            #         #         model_input[k] = v

            #         del model_outputs # , model_input
            #         del self.model
            #         self.model = copy.copy(self.model_free_memory)
            #         gc.collect()
            #         torch.cuda.empty_cache()



                    # scene_latent_codes_tensor = self.scene_latent_codes(torch.tensor([idx]).to(model_input['idx'].device)).squeeze(1).detach() 
                    # train_subdir = self.dataset_train_subdir[idx]

                    # img_names = model_input['img_name'][:,0].cpu().numpy()[0]
                    # img_names = np.array('{}-{}'.format(img_names, idx))

                    # model_input['indices_tensor'] = idx
                    # model_input['masked_point_cloud_indices'] = {}
                    # model_input['chamfer_loss'] = False
                    # model_input['scene_latent_code'] = latent_factory(train_subdir, scene_latent_codes_tensor).to(device)  # interpolated_latent(frame/interpolation_frames).to(device)
                    # model_input['translation'] = translation_learnable(torch.tensor([0]).to(model_input['idx'].device))

                    # with torch.set_grad_enabled(True):
                    #     self.model.training = False
                    #     model_outputs = self.model(model_input)

                    # for k, v in model_outputs.items():
                    #     try:
                    #         model_outputs[k] = v.detach().cpu()
                    #     except:
                    #         model_outputs[k] = v
                    
                    # for k, v in ground_truth.items():
                    #     try:
                    #         ground_truth[k] = v.detach().cpu()
                    #     except:
                    #         ground_truth[k] = v
                    
                    # print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_names))))
                    # utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
                    # if eval_all:
                    #     for dir in plot_dir:
                    #         utils.mkdir_ifnotexists(dir)
                    # plt.plot(img_names,
                    #         model_outputs,
                    #         ground_truth,
                    #         plot_dir,
                    #         self.test_epoch,
                    #         self.img_res,
                    #         is_eval=eval_all,
                    #         first=is_first_batch,
                    #         custom_settings={'novel_view': novel_view_type})

                    # del model_outputs # , model_input
                    # del self.model
                    # self.model = copy.copy(self.model_free_memory)
                    # gc.collect()
                    # torch.cuda.empty_cache()

            del ground_truth

        # if not self.plot_dataset.only_json:
        #     from utils.metrics import run as cal_metrics
        #     cal_metrics(output_dir=default_rendering_plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb_erode_dilate')
        #     cal_metrics(output_dir=default_rendering_plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb_erode_dilate', no_cloth=True)

    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.conf.put('dataset.test.subsample', 1)
        self.conf.put('dataset.test.load_images', False)

        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))

        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        if kwargs['load_path'] != '':
            load_path = kwargs['load_path']
        else:
            load_path = self.train_dir
        assert os.path.exists(load_path)

        utils.mkdir_ifnotexists(self.eval_dir)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.use_background = self.conf.get_bool('dataset.use_background', default=False)

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                          subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                          json_name=self.conf.get_string('dataset.json_name'),
                                                                                          use_mean_expression=self.conf.get_bool('dataset.use_mean_expression', default=False),
                                                                                          use_background=self.use_background,
                                                                                          is_eval=False,
                                                                                          only_json=True,
                                                                                          **self.conf.get_config('dataset.train'))

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         only_json=kwargs['only_json'],
                                                                                         use_background=self.use_background,
                                                                                         is_eval=True,
                                                                                         **self.conf.get_config('dataset.test'))

        print('Finish loading data ...')

        self.model = PointAvatar(conf=self.conf.get_config('model'),
                                shape_params=self.plot_dataset.shape_params,
                                img_res=self.plot_dataset.img_res,
                                canonical_expression=self.train_dataset.mean_expression,
                                canonical_pose=self.conf.get_float(
                                    'dataset.canonical_pose',
                                    default=0.2),
                                use_background=self.use_background)
        if torch.cuda.is_available():
            self.model.cuda()
        old_checkpnts_dir = os.path.join(load_path, 'checkpoints')
        self.checkpoints_path = old_checkpnts_dir
        assert os.path.exists(old_checkpnts_dir)
        saved_model_state = torch.load(
            os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
        n_points = saved_model_state["model_state_dict"]['pc.points'].shape[0]
        self.model.pc.init(n_points)
        self.model.pc = self.model.pc.cuda()

        self.model.raster_settings.radius = saved_model_state['radius']

        self.model.load_state_dict(saved_model_state["model_state_dict"]) #, strict=False)
        self.start_epoch = saved_model_state['epoch']
        self.optimize_expression = self.conf.get_bool('train.optimize_expression')
        self.optimize_pose = self.conf.get_bool('train.optimize_camera')
        self.optimize_inputs = self.optimize_expression or self.optimize_pose

        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=min(int(self.conf.get_int('train.max_points_training') /self.model.pc.points.shape[0]),self.conf.get_int('train.max_batch',default='10')),
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn
                                                           )
        self.optimize_tracking = False
        if self.optimize_inputs:
            self.input_params_subdir = "TestInputParameters"
            test_input_params = []
            if self.optimize_expression:
                init_expression = self.plot_dataset.data["expressions"]

                self.expression = torch.nn.Embedding(len(self.plot_dataset), self.model.deformer_network.num_exp, _weight=init_expression, sparse=True).cuda()
                test_input_params += list(self.expression.parameters())

            if self.optimize_pose:
                self.flame_pose = torch.nn.Embedding(len(self.plot_dataset), 15,
                                                     _weight=self.plot_dataset.data["flame_pose"],
                                                     sparse=True).cuda()
                self.camera_pose = torch.nn.Embedding(len(self.plot_dataset), 3,
                                                      _weight=self.plot_dataset.data["world_mats"][:, :3, 3],
                                                      sparse=True).cuda()
                test_input_params += list(self.flame_pose.parameters()) + list(self.camera_pose.parameters())
            self.optimizer_cam = torch.optim.SparseAdam(test_input_params,
                                                        self.conf.get_float('train.learning_rate_cam'))

            try:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.input_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                if self.optimize_expression:
                    self.expression.load_state_dict(data["expression_state_dict"])
                if self.optimize_pose:
                    self.flame_pose.load_state_dict(data["flame_pose_state_dict"])
                    self.camera_pose.load_state_dict(data["camera_pose_state_dict"])
                print('Using pre-tracked test expressions')
            except:
                self.optimize_tracking = True
                from model.loss import Loss
                self.loss = Loss(mask_weight=0.0)
                print('Optimizing test expressions')

        self.img_res = self.plot_dataset.img_res


    def save_test_tracking(self, epoch):
        if not os.path.exists(os.path.join(self.checkpoints_path, "TestInputParameters")):
            os.mkdir(os.path.join(self.checkpoints_path, "TestInputParameters"))
        if self.optimize_inputs:
            dict_to_save = {}
            dict_to_save["epoch"] = epoch
            if self.optimize_expression:
                dict_to_save["expression_state_dict"] = self.expression.state_dict()
            if self.optimize_pose:
                dict_to_save["flame_pose_state_dict"] = self.flame_pose.state_dict()
                dict_to_save["camera_pose_state_dict"] = self.camera_pose.state_dict()
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, "TestInputParameters", str(epoch) + ".pth"))
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, "TestInputParameters", "latest.pth"))

    def run(self):
        self.model.eval()
        self.model.training = False
        self.optimize_tracking = False
        if self.optimize_tracking:
            print("Optimizing tracking, this is a slow process which is only used for calculating metrics. \n"
                  "for qualitative animation, set optimize_expression and optimize_camera to False in the conf file.")
            for data_index, (indices, model_input, ground_truth) in enumerate(self.plot_dataloader):
                print(list(model_input["idx"].reshape(-1).cpu().numpy()))
                for k, v in model_input.items():
                    try:
                        model_input[k] = v.cuda()
                    except:
                        model_input[k] = v
                for k, v in ground_truth.items():
                    try:
                        ground_truth[k] = v.cuda()
                    except:
                        ground_truth[k] = v

                R = model_input['cam_pose'][:, :3, :3]
                model_input['cam_pose'][-1, -1, -1] += 1.5

                for i in range(20):
                    if self.optimize_expression:
                        model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                    if self.optimize_pose:
                        model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                        model_input['cam_pose'] = torch.cat([R, self.camera_pose(model_input["idx"]).squeeze(1).unsqueeze(-1)], -1)

                    model_outputs = self.model(model_input)
                    loss_output = self.loss(model_outputs, ground_truth)
                    loss = loss_output['loss']
                    self.optimizer_cam.zero_grad()
                    loss.backward()
                    self.optimizer_cam.step()
            self.save_test_tracking(epoch=self.start_epoch)

        eval_all = True
        eval_iterator = iter(self.plot_dataloader)
        is_first_batch = True
        for img_index in range(len(self.plot_dataloader)):
            indices, model_input, ground_truth = next(eval_iterator)
            batch_size = model_input['expression'].shape[0]
            for k, v in model_input.items():
                try:
                    model_input[k] = v.cuda()
                except:
                    model_input[k] = v

            for k, v in ground_truth.items():
                try:
                    ground_truth[k] = v.cuda()
                except:
                    ground_truth[k] = v

            if self.optimize_inputs:
                if self.optimize_expression:
                    model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                if self.optimize_pose:
                    model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                    model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)

            # model_input['cam_pose'][-1, -1, -1] += 1.5
            
            model_outputs = self.model(model_input)
            for k, v in model_outputs.items():
                try:
                    model_outputs[k] = v.detach()
                except:
                    model_outputs[k] = v
            plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_'+str(self.start_epoch)) for i in range(len(model_input['sub_dir']))]

            img_names = model_input['img_name'][:,0].cpu().numpy()
            print("Plotting images: {}".format(img_names))
            utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
            if eval_all:
                for dir in plot_dir:
                    utils.mkdir_ifnotexists(dir)
            plt.plot(img_names,
                     model_outputs,
                     ground_truth,
                     plot_dir,
                     self.start_epoch,
                     self.img_res,
                     is_eval=eval_all,
                     first=is_first_batch,
                     )
            is_first_batch = False
            del model_outputs, ground_truth

            if img_names[0] >= 1000:
                break

        if not self.plot_dataset.only_json:
            from utils.metrics import run as cal_metrics
            cal_metrics(output_dir=plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb_erode_dilate')
            cal_metrics(output_dir=plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb_erode_dilate', no_cloth=True)



    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.conf.put('dataset.test.subsample', 1)
        self.conf.put('dataset.test.load_images', False)

        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))

        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        if kwargs['load_path'] != '':
            load_path = kwargs['load_path']
        else:
            load_path = self.train_dir
        assert os.path.exists(load_path)

        utils.mkdir_ifnotexists(self.eval_dir)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.use_background = self.conf.get_bool('dataset.use_background', default=False)

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(conf=self.conf,
                                                                                          data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                          subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                          json_name=self.conf.get_string('dataset.json_name'),
                                                                                          use_mean_expression=self.conf.get_bool('dataset.use_mean_expression', default=False),
                                                                                          use_background=self.use_background,
                                                                                          mode='train',
                                                                                        #   is_eval=False,
                                                                                          only_json=True,
                                                                                          **self.conf.get_config('dataset.train'))

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(conf=self.conf,
                                                                                         data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         only_json=kwargs['only_json'],
                                                                                         use_background=self.use_background,
                                                                                         mode='test',
                                                                                        #  is_eval=True,
                                                                                         **self.conf.get_config('dataset.test'))

        print('Finish loading data ...')

        self.model = PointAvatar(conf=self.conf.get_config('model'),
                                shape_params=self.plot_dataset.shape_params,
                                img_res=self.plot_dataset.img_res,
                                canonical_expression=self.train_dataset.mean_expression,
                                canonical_pose=self.conf.get_float(
                                    'dataset.canonical_pose',
                                    default=0.2),
                                use_background=self.use_background)
        if torch.cuda.is_available():
            self.model.cuda()
        old_checkpnts_dir = os.path.join(load_path, 'checkpoints')
        self.checkpoints_path = old_checkpnts_dir
        assert os.path.exists(old_checkpnts_dir)
        saved_model_state = torch.load(
            os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
        n_points = saved_model_state["model_state_dict"]['pc.points'].shape[0]
        self.model.pc.init(n_points)
        self.model.pc = self.model.pc.cuda()

        self.model.raster_settings.radius = saved_model_state['radius']

        self.model.load_state_dict(saved_model_state["model_state_dict"]) #, strict=False)
        self.start_epoch = saved_model_state['epoch']
        self.optimize_expression = self.conf.get_bool('train.optimize_expression')
        self.optimize_pose = self.conf.get_bool('train.optimize_camera')
        self.optimize_inputs = self.optimize_expression or self.optimize_pose

        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=min(int(self.conf.get_int('train.max_points_training') /self.model.pc.points.shape[0]),self.conf.get_int('train.max_batch',default='10')),
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn
                                                           )
        self.optimize_tracking = False
        if self.optimize_inputs:
            self.input_params_subdir = "TestInputParameters"
            test_input_params = []
            if self.optimize_expression:
                init_expression = self.plot_dataset.data["expressions"]

                self.expression = torch.nn.Embedding(len(self.plot_dataset), self.model.deformer_network.num_exp, _weight=init_expression, sparse=True).cuda()
                test_input_params += list(self.expression.parameters())

            if self.optimize_pose:
                self.flame_pose = torch.nn.Embedding(len(self.plot_dataset), 15,
                                                     _weight=self.plot_dataset.data["flame_pose"],
                                                     sparse=True).cuda()
                self.camera_pose = torch.nn.Embedding(len(self.plot_dataset), 3,
                                                      _weight=self.plot_dataset.data["world_mats"][:, :3, 3],
                                                      sparse=True).cuda()
                test_input_params += list(self.flame_pose.parameters()) + list(self.camera_pose.parameters())
            self.optimizer_cam = torch.optim.SparseAdam(test_input_params,
                                                        self.conf.get_float('train.learning_rate_cam'))

            try:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.input_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                if self.optimize_expression:
                    self.expression.load_state_dict(data["expression_state_dict"])
                if self.optimize_pose:
                    self.flame_pose.load_state_dict(data["flame_pose_state_dict"])
                    self.camera_pose.load_state_dict(data["camera_pose_state_dict"])
                print('Using pre-tracked test expressions')
            except:
                self.optimize_tracking = True
                from model.loss import Loss
                self.loss = Loss(mask_weight=0.0)
                print('Optimizing test expressions')

        self.img_res = self.plot_dataset.img_res


    def save_test_tracking(self, epoch):
        if not os.path.exists(os.path.join(self.checkpoints_path, "TestInputParameters")):
            os.mkdir(os.path.join(self.checkpoints_path, "TestInputParameters"))
        if self.optimize_inputs:
            dict_to_save = {}
            dict_to_save["epoch"] = epoch
            if self.optimize_expression:
                dict_to_save["expression_state_dict"] = self.expression.state_dict()
            if self.optimize_pose:
                dict_to_save["flame_pose_state_dict"] = self.flame_pose.state_dict()
                dict_to_save["camera_pose_state_dict"] = self.camera_pose.state_dict()
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, "TestInputParameters", str(epoch) + ".pth"))
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, "TestInputParameters", "latest.pth"))

    def run(self):
        self.model.eval()
        self.model.training = False
        self.optimize_tracking = False
        if self.optimize_tracking:
            print("Optimizing tracking, this is a slow process which is only used for calculating metrics. \n"
                  "for qualitative animation, set optimize_expression and optimize_camera to False in the conf file.")
            for data_index, (indices, model_input, ground_truth) in enumerate(self.plot_dataloader):
                print(list(model_input["idx"].reshape(-1).cpu().numpy()))
                for k, v in model_input.items():
                    try:
                        model_input[k] = v.cuda()
                    except:
                        model_input[k] = v
                for k, v in ground_truth.items():
                    try:
                        ground_truth[k] = v.cuda()
                    except:
                        ground_truth[k] = v

                R = model_input['cam_pose'][:, :3, :3]
                for i in range(20):
                    if self.optimize_expression:
                        model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                    if self.optimize_pose:
                        model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                        model_input['cam_pose'] = torch.cat([R, self.camera_pose(model_input["idx"]).squeeze(1).unsqueeze(-1)], -1)

                    model_outputs = self.model(model_input)
                    loss_output = self.loss(model_outputs, ground_truth)
                    loss = loss_output['loss']
                    self.optimizer_cam.zero_grad()
                    loss.backward()
                    self.optimizer_cam.step()
            self.save_test_tracking(epoch=self.start_epoch)

        eval_all = True
        eval_iterator = iter(self.plot_dataloader)
        is_first_batch = True
        for img_index in range(len(self.plot_dataloader)):
            indices, model_input, ground_truth = next(eval_iterator)
            batch_size = model_input['expression'].shape[0]
            for k, v in model_input.items():
                try:
                    model_input[k] = v.cuda()
                except:
                    model_input[k] = v

            for k, v in ground_truth.items():
                try:
                    ground_truth[k] = v.cuda()
                except:
                    ground_truth[k] = v

            

            if self.optimize_inputs:
                if self.optimize_expression:
                    model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                if self.optimize_pose:
                    model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                    model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)
            
            # model_input['cam_pose'][-1, -1, -1] += 1.5
            
            model_outputs = self.model(model_input)
            for k, v in model_outputs.items():
                try:
                    model_outputs[k] = v.detach()
                except:
                    model_outputs[k] = v
            plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_'+str(self.start_epoch)) for i in range(len(model_input['sub_dir']))]

            img_names = model_input['img_name'][:,0].cpu().numpy()
            print("Plotting images: {}".format(img_names))
            utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
            if eval_all:
                for dir in plot_dir:
                    utils.mkdir_ifnotexists(dir)
            plt.plot(img_names,
                     model_outputs,
                     ground_truth,
                     plot_dir,
                     self.start_epoch,
                     self.img_res,
                     is_eval=eval_all,
                     first=is_first_batch,
                     )
            is_first_batch = False
            del model_outputs, ground_truth

            if img_names[0] >= 500:
                break

        if not self.plot_dataset.only_json:
            from utils.metrics import run as cal_metrics
            cal_metrics(output_dir=plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb_erode_dilate')
            cal_metrics(output_dir=plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb_erode_dilate', no_cloth=True)