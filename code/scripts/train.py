import os
from pyhocon import ConfigFactory
import torch
import sys
sys.path.append('./')
import utils.general as utils
import utils.plots as plt
import wandb
from functools import partial
print = partial(print, flush=True)
from datetime import datetime
from shutil import copyfile
import shutil
import natsort
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator
import torch.nn.functional as F
from tqdm import tqdm

#-------------for distributed training--------------
#----------borrowed from https://github.com/The-AI-Summer/pytorch-ddp.git -------------
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):

    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


def find_checkpoint_file(directory, epoch_number):
    file_list = natsort.natsorted(os.listdir(os.path.join(directory, 'ModelParameters')))
    last_file_exists = False
    
    if epoch_number == 'latest':
        search_prefix = 'latest'
        last_files = []
    else:
        search_prefix = epoch_number

    for file_name in file_list:
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


class TrainRunnerPEGASUS():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        opt = kwargs['opt']
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.max_batch = self.conf.get_int('train.max_batch')
        self.batch_size = min(int(self.conf.get_int('train.max_points_training') / 400), self.max_batch)
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.optimize_expression = self.conf.get_bool('train.optimize_expression')
        self.optimize_pose = self.conf.get_bool('train.optimize_camera')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        os.environ['WANDB_DIR'] = os.path.join(self.exps_folder_name)
        wandb.init(project=self.conf.get_string('train.projectname'), name=self.subject + '_' + self.methodname, config=self.conf, tags=kwargs['wandb_tags'])

        # NOTE custom
        self.distributed = self.conf.get_bool('train.distributed')
        self.accelerate = self.conf.get_bool('train.accelerate')
        if 'WORLD_SIZE' in os.environ:
            self.distributed = (int(os.environ['WORLD_SIZE']) > 1)

        if self.distributed:
            torch.cuda.set_device(opt.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            cuda = torch.device(f'cuda:{opt.local_rank}')
            dist.barrier()
        elif self.accelerate:
            self.accelerator = Accelerator()
            cuda = self.accelerator.device
        else:
            torch.set_num_threads(1)   
            cuda = torch.device(f'cuda:0')

        self.is_val = self.conf.get_bool('val.is_val', default=True)
        # self.category_specific = self.conf.get_bool('dataset.category_specific')
        # self.num_category = self.conf.get_int('dataset.num_category')
        # self.category_latent_dim = self.conf.get_int('dataset.category_latent_dim')
        self.optimize_latent_code = self.conf.get_bool('train.optimize_latent_code')
        self.optimize_scene_latent_code = self.conf.get_bool('train.optimize_scene_latent_code')
        self.target_training = self.conf.get_bool('train.target_training')
        self.multi_source_training = self.conf.get_bool('train.multi_source_training', default=False)

        self.print_info = True
        self.dataset_train_subdir = self.conf.get_list('dataset.train.sub_dir')
        
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

        # os.environ['WANDB_DIR'] = os.path.join(self.exps_folder_name)
        # wandb.init(project=self.conf.get_string('train.projectname'), name=self.subject + '_' + self.methodname, config=self.conf, tags=kwargs['wandb_tags'])

        # NOTE customize in self.optimize_inputs
        self.optimize_inputs = self.optimize_expression or self.optimize_pose or self.optimize_latent_code or self.optimize_scene_latent_code
        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.dataset_train_subdir)

        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        # if kwargs['is_continue']:
        #     if kwargs['load_path'] != '':
        #         load_path = kwargs['load_path']
        #     else:
        #         load_path = self.train_dir
        #     if os.path.exists(os.path.join(load_path)):
        #         is_continue = True
        #     else:
        #         is_continue = False
        # else:
        #     is_continue = False
        
        if is_main_process():
            utils.mkdir_ifnotexists(self.train_dir)
            utils.mkdir_ifnotexists(self.eval_dir)
        if self.distributed:
            torch.distributed.barrier()

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.train_dir, 'checkpoints')
        if is_main_process():
            utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        if is_main_process():
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))
        if self.distributed:
            torch.distributed.barrier()

        if self.optimize_inputs:
            self.optimizer_inputs_subdir = "OptimizerInputs"
            self.input_params_subdir = "InputParameters"

            if is_main_process():
                utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir))
                utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.input_params_subdir))
            if self.distributed:
                torch.distributed.barrier()
        
        is_continue = False
        for filename in os.listdir(os.path.join(self.checkpoints_path, self.model_params_subdir)):
            if filename.endswith(".pth"):
                is_continue = True
                
        # os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.train_dir, 'runconf.conf')))
        
        self.file_backup(kwargs['conf'])

        print('shell command : {0}'.format(' '.join(sys.argv)))
        print('Loading data ...')

        self.use_background = self.conf.get_bool('dataset.use_background', default=False)
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(conf=self.conf,
                                                                                          mode='train',
                                                                                          data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                          subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                          json_name=self.conf.get_string('dataset.json_name'),
                                                                                          use_mean_expression=self.conf.get_bool('dataset.use_mean_expression', default=False),
                                                                                          use_var_expression=self.conf.get_bool('dataset.use_var_expression', default=False),
                                                                                          use_background=self.use_background,
                                                                                          **self.conf.get_config('dataset.train'))

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(conf=self.conf,
                                                                                         mode='test',
                                                                                         data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         use_background=self.use_background,
                                                                                         **self.conf.get_config('dataset.test'))
        print('Finish loading data ...')

        self.train_from_scratch = False

        if is_continue:
            print('[INFO] continue training...')
            pcd_init = {}
            saved_model_state = torch.load(
                os.path.join(self.checkpoints_path, self.model_params_subdir, '{}.pth'.format(str(kwargs['checkpoint']))), map_location=cuda)

            pcd_init['n_init_points'] = saved_model_state["model_state_dict"]['pc.points'].shape[0]
            pcd_init['init_radius'] = saved_model_state['radius']

        else:
            meta_learning = self.conf.get_bool('train.meta_learning.load_from_meta_learning')
            meta_learning_path = self.conf.get_string('train.meta_learning.path')
            
            if meta_learning:
                meta_learning_filename = find_checkpoint_file(meta_learning_path, self.conf.get_string('train.meta_learning.epoch_num'))
                print('[INFO] meta learning from {}'.format(meta_learning_path))
                pcd_init = {}
                pretrain_path = os.path.join(meta_learning_path)

                saved_model_state = torch.load(os.path.join(pretrain_path, 'ModelParameters', meta_learning_filename), map_location=cuda)
                saved_input_state = torch.load(os.path.join(pretrain_path, 'InputParameters', meta_learning_filename), map_location=cuda)
                pcd_init['n_init_points'] = saved_model_state["model_state_dict"]['pc.points'].shape[0]
                pcd_init['init_radius'] = saved_model_state['radius']
                self.start_epoch = saved_model_state['epoch']
            else:
                print('[INFO] training from scratch...')
                pcd_init = None
                self.train_from_scratch = True

        latent_code_dim = (self.scene_latent_dim + self.category_latent_dim) * len(self.category_dict)
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf, 
                                                                                shape_params=self.train_dataset.shape_params,
                                                                                img_res=self.train_dataset.img_res,
                                                                                canonical_expression=self.train_dataset.mean_expression,
                                                                                canonical_pose=self.conf.get_float('dataset.canonical_pose', default=0.2),
                                                                                use_background=self.use_background,
                                                                                checkpoint_path=kwargs['path_ckpt'],
                                                                                latent_code_dim=latent_code_dim,
                                                                                pcd_init=pcd_init)
        
        if self.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            print('[INFO] current rank: {}'.format(opt.local_rank))
            self.model = torch.nn.parallel.DistributedDataParallel(self.model.to(device=cuda), device_ids=[opt.local_rank], find_unused_parameters=True)
        else:
            self.model = self.model.to(device=cuda)

        self._init_dataloader()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'), 
                                                                              var_expression=self.train_dataset.var_expression,
                                                                              optimize_scene_latent_code=self.optimize_scene_latent_code)

        self.lr = self.conf.get_float('train.learning_rate')
        
        if self.distributed:
            self.optimizer = torch.optim.Adam([
                {'params': list(self.model.module.parameters())},
            ], lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam([
                {'params': list(self.model.parameters())},
            ], lr=self.lr)

        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)
        self.upsample_freq = self.conf.get_int('train.upsample_freq', default=5)
        # settings for input parameter optimization
        if self.optimize_inputs:
            if self.distributed:
                sparse = False
            else:
                sparse = True
            num_training_frames = len(self.train_dataset)
            param = []
            if self.optimize_expression:
                if self.distributed:
                    init_expression = torch.cat((self.train_dataset.data["expressions"], torch.randn(self.train_dataset.data["expressions"].shape[0], max(self.model.module.deformer_network.num_exp - 50, 0)).float()), dim=1)
                    self.expression = torch.nn.Embedding(num_training_frames, self.model.module.deformer_network.num_exp, _weight=init_expression, sparse=sparse).to(device=cuda)
                else:
                    init_expression = torch.cat((self.train_dataset.data["expressions"], torch.randn(self.train_dataset.data["expressions"].shape[0], max(self.model.deformer_network.num_exp - 50, 0)).float()), dim=1)
                    self.expression = torch.nn.Embedding(num_training_frames, self.model.deformer_network.num_exp, _weight=init_expression, sparse=sparse).to(device=cuda)
                param += list(self.expression.parameters())

            if self.optimize_pose:
                self.flame_pose = torch.nn.Embedding(num_training_frames, 15, _weight=self.train_dataset.data["flame_pose"], sparse=sparse).to(device=cuda)
                self.camera_pose = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.data["world_mats"][:, :3, 3], sparse=sparse).to(device=cuda)
                param += list(self.flame_pose.parameters()) + list(self.camera_pose.parameters())
            
            # hyunsoo added
            if self.optimize_latent_code:
                self.latent_codes = torch.nn.Embedding(num_training_frames, self.latent_dim, sparse=sparse).to(device=cuda)
                torch.nn.init.uniform_(
                    self.latent_codes.weight.data,
                    0.0,
                    1.0,
                )
                param += list(self.latent_codes.parameters())
                if is_main_process():
                    print('[DEBUG] Latent code is used. The latent dimension is {0}x{1}.'.format(num_training_frames, self.latent_dim))
            
            if self.optimize_scene_latent_code:
                self.target_training_latent_freeze = self.conf.get_bool('train.target_training_latent_freeze', default=False)
                # NOTE category에 없을 때 존재하는 latent code. 이걸로 일단 초기화한다.
                self.zero_latent_codes = torch.nn.Embedding(1, self.scene_latent_dim, sparse=sparse).to(device=cuda)
                torch.nn.init.normal_(
                    self.zero_latent_codes.weight.data,
                    0.0,
                    0.25,
                )
                if not self.target_training_latent_freeze:
                    param += list(self.zero_latent_codes.parameters())

                # NOTE 카테고리 레이턴트 코드도 러너블한 방식으로 바꿈.
                # self.category_latent_codes = torch.nn.Embedding(len(self.category_dict), self.category_latent_dim, sparse=sparse).to(device=cuda)
                # torch.nn.init.normal_(
                #     self.category_latent_codes.weight.data,
                #     0.0,
                #     0.25,
                # )
                # param += list(self.category_latent_codes.parameters())

                if self.target_training:
                    # NOTE train subdir도 target human 하나만 있으므로 그대로 사용해도 됨.
                    assert len(self.dataset_train_subdir) == 1, "[INFO] Target training's train subdir should be one."
                    # NOTE 얘는 뭐지???
                    # self.target_scene_latent_codes = torch.nn.Embedding(len(self.dataset_train_subdir), self.scene_latent_dim, sparse=sparse).to(device=cuda)
                    # torch.nn.init.normal_(
                    #     self.target_scene_latent_codes.weight.data,
                    #     0.0,
                    #     0.25,
                    # )
                    # param += list(self.target_scene_latent_codes.parameters())
                    
                    # NOTE TH에 고유하게 존재하는 latent code. 이름을 개명했다.
                    self.target_scene_latent_codes = torch.nn.Embedding(len(self.target_category_dict.keys()) * len(self.dataset_train_subdir), self.scene_latent_dim, sparse=sparse).to(device=cuda)
                    torch.nn.init.normal_(
                        self.target_scene_latent_codes.weight.data,
                        0.0,
                        0.25,
                    )
                    if not self.target_training_latent_freeze:
                        param += list(self.target_scene_latent_codes.parameters())
                        
                elif self.multi_source_training:
                    # NOTE Source Human (Guy)를 위한 latent code. [7, 32]
                    self.source_scene_latent_codes = torch.nn.Embedding(len(self.source_category_dict.keys()), self.scene_latent_dim, sparse=sparse).to(device=cuda)
                    torch.nn.init.normal_(
                        self.source_scene_latent_codes.weight.data,
                        0.0,
                        0.25,
                    )
                    param += list(self.source_scene_latent_codes.parameters())

                    # NOTE Multi Source를 위한 latent code. [7, 32]*21
                    self.multi_source_without_hat_scene_latent_codes = []

                    for subdir in self.no_hat_dataset_train_subdir:
                        multi_source_without_hat_scene_latent_codes = torch.nn.Embedding(len(self.without_hat_category_dict.keys()), self.scene_latent_dim, sparse=sparse).to(device=cuda)
                        torch.nn.init.normal_(
                            multi_source_without_hat_scene_latent_codes.weight.data,
                            0.0,
                            0.25,
                        )
                        param += list(multi_source_without_hat_scene_latent_codes.parameters())
                        self.multi_source_without_hat_scene_latent_codes.append(multi_source_without_hat_scene_latent_codes)

                    self.multi_source_with_hat_scene_latent_codes = []
                    for subdir in self.hat_dataset_train_subdir:
                        multi_source_with_hat_scene_latent_codes = torch.nn.Embedding(len(self.with_hat_category_dict.keys()), self.scene_latent_dim, sparse=sparse).to(device=cuda)
                        torch.nn.init.normal_(
                            multi_source_with_hat_scene_latent_codes.weight.data,
                            0.0,
                            0.25,
                        )
                        param += list(multi_source_with_hat_scene_latent_codes.parameters())
                        self.multi_source_with_hat_scene_latent_codes.append(multi_source_with_hat_scene_latent_codes)

                    # for subdir in self.dataset_train_subdir:
                    #     if subdir not in self.include_hat_datasets:
                    #         multi_source_without_hat_scene_latent_codes = torch.nn.Embedding(len(self.without_hat_category_dict.keys()), self.scene_latent_dim, sparse=sparse).to(device=cuda)
                    #         torch.nn.init.normal_(
                    #             multi_source_without_hat_scene_latent_codes.weight.data,
                    #             0.0,
                    #             0.25,
                    #         )
                    #         param += list(multi_source_without_hat_scene_latent_codes.parameters())
                    #         self.multi_source_without_hat_scene_latent_codes.append(multi_source_without_hat_scene_latent_codes)

                    # NOTE Multi Source를 위한 latent code. [8, 32]*21
                    # self.multi_source_with_hat_scene_latent_codes = []
                    # for subdir in self.dataset_train_subdir:
                    #     if subdir in self.include_hat_datasets:
                    #         multi_source_with_hat_scene_latent_codes = torch.nn.Embedding(len(self.with_hat_category_dict.keys()), self.scene_latent_dim, sparse=sparse).to(device=cuda)
                    #         torch.nn.init.normal_(
                    #             multi_source_with_hat_scene_latent_codes.weight.data,
                    #             0.0,
                    #             0.25,
                    #         )
                    #         param += list(multi_source_with_hat_scene_latent_codes.parameters())
                    #         self.multi_source_with_hat_scene_latent_codes.append(multi_source_with_hat_scene_latent_codes)

                    if is_main_process():
                        print('[DEBUG] Scene latent code is used. The latent dimension is {0}x{1}.'.format(len(self.dataset_train_subdir), self.scene_latent_dim))
                else:
                    # NOTE SH에 고유하게 존재하는 latent code.
                    self.source_scene_latent_codes = torch.nn.Embedding(len(self.source_category_dict.keys()), self.scene_latent_dim, sparse=sparse).to(device=cuda)
                    torch.nn.init.normal_(
                        self.source_scene_latent_codes.weight.data,
                        0.0,
                        0.25,
                    )
                    param += list(self.source_scene_latent_codes.parameters())

                    # NOTE DB마다 하나씩 존재하는 latent code.
                    self.scene_latent_codes = torch.nn.Embedding(len(self.dataset_train_subdir), self.scene_latent_dim, sparse=sparse).to(device=cuda)
                    torch.nn.init.normal_(
                        self.scene_latent_codes.weight.data,
                        0.0,
                        0.25,
                    )
                    param += list(self.scene_latent_codes.parameters())
                    if is_main_process():
                        print('[DEBUG] Scene latent code is used. The latent dimension is {0}x{1}.'.format(len(self.dataset_train_subdir), self.scene_latent_dim))

            if self.distributed:
                self.optimizer_cam = torch.optim.Adam(param, self.conf.get_float('train.learning_rate_cam'))
            else:
                if not self.target_training_latent_freeze:
                    self.optimizer_cam = torch.optim.SparseAdam(param, self.conf.get_float('train.learning_rate_cam'))

        self.start_epoch = 0

        if is_continue:
            old_checkpnts_dir = self.checkpoints_path
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"), map_location=cuda)
            self.start_epoch = saved_model_state['epoch']
            n_points = saved_model_state["model_state_dict"]['pc.points'].shape[0]
            if is_main_process():
                print("[INFO] n_points: {}".format(n_points))
            batch_size = min(int(self.conf.get_int('train.max_points_training') / n_points), self.max_batch)
            if self.batch_size != batch_size:
                self.batch_size = batch_size
                self._init_dataloader()

            if self.distributed:
                self.model.module.pc.init(n_points)
                self.model.module.pc = self.model.module.pc.to(device=cuda)
                self.model.module.load_state_dict(saved_model_state["model_state_dict"], strict=False)
                self.model.module.raster_settings.radius = saved_model_state['radius']
                self.optimizer = torch.optim.Adam([
                    {'params': list(self.model.module.parameters())},
                ], lr=self.lr)

            else:
                self.model.pc.init(n_points)
                self.model.pc = self.model.pc.to(device=cuda)
                self.model.load_state_dict(saved_model_state["model_state_dict"], strict=False)
                self.model.raster_settings.radius = saved_model_state['radius']
                self.optimizer = torch.optim.Adam([
                    {'params': list(self.model.parameters())},
                ], lr=self.lr)

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"), map_location=cuda)
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            if self.optimize_inputs:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_inputs_subdir, str(kwargs['checkpoint']) + ".pth"), map_location=cuda)
                try:
                    self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])
                except:
                    if is_main_process():
                        print("[WARN] input and camera optimizer parameter group doesn't match")
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.input_params_subdir, str(kwargs['checkpoint']) + ".pth"), map_location=cuda)
                try:
                    if self.optimize_expression:
                        self.expression.load_state_dict(data["expression_state_dict"])
                    if self.optimize_pose:
                        self.flame_pose.load_state_dict(data["flame_pose_state_dict"])
                        self.camera_pose.load_state_dict(data["camera_pose_state_dict"])
                except:
                    if is_main_process():
                        print("[WARN] expression or pose parameter group doesn't match")

                # hyunsoo added
                if self.optimize_latent_code:
                    self.latent_codes.load_state_dict(data["latent_codes_state_dict"])
                    if is_main_process():
                        print('[DEBUG] loaded latent code successfully.')
                # if self.optimize_scene_latent_code:
                #     if self.target_training:
                #         self.target_scene_latent_codes.load_state_dict(data["target_scene_latent_codes_state_dict"])
                #         if is_main_process():
                #             print('[DEBUG] loaded target scene latent code successfully.')
                #         self.target_scene_latent_category_codes.load_state_dict(data["target_scene_latent_category_codes_state_dict"])
                #     else:
                #         self.scene_latent_codes.load_state_dict(data["scene_latent_codes_state_dict"])
                #     print('[DEBUG] loaded scene latent code successfully.')
                #     if self.category_specific:
                #         if not self.target_training:
                #             self.source_scene_latent_codes.load_state_dict(data["source_scene_latent_codes_state_dict"])
                #             print('[DEBUG] loaded source scene latent code successfully.')
                #             try:
                #                 self.source_scene_latent_category_codes.load_state_dict(data["source_scene_latent_category_codes_state_dict"])
                #             except:
                #                 print("[WARN] source scene latent category code parameter group doesn't match")
                        
                #         self.zero_latent_codes.load_state_dict(data["zero_latent_codes_state_dict"])
                if self.optimize_scene_latent_code:
                    self.zero_latent_codes.load_state_dict(data["zero_latent_codes_state_dict"])
                    # self.category_latent_codes.load_state_dict(data["category_latent_codes_state_dict"])
                    if self.target_training:
                        self.target_scene_latent_codes.load_state_dict(data["target_scene_latent_codes_state_dict"])
                        # self.target_scene_latent_category_codes.load_state_dict(data["target_scene_latent_category_codes_state_dict"])
                    elif self.multi_source_training:
                        self.source_scene_latent_codes.load_state_dict(data["source_scene_latent_codes_state_dict"])

                        for idx, subdir in enumerate(self.hat_dataset_train_subdir):
                            self.multi_source_with_hat_scene_latent_codes[idx].load_state_dict(data["multi_source_with_hat_scene_latent_codes_{}_state_dict".format(subdir)])

                        for idx, subdir in enumerate(self.no_hat_dataset_train_subdir):
                            self.multi_source_without_hat_scene_latent_codes[idx].load_state_dict(data["multi_source_without_hat_scene_latent_codes_{}_state_dict".format(subdir)])

                        # i_wo_hat, i_w_hat = 0, 0
                        # for _, subdir in enumerate(self.dataset_train_subdir):
                        #     if subdir not in self.include_hat_datasets:
                        #         self.multi_source_without_hat_scene_latent_codes[i_wo_hat].load_state_dict(data["multi_source_without_hat_scene_latent_codes_{}_state_dict".format(subdir)])
                        #         i_wo_hat += 1
                        #     else:
                        #         self.multi_source_with_hat_scene_latent_codes[i_w_hat].load_state_dict(data["multi_source_with_hat_scene_latent_codes_{}_state_dict".format(subdir)])
                        #         i_w_hat += 1
                    else:
                        self.scene_latent_codes.load_state_dict(data["scene_latent_codes_state_dict"])
                        self.source_scene_latent_codes.load_state_dict(data["source_scene_latent_codes_state_dict"])

        else:
            if meta_learning:
                self.start_epoch = saved_model_state['epoch']
                n_points = saved_model_state["model_state_dict"]['pc.points'].shape[0]
                print("[INFO] n_points: {}".format(n_points))
                batch_size = min(int(self.conf.get_int('train.max_points_training') / n_points), self.max_batch)
                if self.batch_size != batch_size:
                    self.batch_size = batch_size
                    self._init_dataloader()
                
                if self.distributed:
                    self.model.module.pc.init(n_points)
                    self.model.module.pc = self.model.module.pc.to(device=cuda)
                    self.model.module.load_state_dict(saved_model_state["model_state_dict"], strict=True)
                    self.model.module.raster_settings.radius = saved_model_state['radius']
                    self.optimizer = torch.optim.Adam([
                        {'params': list(self.model.module.parameters())},
                    ], lr=self.lr)
                    self.model.module.visible_points = torch.zeros(self.model.module.pc.points.shape[0]).bool().cuda()
                else:
                    self.model.pc.init(n_points)
                    self.model.pc = self.model.pc.to(device=cuda)
                    self.model.load_state_dict(saved_model_state["model_state_dict"], strict=True)
                    self.model.raster_settings.radius = saved_model_state['radius']
                    # self.model.raster_settings.radius = saved_model_state['state_dict']['model.pc.radius'].item()

                    self.optimizer = torch.optim.Adam([
                        {'params': list(self.model.parameters())},
                    ], lr=self.lr)
                    self.model.visible_points = torch.zeros(self.model.pc.points.shape[0]).bool().cuda()

            else:
                print('[INFO] No pretrain model is used.')
        
        # if self.distributed:
        #     train_sampler = DistributedSampler(dataset=self.train_dataset, shuffle=True)
        #     self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
        #                                                         batch_size=self.batch_size,
        #                                                         # shuffle=True,
        #                                                         collate_fn=self.train_dataset.collate_fn,
        #                                                         num_workers=4,
        #                                                         sampler=train_sampler,
        #                                                         )
        # else:
        #     self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
        #                                                         batch_size=self.batch_size,
        #                                                         shuffle=True,
        #                                                         collate_fn=self.train_dataset.collate_fn,
        #                                                         num_workers=4,
        #                                                         )
        # self.n_batches = len(self.train_dataloader)
        self.img_res = self.plot_dataset.img_res
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.log_freq = self.conf.get_int('train.log_freq')

        self.GT_lbs_milestones = self.conf.get_list('train.GT_lbs_milestones', default=[])
        self.GT_lbs_factor = self.conf.get_float('train.GT_lbs_factor', default=0.5)
        for acc in self.GT_lbs_milestones:
            if self.start_epoch > acc:
                self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor
        # if len(self.GT_lbs_milestones) > 0 and self.start_epoch >= self.GT_lbs_milestones[-1]:
        #    self.loss.lbs_weight = 0.
        if self.accelerate:
            self.model, self.optimizer, self.train_dataloader, self.plot_dataloader = self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.plot_dataloader)
        
        self.start_time = torch.cuda.Event(enable_timing=True)
        self.end_time = torch.cuda.Event(enable_timing=True)

        self.start_time_step = torch.cuda.Event(enable_timing=True)
        self.end_time_step = torch.cuda.Event(enable_timing=True)

        # MVDream related
        # from omegaconf import OmegaConf
        # cfg = OmegaConf.load('/media/ssd2/hyunsoocha/GitHub/PointAvatar/code/configs/mvdream-sd21.yaml')
        # self.guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
        # self.guidance.requires_grad_(False)

    # NOTE related to mvdream
    def on_load_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                return
        guidance_state_dict = {"guidance."+k : v for (k,v) in self.guidance.state_dict().items()}
        checkpoint['state_dict'] = {**checkpoint['state_dict'], **guidance_state_dict}
        return 

    # NOTE related to mvdream
    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                checkpoint['state_dict'].pop(k)
        return 

    def _init_dataloader(self):
        if self.distributed:
            train_sampler = DistributedSampler(dataset=self.train_dataset, shuffle=True)
            test_sampler = DistributedSampler(dataset=self.plot_dataset, shuffle=False)
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                                batch_size=self.batch_size,
                                                                collate_fn=self.train_dataset.collate_fn,
                                                                num_workers=0,
                                                                sampler=train_sampler,
                                                                )
            self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                                batch_size=self.batch_size, # min(self.batch_size * 2, 10),
                                                                collate_fn=self.plot_dataset.collate_fn,
                                                                num_workers=4,
                                                                sampler=test_sampler,
                                                                )
        else:
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                                batch_size=self.batch_size,
                                                                shuffle=True,
                                                                collate_fn=self.train_dataset.collate_fn,
                                                                num_workers=0,
                                                                )
            self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                            batch_size=self.batch_size, # min(self.batch_size * 2, 10),
                                                            shuffle=False,
                                                            collate_fn=self.plot_dataset.collate_fn,
                                                            num_workers=4,
                                                            )
        self.n_batches = len(self.train_dataloader)

    def save_checkpoints(self, epoch, only_latest=False):
        if not only_latest:
            if self.distributed:
                torch.save(
                    {"epoch": epoch, "radius": self.model.module.raster_settings.radius,
                    "model_state_dict": self.model.module.state_dict()},
                    os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
            else:
                torch.save(
                    {"epoch": epoch, "radius": self.model.raster_settings.radius,
                    "model_state_dict": self.model.state_dict()},
                    os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
                os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))

        if self.distributed:
            torch.save(
                {"epoch": epoch, "radius": self.model.module.raster_settings.radius,
                "model_state_dict": self.model.module.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        else:
            torch.save(
                {"epoch": epoch, "radius": self.model.raster_settings.radius,
                "model_state_dict": self.model.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        if self.optimize_inputs:
            dict_to_save = {}
            dict_to_save["epoch"] = epoch
            if self.optimize_expression:
                dict_to_save["expression_state_dict"] = self.expression.state_dict()
            if self.optimize_pose:
                dict_to_save["flame_pose_state_dict"] = self.flame_pose.state_dict()
                dict_to_save["camera_pose_state_dict"] = self.camera_pose.state_dict()
            
            # hyunsoo added
            if self.optimize_latent_code:
                dict_to_save["latent_codes_state_dict"] = self.latent_codes.state_dict()

            if self.optimize_scene_latent_code:
                dict_to_save["zero_latent_codes_state_dict"] = self.zero_latent_codes.state_dict()
                # dict_to_save["category_latent_codes_state_dict"] = self.category_latent_codes.state_dict()
                if self.target_training:
                    dict_to_save["target_scene_latent_codes_state_dict"] = self.target_scene_latent_codes.state_dict()
                    # dict_to_save["target_scene_latent_category_codes_state_dict"] = self.target_scene_latent_category_codes.state_dict()
                elif self.multi_source_training:
                    dict_to_save["source_scene_latent_codes_state_dict"] = self.source_scene_latent_codes.state_dict()

                    for idx, subdir in enumerate(self.hat_dataset_train_subdir):
                        dict_to_save["multi_source_with_hat_scene_latent_codes_{}_state_dict".format(subdir)] = self.multi_source_with_hat_scene_latent_codes[idx].state_dict()
                    
                    for idx, subdir in enumerate(self.no_hat_dataset_train_subdir):
                        dict_to_save["multi_source_without_hat_scene_latent_codes_{}_state_dict".format(subdir)] = self.multi_source_without_hat_scene_latent_codes[idx].state_dict()

                    # i_wo_hat, i_w_hat = 0, 0
                    # for _, subdir in enumerate(self.dataset_train_subdir):
                    #     if subdir not in self.include_hat_datasets:
                    #         dict_to_save["multi_source_without_hat_scene_latent_codes_{}_state_dict".format(subdir)] = self.multi_source_without_hat_scene_latent_codes[i_wo_hat].state_dict()
                    #         i_wo_hat += 1
                    #     else:
                    #         dict_to_save["multi_source_with_hat_scene_latent_codes_{}_state_dict".format(subdir)] = self.multi_source_with_hat_scene_latent_codes[i_w_hat].state_dict()
                    #         i_w_hat += 1
                else:
                    dict_to_save["scene_latent_codes_state_dict"] = self.scene_latent_codes.state_dict()
                    dict_to_save["source_scene_latent_codes_state_dict"] = self.source_scene_latent_codes.state_dict()

            if not only_latest:
                torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, str(epoch) + ".pth"))
                if not self.target_training_latent_freeze:
                    torch.save(
                        {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                        os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir, str(epoch) + ".pth"))
                
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, "latest.pth"))
            if not self.target_training_latent_freeze:
                torch.save(
                    {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                    os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir, "latest.pth"))

    def upsample_points(self):
        if self.distributed:
            current_radius = self.model.module.raster_settings.radius
            points = self.model.module.pc.points.data

            # if is_main_process():           # NOTE new_points를 만들어주는 과정
            #     if self.model.module.pc.points.shape[0] <= self.model.module.pc.max_points / 2:           # 2배씩 증가
            #         noise = (torch.rand(*points.shape, device=points.device) - 0.5) * current_radius
            #         new_points = noise + points
            #     else:
            #         new_points_shape = (self.model.module.pc.max_points - points.shape[0], points.shape[1])
            #         noise = (torch.rand(new_points_shape, device=points.device) - 0.5) * current_radius     # NOTE cuda 삭제

            #         indices = torch.randperm(points.size(0))
            #         selected_points = points[indices[:new_points_shape[0]]]
            #         new_points = noise + selected_points
            # else:
            #     if self.model.module.pc.points.shape[0] <= self.model.module.pc.max_points / 2:
            #         new_points = torch.empty_like(points)
            #     else:
            #         new_points_shape = (self.model.module.pc.max_points - points.shape[0], points.shape[1])
            #         new_points = torch.empty(new_points_shape)

            if self.model.module.pc.points.shape[0] <= self.model.module.pc.max_points / 2:           # 2배씩 증가
                noise = (torch.rand(*points.shape, device=points.device) - 0.5) * current_radius
                new_points = noise + points
            else:
                new_points_shape = (self.model.module.pc.max_points - points.shape[0], points.shape[1])
                noise = (torch.rand(new_points_shape, device=points.device) - 0.5) * current_radius     # NOTE cuda 삭제

                indices = torch.randperm(points.size(0))
                selected_points = points[indices[:new_points_shape[0]]]
                new_points = noise + selected_points

            # 주 GPU의 결과를 다른 GPU와 동기화
            dist.barrier()
            dist.broadcast(new_points, src=0)
            self.model.module.pc.upsample_points(new_points)
            # self.model.module.pc.upsample_points_with_the_current(points, new_points)
            
            if self.model.module.pc.min_radius < current_radius:
                new_radius = 0.75 * current_radius
                # self.model.module.pc.register_radius(new_radius)
                self.model.module.raster_settings.radius = new_radius
            print("***************************************************")
            print("old radius: {}, new radius: {}".format(current_radius, self.model.module.raster_settings.radius))
            # print("old points: {}, new points: {}".format(self.model.pc.points.data.shape[0]/2, self.model.pc.points.data.shape[0]))      # NOTE original code
            print("old points: {}, new points: {}".format(points.shape[0], self.model.module.pc.points.data.shape[0]))                             # NOTE custom code. 더 정확하게 표시하기 위해서. 정말 upsample이 된게 맞나.     
            print("***************************************************") 
            self.optimizer = torch.optim.Adam([
                {'params': list(self.model.parameters())},
            ], lr=self.lr)
        else:
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
                # self.model.pc.register_radius(new_radius)
                self.model.raster_settings.radius = new_radius
            print("***************************************************")
            print("old radius: {}, new radius: {}".format(current_radius, self.model.raster_settings.radius))
            # print("old points: {}, new points: {}".format(self.model.pc.points.data.shape[0]/2, self.model.pc.points.data.shape[0]))      # NOTE original code
            print("old points: {}, new points: {}".format(points.shape[0], self.model.pc.points.data.shape[0]))                             # NOTE custom code. 더 정확하게 표시하기 위해서. 정말 upsample이 된게 맞나.     
            print("***************************************************") 
            self.optimizer = torch.optim.Adam([
                {'params': list(self.model.parameters())},
            ], lr=self.lr)

    # def file_backup(self):
    #     from shutil import copyfile
    #     dir_lis = ['./model', './scripts', './utils', './flame', './datasets']
    #     os.makedirs(os.path.join(self.train_dir, 'recording'), exist_ok=True)
    #     for dir_name in dir_lis:
    #         cur_dir = os.path.join(self.train_dir, 'recording', dir_name)
    #         os.makedirs(cur_dir, exist_ok=True)
    #         files = os.listdir(dir_name)
    #         for f_name in files:
    #             if f_name[-3:] == '.py':
    #                 copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

    #     # copyfile(self.conf_path, os.path.join(self.train_dir, 'recording', 'config.conf'))

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

    def validation(self, epoch, step=None):
        eval_iterator = iter(self.plot_dataloader)
        self.start_time.record()
        
        novel_view_type = 'validation'

        indices, model_input, ground_truth = next(eval_iterator)

        model_input['cam_pose'][-1, -1, -1] += 1.5

        if step is None:
            data_index = 1
            step = epoch * len(self.train_dataset) + data_index * self.batch_size

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
            if self.optimize_scene_latent_code:
                if self.target_training:
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

                    for i, v in enumerate(target_index_list):
                        target_start_idx = v*learnable_tensor_size
                        target_end_idx = (v+1)*learnable_tensor_size
                        input_latent_codes[:, target_start_idx:target_end_idx] = torch.cat((target_category_latent_codes[i], self.target_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                    # Add to the model_input dictionary
                    model_input['scene_latent_code'] = input_latent_codes.cuda()
                
                elif self.multi_source_training:
                    learnable_tensor_size = self.category_latent_dim + self.scene_latent_dim

                    input_latent_codes = torch.zeros(len(model_input['sub_dir']), learnable_tensor_size*len(self.category_dict))

                    if model_input['sub_dir'] == self.source_datasets:
                        # indices = [self.dataset_train_subdir.index(name) for name in model_input['sub_dir']]

                        # indices_tensor = torch.tensor(indices, dtype=torch.long, device=model_input['idx'].device)
                        # scene_latent_codes_tensor = self.scene_latent_codes(indices_tensor)         # [B, 28] 데이터셋 폴더마다 한개씩 존재
                        
                        # index_list = []
                        # for sub_dir_item in model_input['sub_dir']:
                        #     category_idx = self.category_dict[sub_dir_item.split('_')[0]]
                        #     index_list.append(category_idx)
                        # category_latent_codes = F.one_hot(torch.tensor(index_list), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(index_list).cuda())
                        # scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

                        # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. [B, 320]
                        # input_latent_codes = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))
                        
                        # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                        for i, v in self.category_dict.items():
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                            start_idx = v* learnable_tensor_size
                            end_idx = (v+1)* learnable_tensor_size
                            input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                            
                        # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                        for i, v in enumerate(self.source_category_dict.values()):
                            category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                            source_start_idx = v* learnable_tensor_size
                            source_end_idx = (v+1)* learnable_tensor_size
                            input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)
                        
                        # Add to the model_input dictionary
                        model_input['scene_latent_code'] = input_latent_codes.cuda()


                    elif model_input['sub_dir'][0] in self.hat_dataset_train_subdir:            # NOTE hat가 있는 dataset.
                        # indices_tensor = torch.tensor(indices, dtype=torch.long, device=model_input['idx'].device)
                        # scene_latent_codes_tensor = self.scene_latent_codes(indices_tensor)         # [B, 28] 데이터셋 폴더마다 한개씩 존재
                        
                        # index_list = []
                        # for sub_dir_item in model_input['sub_dir']:
                        #     category_idx = self.category_dict[sub_dir_item.split('_')[0]]
                        #     index_list.append(category_idx)
                        # category_latent_codes = F.one_hot(torch.tensor(index_list), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(index_list).cuda())
                        # scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

                        indices = [self.hat_dataset_train_subdir.index(name) for name in model_input['sub_dir']]
                        multi_source_with_hat_scene_latent_codes = self.multi_source_with_hat_scene_latent_codes[indices[0]]

                        # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. [B, 320]
                        # input_latent_codes = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))
                        
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
                        
                        # NOTE 마지막으로 db human의 latent code를 넣어준다.
                        # for i in range(len(index_list)):
                        #     start_idx = index_list[i]*scene_latent_codes_tensor.shape[1]
                        #     end_idx = (index_list[i]+1)*scene_latent_codes_tensor.shape[1]
                        #     input_latent_codes[i, start_idx:end_idx] = scene_latent_codes_tensor[i]

                        # Add to the model_input dictionary
                        model_input['scene_latent_code'] = input_latent_codes.cuda()
                    else:       # NOTE 모자가 없는 dataset.
                        # indices = [self.not_include_hat_datasets.index(name) for name in model_input['sub_dir']]

                        # indices_tensor = torch.tensor(indices, dtype=torch.long, device=model_input['idx'].device)
                        # scene_latent_codes_tensor = self.scene_latent_codes(indices_tensor)         # [B, 28] 데이터셋 폴더마다 한개씩 존재
                        
                        # index_list = []
                        # for sub_dir_item in model_input['sub_dir']:
                        #     category_idx = self.category_dict[sub_dir_item.split('_')[0]]
                        #     index_list.append(category_idx)
                        # category_latent_codes = F.one_hot(torch.tensor(index_list), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(index_list).cuda())
                        # scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

                        # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. [B, 320]
                        # input_latent_codes = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))
                        
                        indices = [self.no_hat_dataset_train_subdir.index(name) for name in model_input['sub_dir']]
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
                        
                        # NOTE 마지막으로 db human의 latent code를 넣어준다.
                        # for i in range(len(index_list)):
                        #     start_idx = index_list[i]*scene_latent_codes_tensor.shape[1]
                        #     end_idx = (index_list[i]+1)*scene_latent_codes_tensor.shape[1]
                        #     input_latent_codes[i, start_idx:end_idx] = scene_latent_codes_tensor[i]

                        # Add to the model_input dictionary
                        model_input['scene_latent_code'] = input_latent_codes.cuda()
                
                else:
                    # indices = list(range(0, len(self.dataset_train_subdir), 4))[:self.batch_size] # [self.dataset_train_subdir.index(name) for name in model_input['sub_dir']]
                    indices = [self.dataset_train_subdir.index('hat_Syuka_foxhat')]

                    indices_tensor = torch.tensor(indices, dtype=torch.long, device=model_input['idx'].device)
                    scene_latent_codes_tensor = self.scene_latent_codes(indices_tensor)     
                    
                    index_list = []
                    valid_sub_dir = []
                    for idx in indices:
                        valid_sub_dir.append(self.dataset_train_subdir[idx])
                    for sub_dir_item in valid_sub_dir:
                        category_idx = self.category_dict[sub_dir_item.split('_')[0]]
                        index_list.append(category_idx)
                    category_latent_codes = F.one_hot(torch.tensor(index_list), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(index_list).cuda())
                    scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

                    # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. [B, 320]
                    input_latent_codes = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))
                    
                    # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                    for i, v in self.category_dict.items():
                        category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).detach() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                        start_idx = v*scene_latent_codes_tensor.shape[1]
                        end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                        input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                        
                    # NOTE source human의 category latent code를 일괄적으로 만들어준다.
                    source_index_list = []
                    source_category_latent_codes = []
                    for i, v in self.source_category_dict.items():
                        source_index_list.append(v)
                        tensor = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).detach().unsqueeze(0) # self.category_latent_codes(torch.tensor(v).cuda()).detach().unsqueeze(0)
                        source_category_latent_codes.append(tensor)
                    source_category_latent_codes = torch.cat(source_category_latent_codes, dim=0).cuda().detach()
                    
                    # NOTE source의 나머지 latent code들도 넣어준다.
                    for i, v in enumerate(source_index_list):
                        source_start_idx = v*scene_latent_codes_tensor.shape[1]
                        source_end_idx = (v+1)*scene_latent_codes_tensor.shape[1]
                        input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((source_category_latent_codes[i], self.source_scene_latent_codes(torch.tensor(i).cuda()).detach()), dim=0)
                    
                    # NOTE 마지막으로 db human의 latent code를 넣어준다.
                    for i in range(len(index_list)):
                        start_idx = index_list[i]*scene_latent_codes_tensor.shape[1]
                        end_idx = (index_list[i]+1)*scene_latent_codes_tensor.shape[1]
                        input_latent_codes[i, start_idx:end_idx] = scene_latent_codes_tensor[i]

                    # Add to the model_input dictionary
                    model_input['scene_latent_code'] = input_latent_codes.cuda()

            model_outputs = self.model(model_input)
            for k, v in model_outputs.items():
                try:
                    model_outputs[k] = v.detach()
                except:
                    model_outputs[k] = v
            plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(str(epoch), novel_view_type)) for i in range(len(model_input['sub_dir']))]
            img_names = model_input['img_name'][:, 0].cpu().numpy()
            # if step is not None:
            # print("Plotting images: {}".format(img_names))
            tqdm.write("Plotting images: {}".format(os.path.join(os.path.dirname(plot_dir[0]), 'rendering_validation')))
            utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
            
            rendering_switch = {
                'rendering_grid': True,
                'rendering_rgb': False,
                'rendering_rgb_dilate_erode': False,
                'rendering_normal': False,
                'rendering_normal_dilate_erode': False,
                'rendering_albedo': False,
                'rendering_shading': False,
                'rendering_segment': False,
                'rendering_mask_hole': False
            }
            
            plt.plot(img_names,
                    model_outputs,
                    ground_truth,
                    plot_dir,
                    epoch,
                    self.img_res,
                    first=True,
                    custom_settings={'novel_view': novel_view_type, 'step': step, 'rendering_select': rendering_switch})
            del model_outputs, ground_truth, model_input
        
        self.end_time.record()
        torch.cuda.synchronize()
        tqdm.write("Plot time per image: {} ms".format(self.start_time.elapsed_time(self.end_time) / len(self.plot_dataset)))

    def eval_mode(self):
        self.model.eval()
        if self.optimize_inputs:
            if self.optimize_expression:
                self.expression.eval()
            if self.optimize_pose:
                self.flame_pose.eval()
                self.camera_pose.eval()

            # hyunsoo added
            if self.optimize_latent_code:
                self.latent_codes.eval()
            if self.optimize_scene_latent_code:
                self.zero_latent_codes.eval()
                # self.category_latent_codes.eval()
                if self.target_training:
                    self.target_scene_latent_codes.eval()
                    # self.target_scene_latent_category_codes.eval()
                elif self.multi_source_training:
                    self.source_scene_latent_codes.eval()
                    for idx, subdir in enumerate(self.hat_dataset_train_subdir):
                        self.multi_source_with_hat_scene_latent_codes[idx].eval()
                    for idx, subdir in enumerate(self.no_hat_dataset_train_subdir):
                        self.multi_source_without_hat_scene_latent_codes[idx].eval()

                    # i_wo_hat, i_w_hat = 0, 0
                    # for _, subdir in enumerate(self.dataset_train_subdir):
                    #     if subdir not in self.hat_dataset_train_subdir:
                    #         self.multi_source_without_hat_scene_latent_codes[i_wo_hat].eval()
                    #         i_wo_hat += 1
                    #     else:
                    #         self.multi_source_with_hat_scene_latent_codes[i_w_hat].eval()
                    #         i_w_hat += 1
                else:
                    self.source_scene_latent_codes.eval()
                    self.scene_latent_codes.eval()
    
    def train_mode(self):
        self.model.train()
        if self.optimize_inputs:
            if self.optimize_expression:
                self.expression.train()
            if self.optimize_pose:
                self.flame_pose.train()
                self.camera_pose.train()
            
            # hyunsoo added
            if self.optimize_latent_code:
                self.latent_codes.train()
            if self.optimize_scene_latent_code:
                self.zero_latent_codes.train()
                # self.category_latent_codes.train()
                
                if self.target_training:
                    self.target_scene_latent_codes.train()
                    # self.target_scene_latent_category_codes.train()
                elif self.multi_source_training:
                    self.source_scene_latent_codes.train()
                    for idx, subdir in enumerate(self.hat_dataset_train_subdir):
                        self.multi_source_with_hat_scene_latent_codes[idx].train()
                    for idx, subdir in enumerate(self.no_hat_dataset_train_subdir):
                        self.multi_source_without_hat_scene_latent_codes[idx].train()
                        
                    # i_wo_hat, i_w_hat = 0, 0
                    # for _, subdir in enumerate(self.dataset_train_subdir):
                    #     if subdir not in self.include_hat_datasets:
                    #         self.multi_source_without_hat_scene_latent_codes[i_wo_hat].train()
                    #         i_wo_hat += 1
                    #     else:
                    #         self.multi_source_with_hat_scene_latent_codes[i_w_hat].train()
                    #         i_w_hat += 1
                else:
                    self.scene_latent_codes.train()
                    self.source_scene_latent_codes.train()

    def run(self):
        acc_loss = {}
        
        for epoch in range(self.start_epoch, self.nepochs + 1):     
            if self.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch in self.GT_lbs_milestones:
                self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor
            # if len(self.GT_lbs_milestones) > 0 and epoch >= self.GT_lbs_milestones[-1]:
            #   self.loss.lbs_weight = 0.

            if epoch % (self.save_freq * 1) == 0 and epoch != self.start_epoch:
                self.save_checkpoints(epoch)
            else:
                if epoch % self.save_freq == 0 and (epoch != self.start_epoch or self.start_epoch == 0):
                    self.save_checkpoints(epoch, only_latest=True)

            if self.distributed:
                # NOTE 매 epoch마다 point를 동기화.
                points = self.model.module.pc.points.data
                dist.barrier()
                dist.broadcast(points, src=0)
                self.model.module.pc.points_parameter(points)

            if (epoch % self.plot_freq == 0 and epoch < 5) or (epoch % (self.plot_freq) == 0):
                self.eval_mode()
                self.validation(epoch)

            self.train_mode()

            self.start_time.record()

            if self.distributed:
                visible_percentage = (torch.sum(self.model.module.visible_points)/self.model.module.pc.points.shape[0]).unsqueeze(0)
            else:
                visible_percentage = (torch.sum(self.model.visible_points)/self.model.pc.points.shape[0]).unsqueeze(0)

            # Prunning
            if self.distributed:
                if epoch != self.start_epoch and self.model.module.raster_settings.radius >= 0.006:
                    # if (visible_percentage >= 0.8): # and (self.model.pc.points.shape[0] != self.model.pc.max_points):

                    visible_points = self.model.module.visible_points
                    dist.barrier()
                    dist.broadcast(visible_points, src=0)

                    self.model.module.pc.prune(visible_points)
                    self.optimizer = torch.optim.Adam([
                        {'params': list(self.model.parameters())},
                    ], lr=self.lr)
            else:
                if epoch != self.start_epoch and self.model.raster_settings.radius >= 0.006:
                    # if (visible_percentage >= 0.8): # and (self.model.pc.points.shape[0] != self.model.pc.max_points):
                    self.model.pc.prune(self.model.visible_points)
                    self.optimizer = torch.optim.Adam([
                        {'params': list(self.model.parameters())},
                    ], lr=self.lr)

            # Upsampling
            if epoch % self.upsample_freq == 0:
                if self.distributed:
                    if epoch != 0 and self.model.module.pc.points.shape[0] < self.model.module.pc.max_points:
                        self.upsample_points()
                        batch_size = min(int(self.conf.get_int('train.max_points_training') / self.model.module.pc.points.shape[0]), self.max_batch)
                        if batch_size != self.batch_size:
                            self.batch_size = batch_size
                            self._init_dataloader()

                    # elif self.model.module.pc.points.shape[0] == self.model.module.pc.max_points:
                    #     # NOTE max_points에 도달하면 radius만 줄인다.
                    #     current_radius = self.model.module.raster_settings.radius
                    #     if self.model.module.pc.min_radius < 0.75 * current_radius:
                    #         new_radius = 0.75 * current_radius
                    #         # self.model.module.pc.register_radius(new_radius)
                    #         self.model.module.raster_settings.radius = new_radius
                else:
                    if epoch != 0 and self.model.pc.points.shape[0] < self.model.pc.max_points:
                        self.upsample_points()
                        batch_size = min(int(self.conf.get_int('train.max_points_training') / self.model.pc.points.shape[0]), self.max_batch)
                        if batch_size != self.batch_size:
                            self.batch_size = batch_size
                            self._init_dataloader()

                    elif self.model.pc.points.shape[0] == self.model.pc.max_points:
                        # NOTE max_points에 도달하면 radius만 줄인다.
                        current_radius = self.model.raster_settings.radius
                        if self.model.pc.min_radius < 0.75 * current_radius:
                            new_radius = 0.75 * current_radius
                            # self.model.pc.register_radius(new_radius)
                            self.model.raster_settings.radius = new_radius

            # re-init visible point tensor each epoch
            if self.distributed:
                self.model.module.visible_points = torch.zeros(self.model.module.pc.points.shape[0]).bool().cuda()
            else:
                self.model.visible_points = torch.zeros(self.model.pc.points.shape[0]).bool().cuda()

            upsample_freq_iter = len(self.train_dataloader) // 4 # len(self.dataset_train_subdir) * 100 # NOTE 100번 같은 dataset에서 step이 이루어진 경우 train epoch내에서 upsample.
            for data_index, (indices, model_input, ground_truth) in tqdm(enumerate(self.train_dataloader), desc='[INFO] training...', total=len(self.train_dataloader)):
                current_step = epoch * len(self.train_dataset) + data_index * self.batch_size
                if current_step % upsample_freq_iter == 0 and not self.target_training and not self.multi_source_training:
                    self.eval_mode()
                    self.validation(epoch, step=current_step)
                    self.train_mode()
                
                #     if self.distributed:
                #         if current_step != 0 and self.model.module.pc.points.shape[0] < self.model.module.pc.max_points:
                #             self.upsample_points()

                #         elif self.model.module.pc.points.shape[0] == self.model.module.pc.max_points:
                #             current_radius = self.model.module.raster_settings.radius
                #             if self.model.module.pc.min_radius < 0.75 * current_radius:
                #                 new_radius = 0.75 * current_radius
                #                 self.model.module.raster_settings.radius = new_radius
                #     else:
                #         if current_step != 0 and self.model.pc.points.shape[0] < self.model.pc.max_points:
                #             self.upsample_points()

                #         elif self.model.pc.points.shape[0] == self.model.pc.max_points:
                #             current_radius = self.model.raster_settings.radius
                #             if self.model.pc.min_radius < 0.75 * current_radius:
                #                 new_radius = 0.75 * current_radius
                #                 self.model.raster_settings.radius = new_radius

                #     if self.distributed:
                #         self.model.module.visible_points = torch.zeros(self.model.module.pc.points.shape[0]).bool().cuda()
                #     else:
                #         self.model.visible_points = torch.zeros(self.model.pc.points.shape[0]).bool().cuda()

                self.start_time_step.record()
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
                    
                    # hyunsoo added
                    if self.optimize_latent_code:
                        model_input['latent_code'] = self.latent_codes(model_input["idx"]).squeeze(1)
                    if self.optimize_scene_latent_code:
                        if self.target_training:
                            learnable_tensor_size = self.category_latent_dim + self.scene_latent_dim

                            input_latent_codes = torch.zeros(len(model_input['sub_dir']), learnable_tensor_size*len(self.category_dict))

                            # NOTE zero scene latent로 일단 다 초기화한다.
                            for i, v in self.category_dict.items():
                                category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                                start_idx = v*learnable_tensor_size
                                end_idx = (v+1)*learnable_tensor_size
                                input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)

                            # NOTE target human의 latent code만 넣어준다.
                            for i, v in enumerate(self.target_category_dict.values()):
                                category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                                target_start_idx = v*learnable_tensor_size
                                target_end_idx = (v+1)*learnable_tensor_size
                                input_latent_codes[:, target_start_idx:target_end_idx] = torch.cat((category_latent_code, self.target_scene_latent_codes(torch.tensor(i).cuda())), dim=0)

                            # Add to the model_input dictionary
                            model_input['scene_latent_code'] = input_latent_codes.cuda()
                            
                        elif self.multi_source_training:
                            learnable_tensor_size = self.category_latent_dim + self.scene_latent_dim

                            input_latent_codes = torch.zeros(len(model_input['sub_dir']), learnable_tensor_size*len(self.category_dict))

                            if model_input['sub_dir'][0] == self.source_datasets[0]:
                                # indices = [self.dataset_train_subdir.index(name) for name in model_input['sub_dir']]

                                # indices_tensor = torch.tensor(indices, dtype=torch.long, device=model_input['idx'].device)
                                # scene_latent_codes_tensor = self.scene_latent_codes(indices_tensor)         # [B, 28] 데이터셋 폴더마다 한개씩 존재
                                
                                # index_list = []
                                # for sub_dir_item in model_input['sub_dir']:
                                #     category_idx = self.category_dict[sub_dir_item.split('_')[0]]
                                #     index_list.append(category_idx)
                                # category_latent_codes = F.one_hot(torch.tensor(index_list), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(index_list).cuda())
                                # scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

                                # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. [B, 320]
                                # input_latent_codes = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))
                                
                                # NOTE zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                                for i, v in self.category_dict.items():
                                    category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(v).cuda()).detach()
                                    start_idx = v* learnable_tensor_size
                                    end_idx = (v+1)* learnable_tensor_size
                                    input_latent_codes[:, start_idx:end_idx] = torch.cat((category_latent_code.cuda(), self.zero_latent_codes(torch.tensor(0).cuda())), dim=0)
                                    
                                # NOTE source에 관한 latent code들을 넣어준다. 모든 category에 대해서.
                                for i, v in enumerate(self.source_category_dict.values()):
                                    category_latent_code = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).cuda()
                                    source_start_idx = v* learnable_tensor_size
                                    source_end_idx = (v+1)* learnable_tensor_size
                                    input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_latent_code, self.source_scene_latent_codes(torch.tensor(i).cuda())), dim=0)
                                
                                # Add to the model_input dictionary
                                model_input['scene_latent_code'] = input_latent_codes.cuda()


                            elif model_input['sub_dir'][0] in self.hat_dataset_train_subdir:            # NOTE hat가 있는 dataset.
                                # indices_tensor = torch.tensor(indices, dtype=torch.long, device=model_input['idx'].device)
                                # scene_latent_codes_tensor = self.scene_latent_codes(indices_tensor)         # [B, 28] 데이터셋 폴더마다 한개씩 존재
                                
                                # index_list = []
                                # for sub_dir_item in model_input['sub_dir']:
                                #     category_idx = self.category_dict[sub_dir_item.split('_')[0]]
                                #     index_list.append(category_idx)
                                # category_latent_codes = F.one_hot(torch.tensor(index_list), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(index_list).cuda())
                                # scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

                                indices = [self.hat_dataset_train_subdir.index(name) for name in model_input['sub_dir']]
                                multi_source_with_hat_scene_latent_codes = self.multi_source_with_hat_scene_latent_codes[indices[0]]

                                # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. [B, 320]
                                # input_latent_codes = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))
                                
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
                                
                                # NOTE 마지막으로 db human의 latent code를 넣어준다.
                                # for i in range(len(index_list)):
                                #     start_idx = index_list[i]*scene_latent_codes_tensor.shape[1]
                                #     end_idx = (index_list[i]+1)*scene_latent_codes_tensor.shape[1]
                                #     input_latent_codes[i, start_idx:end_idx] = scene_latent_codes_tensor[i]

                                # Add to the model_input dictionary
                                model_input['scene_latent_code'] = input_latent_codes.cuda()
                            else:       # NOTE 모자가 없는 dataset.
                                # indices = [self.not_include_hat_datasets.index(name) for name in model_input['sub_dir']]

                                # indices_tensor = torch.tensor(indices, dtype=torch.long, device=model_input['idx'].device)
                                # scene_latent_codes_tensor = self.scene_latent_codes(indices_tensor)         # [B, 28] 데이터셋 폴더마다 한개씩 존재
                                
                                # index_list = []
                                # for sub_dir_item in model_input['sub_dir']:
                                #     category_idx = self.category_dict[sub_dir_item.split('_')[0]]
                                #     index_list.append(category_idx)
                                # category_latent_codes = F.one_hot(torch.tensor(index_list), num_classes=len(self.category_dict)).cuda() # self.category_latent_codes(torch.tensor(index_list).cuda())
                                # scene_latent_codes_tensor = torch.cat((category_latent_codes, scene_latent_codes_tensor), dim=1)

                                # NOTE 해당하는 latent만 놓고 나머지는 0으로 놓는다. [B, 320]
                                # input_latent_codes = torch.zeros(scene_latent_codes_tensor.shape[0], scene_latent_codes_tensor.shape[1]*len(self.category_dict))
                                
                                indices = [self.no_hat_dataset_train_subdir.index(name) for name in model_input['sub_dir']]
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
                                
                                # NOTE 마지막으로 db human의 latent code를 넣어준다.
                                # for i in range(len(index_list)):
                                #     start_idx = index_list[i]*scene_latent_codes_tensor.shape[1]
                                #     end_idx = (index_list[i]+1)*scene_latent_codes_tensor.shape[1]
                                #     input_latent_codes[i, start_idx:end_idx] = scene_latent_codes_tensor[i]

                                # Add to the model_input dictionary
                                model_input['scene_latent_code'] = input_latent_codes.cuda()

                        else:       # NOTE source human + DB
                            indices = [self.dataset_train_subdir.index(name) for name in model_input['sub_dir']]

                            indices_tensor = torch.tensor(indices, dtype=torch.long, device=model_input['idx'].device)
                            scene_latent_codes_tensor = self.scene_latent_codes(indices_tensor)         # [B, 28] 데이터셋 폴더마다 한개씩 존재
                            
                            index_list = []
                            for sub_dir_item in model_input['sub_dir']:
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

                            # Add to the model_input dictionary
                            model_input['scene_latent_code'] = input_latent_codes.cuda()
                            

                model_outputs = self.model(model_input)

                # if True:
                #     # NOTE rendering camera four views.

                loss_output = self.loss(model_outputs, ground_truth, model_input)

                loss = loss_output['loss']

                self.optimizer.zero_grad()
                if self.optimize_inputs and epoch > 10 and not self.target_training_latent_freeze:
                    self.optimizer_cam.zero_grad()

                # if loss.item() > 2:
                #     tqdm.write("Loss is too high")
                #     tqdm.write("Loss: {}".format(loss.item()))
                #     tqdm.write("subdir: {}, img_name: {}".format(model_input['sub_dir'][0], model_input['img_name'].item()))

                if self.accelerate:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                self.optimizer.step()
                if self.optimize_inputs and epoch > 10 and not self.target_training_latent_freeze:
                    self.optimizer_cam.step()

                for k, v in loss_output.items():
                    loss_output[k] = v.detach().item()
                    if k not in acc_loss:
                        acc_loss[k] = [v]
                    else:
                        acc_loss[k].append(v)

                log_freq = self.log_freq
                if self.distributed:
                    acc_loss['visible_percentage'] = (torch.sum(self.model.module.visible_points)/self.model.module.pc.points.shape[0]).unsqueeze(0)
                    if data_index % log_freq == 0:
                        for k, v in acc_loss.items():
                            acc_loss[k] = sum(v) / len(v)
                        print_str = '{0} [{1}] ({2}/{3}): '.format(self.methodname, epoch, data_index, self.n_batches)
                        for k, v in acc_loss.items():
                            print_str += '{}: {:.3g} '.format(k, v)
                        print_str += 'num_points: {} radius: {}'.format(self.model.module.pc.points.shape[0], self.model.module.raster_settings.radius)
                        print(print_str)
                        acc_loss['num_points'] = self.model.module.pc.points.shape[0]
                        acc_loss['radius'] = self.model.module.raster_settings.radius

                        acc_loss['lr'] = self.scheduler.get_last_lr()[0]
                        wandb.log(acc_loss, step=epoch * len(self.train_dataset) + data_index * self.batch_size)
                        acc_loss = {}
                        self.end_time_step.record()
                        torch.cuda.synchronize()
                        wandb.log({"timing_step": self.start_time_step.elapsed_time(self.end_time_step)}, step=epoch * len(self.train_dataset) + data_index * self.batch_size)
                        # tqdm.write("[INFO] Iteration time: {} s".format(self.start_time_step.elapsed_time(self.end_time_step)/1000))
                else:
                    acc_loss['visible_percentage'] = (torch.sum(self.model.visible_points)/self.model.pc.points.shape[0]).unsqueeze(0)
                    if data_index % log_freq == 0:
                        for k, v in acc_loss.items():
                            acc_loss[k] = sum(v) / len(v)
                        print_str = '{0} [{1}] ({2}/{3}): '.format(self.methodname, epoch, data_index, self.n_batches)
                        for k, v in acc_loss.items():
                            print_str += '{}: {:.3g} '.format(k, v)
                        print_str += 'num_points: {} radius: {}'.format(self.model.pc.points.shape[0], self.model.raster_settings.radius)
                        # print(print_str)
                        tqdm.write(print_str)
                        acc_loss['num_points'] = self.model.pc.points.shape[0]
                        acc_loss['radius'] = self.model.raster_settings.radius

                        acc_loss['lr'] = self.scheduler.get_last_lr()[0]
                        wandb.log(acc_loss, step=epoch * len(self.train_dataset) + data_index * self.batch_size)
                        acc_loss = {}
                        self.end_time_step.record()
                        torch.cuda.synchronize()
                        wandb.log({"timing_step": self.start_time_step.elapsed_time(self.end_time_step)}, step=epoch * len(self.train_dataset) + data_index * self.batch_size)
                        # tqdm.write("Iteration time: {} s".format(self.start_time_step.elapsed_time(self.end_time_step)/1000))

                if data_index % self.save_freq == 0:
                    self.save_checkpoints(epoch)
                
                
            self.scheduler.step()
            self.end_time.record()
            torch.cuda.synchronize()
            wandb.log({"timing_epoch": self.start_time.elapsed_time(self.end_time)}, step=(epoch+1) * len(self.train_dataset))
            print("Epoch time: {} s".format(self.start_time.elapsed_time(self.end_time)/1000))
        self.save_checkpoints(self.nepochs + 1)




