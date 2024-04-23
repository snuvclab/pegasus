import sys
sys.path.append('../code')
sys.path.append('./')
import argparse

from pyhocon import ConfigFactory
import torch
import random, os
import numpy as np
import utils.general as utils


def seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    parser.add_argument('--is_eval', default=False, action="store_true", help='If set, only render images')
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--wandb_workspace', type=str)
    parser.add_argument('--wandb_tags', type=str, nargs="+", default=[])
    parser.add_argument('--only_json', default=False, action="store_true", help='If set, do not load images during testing. ')
    parser.add_argument('--checkpoint', default='latest', type=str, help='The checkpoint epoch number in case of continuing from a previous run.')
    parser.add_argument('--local_rank', type=int, default=0)
    opt = parser.parse_args()

    conf = ConfigFactory.parse_file(opt.conf)

    torch.backends.cuda.matmul.allow_tf32 = conf.get_bool('train.tf32')
    torch.backends.cudnn.allow_tf32 = conf.get_bool('train.tf32')

    seed(42)

    if not opt.is_eval:
        runner = utils.get_class(conf.get_string('train.train_runner'))(opt=opt,
                                                                        conf=opt.conf,
                                                                        nepochs=opt.nepoch,
                                                                        checkpoint=opt.checkpoint,
                                                                        wandb_tags=opt.wandb_tags,
                                                                        path_ckpt=None,
                                                                        )
        runner.run()
    else:
        runner = utils.get_class(conf.get_string('test.test_runner'))(conf=opt.conf,
                                                                        checkpoint=opt.checkpoint,
                                                                        only_json=opt.only_json,
                                                                        )

        runner.run()
