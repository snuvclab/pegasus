from typing import Any, Dict
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, BaseFinetuning
import torch.optim as optim


class PointAvatarTrainCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):  # NOTE callback -> on_train_epoch_start
        if pl_module.start_epoch != trainer.current_epoch: # and trainer.current_epoch % pl_module.upsample_freq == 1:
            new_optimizers = optim.Adam([{'params': list(pl_module.model.parameters()) + list(pl_module.input_params)}], lr=pl_module.lr)
            trainer.optimizers = [new_optimizers]
            print('*******************************')
            print('[INFO] Reinitialize Optimizer')
            print('*******************************')
    
    # def on_train_epoch_end(self, trainer, pl_module): # NOTE callback -> on_train_epoch_end
    #     if trainer.current_epoch % pl_module.upsample_freq == 0:
    #         new_optimizers = optim.Adam([{'params': list(pl_module.model.parameters()) + list(pl_module.input_params)}], lr=pl_module.lr)
    #         trainer.optimizers = [new_optimizers]