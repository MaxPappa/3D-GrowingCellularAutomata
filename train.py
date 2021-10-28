import CAModel
from pytorch_lightning import Trainer
import Config
import VoxelDataModule as vdm
import numpy as np
from utils import readPLY, getCentroid
from pytorch_lightning.loggers import WandbLogger
from MyCallbacks import GrowImprovementPerEpochLogger
import torch

if __name__ == '__main__':

    listCoords = []
    target = readPLY('./plyFiles/x23_kangaroo.ply', pad=16)
    coords = np.where(target[:,:,:,3:4]==1)
    for i in range(0,len(coords[0])):
        listCoords += [(coords[0][i:i+1][0], coords[1][i:i+1][0], coords[2][i:i+1][0])]

    centerCoords = getCentroid(listCoords)
    seed = np.zeros(list(target.shape)[:-1]+[16], np.float32)
    seed[centerCoords[0],centerCoords[1],centerCoords[2], 3:] = 1.

    cfgModel = Config.ModelConfig()
    cfgData = Config.DataModuleConfig()
    seed = torch.from_numpy(seed).to(cfgModel.device)
    target = torch.from_numpy(target).to(cfgModel.device)

    ca_model = CAModel.CAModel(cfgModel)
    data_module = vdm.VoxelDataModule(cfgData)

    wandb_logger = WandbLogger(project="lit-3D-Grow")

    trainer = Trainer(devices=1, accelerator="gpu", log_every_n_steps=1, reload_dataloaders_every_n_epochs=1,
            default_root_dir="./checkpoints/", logger=wandb_logger,
            callbacks=[GrowImprovementPerEpochLogger(seed=seed, target=target)],
            accumulate_grad_batches=2
            )
    trainer.fit(ca_model, datamodule=data_module)