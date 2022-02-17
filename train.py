import CAModel as cm
from pytorch_lightning import Trainer
import Config
import VoxelDataModule as vdm
from utils import readPLY, getCentroid
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from MyCallbacks import GrowImprovementPerEpochLogger, PoolPatternSampleCallback
import wandb
import torch
import numpy as np

if __name__ == '__main__':

    cfgModel = Config.ModelConfig()
    cfgData = Config.DataModuleConfig()

    listCoords = []
    target = readPLY(cfgData.filePath)
    coords = np.where(target[:,:,:,3:4]==1)
    for i in range(0,len(coords[0])):
        listCoords += [np.array([coords[0][i:i+1][0], coords[1][i:i+1][0], coords[2][i:i+1][0]])]

    centerCoords = getCentroid(listCoords, np.array(target.shape[:3])//2)
    seed = np.zeros(list(target.shape)[:-1]+[16], np.float32)
    seed[centerCoords[0],centerCoords[1],centerCoords[2], 3:] = 1.0

    seed = torch.from_numpy(seed).to('cuda')
    target = torch.from_numpy(target).to('cuda')

    min_steps = (max(target.shape)*2)+10
    ca_model = cm.CAModel(cfgModel, min_step=min_steps, max_step=min_steps+40)#min_step=120, max_step=160)# BEST: MIN=70, MAX=105
    data_module = vdm.VoxelDataModule(cfgData)

    animal_name = cfgData.animalName
    name_run = f"{animal_name}-modified-growSeed1-persist1-cropDmg2-noiseRem1(lastCropDmg)-stepLR"

    wandb_logger = WandbLogger(name=name_run, project="lit-3D-Grow", entity="maxpappa")

    checkpoint_callback = ModelCheckpoint(
        monitor="loss_from_seed",
        dirpath=f"./checkpoints_{animal_name}/",
        filename=name_run+"_3DCellularGrow-{epoch:02d}-{loss_from_seed:.2f}",
        save_top_k=3,
        mode="min",
    )

    wandb_logger.watch(ca_model, log="all", log_freq=50)
    trainer = Trainer(devices=1, accelerator="gpu", log_every_n_steps=1, max_epochs=ca_model.hparams.n_epochs,
                    reload_dataloaders_every_n_epochs=1, default_root_dir=f"./checkpoints_{animal_name}/", 
                    callbacks=[checkpoint_callback, PoolPatternSampleCallback()],
                    precision=16, progress_bar_refresh_rate=2, logger=wandb_logger, num_sanity_val_steps=0
                    )
    trainer.fit(ca_model, datamodule=data_module)
    trainer.save_checkpoint(name_run+".pth")
    wandb.save(name_run+".pth")
    wandb.save(f"./checkpoints_{animal_name}/*")