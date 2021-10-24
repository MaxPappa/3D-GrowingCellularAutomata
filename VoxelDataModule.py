import pytorch_lightning as pl
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from voxelDataset import voxelDataset
from typing import Union, Dict
from Config import DataModuleConfig
from dataclasses import asdict
from utils import readPLY, getCentroid
import numpy as np
import torch
#import pdb


class VoxelDataModule(pl.LightningDataModule):
    def __init__(self, hparams: Union[Dict, DataModuleConfig], train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)
        self.save_hyperparameters(asdict(hparams) if not isinstance(hparams, Dict) else hparams)

    def prepare_data(self, *args, **kwargs):
        self.target = readPLY(self.hparams["filePath"])
        listCoords = []
        coords = np.where(self.target[:,:,:,3:4]==1)
        for i in range(0,len(coords[0])):
            listCoords += [(coords[0][i:i+1][0], coords[1][i:i+1][0], coords[2][i:i+1][0])]

        centerCoords = getCentroid(listCoords)
        self.seed = np.zeros(list(self.target.shape)[:-1]+[self.hparams["channels"]], np.float32)
        self.seed[centerCoords[0],centerCoords[1],centerCoords[2], 3:] = 1.0
        self.pool = torch.from_numpy(np.repeat(self.seed[None, ...], self.hparams["pool_size"], 0))
        self.pool_target = torch.from_numpy(np.repeat(self.target[None, ...], self.hparams["pool_size"], 0))

    def setup(self, stage):
        self.train_data = TensorDataset(self.pool, self.pool_target)# torch.from_numpy(self.target))

    def train_dataloader(self):
        self.train_data = TensorDataset(self.pool, self.pool_target) # reload at each epoch setting Trainer flag "reload_dataloaders_every_n_epocs=1"
        return [DataLoader(self.train_data, batch_size=self.hparams.batch_size)]#self.hparams["batch_size"])


    def on_before_batch_transfer(self, batch: torch.tensor, dataloader_idx: int) -> torch.tensor:
        # calculate loss between initial batch and target, sort their indexes (in the batch) in ascending order
        # using calculated loss, then take the last one (which has the higher loss in the batch)
        idx_worst = torch.mean(torch.pow(batch[0][0][..., :4] - self.target, 2), [-1,-2,-3,-4]).argsort()[-1]
        batch[0][0][idx_worst] = torch.from_numpy(self.seed)
        return batch