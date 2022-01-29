import pytorch_lightning as pl
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from typing import Union, Dict
from Config import DataModuleConfig
from dataclasses import asdict
from utils import readPLY, getCentroid
import numpy as np
import torch
from random import randint, sample


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
        self.listCoords = listCoords

        self.seed = np.zeros(list(self.target.shape)[:-1]+[self.hparams["channels"]], np.float32)
        self.seed[centerCoords[0],centerCoords[1],centerCoords[2], 3:] = 1.0
        self.pool = torch.from_numpy(np.repeat(self.seed[None, ...], self.hparams["pool_size"], 0))
        self.pool_target = torch.from_numpy(np.repeat(self.target[None, ...], self.hparams["pool_size"], 0))
        self.seed = torch.from_numpy(self.seed)
        self.target = torch.from_numpy(self.target)

    def setup(self, stage):
        self.train_data = TensorDataset(self.pool, self.pool_target)# torch.from_numpy(self.target))

    def train_dataloader(self):
        self.train_data = TensorDataset(self.pool, self.pool_target)# reload at each epoch setting Trainer flag "reload_dataloaders_every_n_epocs=1"
        dataloader =  DataLoader(self.train_data, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        class simpleDataset(torch.utils.data.Dataset):
            def __init__(self, seed, target):
                self.x = seed
                self.y = target
            
            def __getitem__(self, index):
                return self.x, self.y
            
            def __len__(self):
                return 1
        simple = simpleDataset(self.seed, self.target)
        return DataLoader(simple, batch_size=1, num_workers=self.hparams.num_workers)


    def make_cube_damage(self, batch):
        for anim in batch[:1]:
            lungh = len(torch.where(anim[:,:,:,3:4]>0.1)[0])
            if lungh <= 10:
                continue
            num = sample(range(0,lungh-1), k=1)[0]
            coords = torch.where(anim[:,:,:,3:4]>0.1)
            x,y,z = coords[0][num],coords[1][num],coords[2][num]
            r = randint(4,8)
            anim[max(0,x-r):x+r, max(0,y-r):y+r, max(0,z-r):z+r,:] = 0
        return batch

    def take_cube(self, inp):
        for i in range(0,inp.shape[0]):
            r = randint(5,8)
            x,y,z = torch.where(inp[i,:,:,:,3:4] > 0.1)[1:4]
            if len(x) == 0:
                continue
            index = randint(0,len(x)-1)
            x,y,z = x[index],y[index],z[index]
            block = inp[i, max(0,x - r) : x + r, max(0,y - r) : y + r, max(0, z - r) : z + r, :].clone()

            # this is to avoid collapse of patterns in pool (it can happen that sometimes it can't reconstruct nothing and in the pool we
            # will have something that collapse with zero mature/growing cells, so with an empty/ghost object)
            if block[:,:,:,3:4].sum() <= 10: 
                 continue
            
            inp[i,:, :, :, :] = 0
            inp[i, max(0,x - r) : x + r, max(0,y - r) : y + r, max(0, z - r) : z + r, :] = block.clone()
        return inp


    def on_before_batch_transfer(self, batch: torch.tensor, dataloader_idx: int) -> torch.tensor:
        # calculate loss between initial batch and target, sort their indexes (in the batch) in descending order
        # using calculated loss, then take the first one (which has the highest loss in the batch)
        if self.trainer.validating:
            return batch
        with torch.no_grad():
            idx_worst = torch.mean(torch.pow(batch[0][0][..., :4] - self.target, 2), [-1,-2,-3,-4]).argsort(descending=True)[:1]
            batch[0][idx_worst[0]] = self.seed.clone()
            if self.hparams.damage and self.trainer.global_step > 100:
                batch[0][1:] = self.take_cube(batch[0][1:])
        return batch