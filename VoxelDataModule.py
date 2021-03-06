import pytorch_lightning as pl
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from typing import Union, Dict
from Config import DataModuleConfig
from dataclasses import asdict
from utils import readPLY, getCentroid, take_cube
import numpy as np
import torch
from random import randint, sample


class VoxelDataModule(pl.LightningDataModule):
    def __init__(self, hparams: Union[Dict, DataModuleConfig], train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)
        self.save_hyperparameters(asdict(hparams) if not isinstance(hparams, Dict) else hparams)

    def prepare_data(self, *args, **kwargs):
        self.target = readPLY(self.hparams["filePath"])
        self.batch_idx = list(range(0,self.hparams["batch_size"]))
        listCoords = []

        coords = np.where(self.target[:,:,:,3:4]==1)
        for i in range(0,len(coords[0])):
            listCoords += [np.array([coords[0][i:i+1][0], coords[1][i:i+1][0], coords[2][i:i+1][0]])]

        centerCoords = getCentroid(listCoords, np.array(self.target.shape[:3])//2)
        self.listCoords = listCoords
        self.seed = np.zeros(list(self.target.shape)[:-1]+[self.hparams["channels"]], np.float32)
        self.seed[centerCoords[0],centerCoords[1],centerCoords[2], 3:] = 1.0

        self.pool = torch.from_numpy(np.repeat(self.seed[None, ...], self.hparams["pool_size"], 0))
        self.pool_target = torch.from_numpy(np.repeat(self.target[None, ...], self.hparams["pool_size"], 0))
        self.seed = torch.from_numpy(self.seed)
        self.target = torch.from_numpy(self.target)

    def setup(self, stage):
        self.train_data = TensorDataset(self.pool, self.pool_target)

    def train_dataloader(self):
        # reload at each epoch setting Trainer flag "reload_dataloaders_every_n_epocs=1"
        self.train_data = TensorDataset(self.pool, self.pool_target)
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


    def make_cube_damage(self, batch: torch.Tensor) -> torch.Tensor:
        ''' make cube-shaped damage to the input batch

        Args:
            batch (torch.Tensor): input batch to damage

        Returns:
            torch.Tensor: damaged input batch
        '''        
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

    def perturbate(self, inp: torch.Tensor, perc: float) -> torch.Tensor:
        ''' for each example in the input batch (inp), replace in a different manner
            a percentage (perc) of cells with some random values. 

        Args:
            inp (torch.Tensor): input batch
            perc (float): percentage of cells to replace

        Returns:
            torch.Tensor: input batch where some voxels are replaced with noisy values.
        '''        
        for i in range(0, inp.shape[0]):
            indices = (inp[i,:,:,:,3:4]>0.1)[:,:,:,0].squeeze().nonzero(as_tuple=False)
            mask = torch.rand(indices.shape[0]) <= perc
            indices = indices[mask,:]
            noise = torch.rand(list(inp.shape[1:4])+[16], dtype=inp.dtype)
            inp[i, indices[:,0], indices[:,1], indices[:,2],:] = noise[indices[:,0],indices[:,1],indices[:,2],:]
        return inp

    def on_before_batch_transfer(self, batch: torch.tensor, dataloader_idx: int) -> torch.tensor:
        # calculate loss between initial batch and target, sort their indexes (in the batch) in ascending order
        # using calculated loss, then take the last one (which has the lower loss in the batch)
        if self.trainer.validating:
            return batch
        with torch.no_grad():
            idx_worst = torch.mean(torch.pow(batch[0][..., :4] - self.target, 2), [-1,-2,-3,-4]).argsort(descending=True)[:3]
            batch[0][idx_worst[:1]] = self.seed.clone()
            if self.hparams.modifiedTrainingMode:
                if self.hparams.randomNoise:
                    batch[0][idx_worst[-1]] = self.perturbate(batch[0][idx_worst[-1]][None,...], perc=0.05).squeeze()
                batch[0][idx_worst[1:]] = take_cube(batch[0][idx_worst[1:]])
            else:
                batch[0][idx_worst[1:]] = self.make_cube_damage(batch[0][idx_worst[1:]])
        return batch