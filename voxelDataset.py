from torch.utils.data import Dataset
from utils import readPLY, getCentroid
from typing import Dict
import numpy as np
import torch
import random

class voxelDataset(Dataset):
    def __init__(self, filePath: str, n_channels: int = 16, pool_size: int = 128, batch_size: int = 4) -> None:
        
        self.target = readPLY(filePath)
        self.batch_size = batch_size
        self.batch_idx = list(range(0,batch_size))
        listCoords = []
        coords = np.where(self.target[:,:,:,3:4]==1)
        for i in range(0,len(coords[0])):
            listCoords += [(coords[0][i:i+1][0], coords[1][i:i+1][0], coords[2][i:i+1][0])]

        centerCoords = getCentroid(listCoords)
        self.seed = np.zeros(list(self.target.shape)[:-1]+[n_channels], np.float32)
        self.seed[centerCoords[0],centerCoords[1],centerCoords[2], 3:] = 1.0
        self.pool = torch.from_numpy(np.repeat(self.seed[None, ...], pool_size, 0))
        #self.batch = torch.from_numpy(np.repeat(self.seed[None, ...], batch_size, 0))

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        return {'input':self.pool[index], 'target':self.target}


    # def swap(self):
    #     tmp_idx = random.choices(self.pool, k=self.batch_size)
    #     tmp_batch = self.batch.clone()
    #     self.batch = torch.tensor([self.pool[i] for i in tmp_idx])
    #     for i in range(0,self.batch_size):
    #         self.pool[tmp_idx[i]] = tmp_batch[i]


        # for i in range(0, n_epoch+1):
        #     if USE_PATTERN_POOL:
        #         batch = pool.sample(BATCH_SIZE)
        #         x0 = torch.from_numpy(batch.x.astype(np.float32)).to(device)
        #         loss_rank = loss_f(x0, pad_target).detach().cpu().numpy().argsort()[::-1]
        #         x0 = batch.x[loss_rank]
        #         x0[:1] = seed
        #     else:
        #         x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)    
        #     x0 = torch.from_numpy(x0.astype(np.float32)).to(device)
        # pass