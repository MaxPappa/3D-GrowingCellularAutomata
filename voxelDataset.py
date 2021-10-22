from torch.utils.data import Dataset
from utils import readPLY, getCentroid
from typing import Dict
import numpy as np

class voxelDataset(Dataset):
    def __init__(self, filePath: str, n_channels: int = 16, pool_size: int = 128) -> None:
        
        self.target = readPLY(filePath)

        listCoords = []
        coords = np.where(self.target[:,:,:,3:4]==1)
        for i in range(0,len(coords[0])):
            listCoords += [(coords[0][i:i+1][0], coords[1][i:i+1][0], coords[2][i:i+1][0])]

        centerCoords = getCentroid(listCoords)
        self.seed = np.zeros(list(self.target.shape)[:-1]+[n_channels], np.float32)
        self.seed[centerCoords[0],centerCoords[1],centerCoords[2], 3:] = 1.0
        self.pool = np.repeat(self.seed[None, ...], pool_size, 0)

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        return {'input':self.pool[index], 'target':self.target}