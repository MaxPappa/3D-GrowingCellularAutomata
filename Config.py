from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    n_epochs: int = 8000  # number of epochs of training
    batch_size: int = 4  # size of the batches
    
    # adam parameters
    lr: float = 2e-3  # adam: learning rate
    lr_gamma = 0.9999
    betas: Tuple[float] = (0.5,0.5)

    # CAModel parameters
    filePath: str = "kangaroo.ply"  # name of file from which start learning
    fire_rate: float = 0.5
    pool_size: int = 128
    n_channels: int = 16  # number of image channels
    hidden_size: int = 128

    # dataloader parameters
    n_cpu: int = 8  # number of cpu threads to use for the dataloaders