from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    n_epochs: int = 400  # number of epochs of training
    # adam parameters
    lr: float = 3e-3  # adam: learning rate
    lr_gamma: float = 0.9999
    betas: Tuple[float] = (0.5,0.5)
    # CAModel parameters
    fire_rate: float = 0.5
    n_channels: int = 16  # number of image channels
    hidden_size: int = 128

@dataclass
class DataModuleConfig:
    filePath: str = "./plyFiles/large puffin.ply"
    channels: int = 16
    pool_size: int = 128
    batch_size: int = 4
    num_workers: int = 0
    damage: bool = True
    modifiedTrainingMode: bool = True # True for the classic version of training
    randomNoise: bool = True