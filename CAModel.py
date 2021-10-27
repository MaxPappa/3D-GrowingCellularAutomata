import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from visualizationUtils import visualizeGO
from typing import Union, Dict, List
from Config import ModelConfig
from dataclasses import asdict
from MyCallbacks import PoolSamplerCallback
from random import randint

class CAModel(pl.LightningModule):
    def __init__(self, hparams: Union[Dict, ModelConfig], min_step:int = 50, max_step: int = 95):
        super(CAModel, self).__init__()

        self.save_hyperparameters(asdict(hparams) if not isinstance(hparams, Dict) else hparams)
        self.min_step = min_step
        self.max_step = max_step

        self.fc0 = nn.Linear(self.hparams["n_channels"]*4, self.hparams["hidden_size"])
        self.fc1 = nn.Linear(self.hparams["hidden_size"], self.hparams["n_channels"], bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()
        self.automatic_optimization = True

    def alive(self, x):
        return F.max_pool3d(x[:, 3:4, :, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, kernel):
            kernel = kernel[None,None,...].repeat(self.hparams.n_channels, 1, 1, 1, 1)
            return F.conv3d(x, kernel, padding=1, groups=self.hparams.n_channels)

        kernel_x = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                             [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
                             [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], device=self.hparams.device) / 26
        kernel_y = kernel_x.transpose(0,1)
        kernel_z = kernel_x.transpose(1,2)

        y1 = _perceive_with(x,kernel_x)
        y2 = _perceive_with(x,kernel_y)
        y3 = _perceive_with(x,kernel_z)
        y = torch.cat((x,y1,y2,y3),1)
        return y

    def update(self, x:torch.tensor, fire_rate:float, angle:float = 0) -> torch.tensor:
        x = x.transpose(1,4)
        mask = x[:, 3:4, :, :, :]
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.hparams.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,4)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,4)

    def forward(self, x:torch.tensor, steps:int=1, fire_rate:float=0.0, angle:float=0.0, path:str=None) -> torch.tensor:
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
            if path:
                fig = visualizeGO(x[0], True)
                fig.write_image(f"{path}_{step}.png")
        return x

    def training_step(self, train_batch:torch.tensor, batch_idx:int) -> float:
        x, target = train_batch[0]
        for step in range(0, randint(self.min_step, self.max_step+1)):
            x = self.update(x, self.hparams['fire_rate'])
        loss = F.mse_loss(x[:, :, :, :, :4], target[:,:,:,:,:4])
        self.log("train_loss", loss)
        return {"loss":loss, "out":x.detach()}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"], betas=self.hparams["betas"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams["lr_gamma"])
        return {"optimizer":optimizer, "lr_scheduler":scheduler}

    def configure_callbacks(self) -> List[pl.Callback]:
        callbacks = [PoolSamplerCallback()]
        return callbacks