import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import pytorch_lightning as pl
from visualizationUtils import visualizeGO, visualizeImprovements
from typing import Union, Dict, List
from Config import ModelConfig
from dataclasses import asdict
from MyCallbacks import PoolSamplerCallback
from random import randint, sample

class CAModel(pl.LightningModule):
    def __init__(self, hparams: Union[Dict, ModelConfig], min_step:int = 50, max_step: int = 95):
        super(CAModel, self).__init__()

        self.save_hyperparameters(asdict(hparams) if not isinstance(hparams, Dict) else hparams)
        self.min_step = min_step
        self.max_step = max_step

        self.fc0 = weight_norm(nn.Linear(self.hparams["n_channels"]*4, self.hparams["hidden_size"]))
        self.fc1 = nn.Linear(self.hparams["hidden_size"], self.hparams["n_channels"], bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()
        self.automatic_optimization = True

        kernel = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                      [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
                                      [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]) / 26
        self.register_buffer("kernel_x", kernel[None,None,...].repeat(self.hparams.n_channels,1,1,1,1))

        self.register_buffer("kernel_y", kernel.transpose(0,1)[None,None,...].repeat(self.hparams.n_channels,1,1,1,1))

        self.register_buffer("kernel_z", kernel.transpose(1,2)[None,None,...].repeat(self.hparams.n_channels,1,1,1,1))


    def alive(self, x):
        return F.max_pool3d(x[:, 3:4, :, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, kernel):
            return F.conv3d(x, kernel, padding=1, groups=self.hparams.n_channels)

        y1 = _perceive_with(x,self.kernel_x)
        y2 = _perceive_with(x,self.kernel_y)
        y3 = _perceive_with(x,self.kernel_z)

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
        return x.detach()

    def training_step(self, train_batch:torch.tensor, batch_idx:int) -> float:
        x, target = train_batch
        num_step = randint(self.min_step, self.max_step)
        for step in range(0, num_step):
            x = self.update(x, self.hparams['fire_rate'])
        loss_growth = F.mse_loss(x[:, :, :, :, :4], target[:,:,:,:,:4])
        x_2 = self.take_cube(x.detach().clone())
        for step in range(0, num_step-20):
            x_2 = self.update(x_2, self.hparams['fire_rate'])
        loss_reconstructed = F.mse_loss(x_2[:,:,:,:,:4], target[:,:,:,:,:4])
        loss = (loss_growth + loss_reconstructed)/2
        self.log("train_lossgrowth", loss_growth)
        self.log("train_lossreconstructed", loss_reconstructed)
        self.log("train_loss", loss)
        return {"loss":loss, "out":x.detach()}

    def make_cube_damage(self, inp):
        lungh = len(torch.where(inp[:,:,:,:,3:4]>0.1)[0])
        if lungh <=1:
            return inp
        num = sample(range(0,lungh-1), k=1)[0]
        coords = torch.where(inp[:,:,:,:,3:4]>0.1)
        x,y,z = coords[1][num],coords[2][num],coords[3][num]
        r = randint(4,8)
        inp[:,max(0,x-r):x+r, max(0,y-r):y+r, max(0,z-r):z+r,:] = 0
        return inp

    def take_cube(self, inp):
        side = randint(5,8)
        x,y,z = torch.where(inp[:,:,:,:,3:4] > 0.1)[1:4]
        if len(x) == 0:
            return inp
        # pdb.set_trace()
        index = randint(0,len(x)-1)
        x,y,z = x[index],y[index],z[index]
        block = inp[:, max(0,x - side) : x + side, max(0,y - side) : y + side, max(0, z - side) : z + side, :].clone()

        # this is to avoid collapse of patterns in pool (it can happen that sometimes it can't reconstruct nothing and in the pool we
        # will have something that collapse with zero mature/growing cells, so with an empty/ghost object)
        if block[:,:,:,:,3:4].sum() <= 10: 
            return inp 

        inp[:, :, :, :, :] = 0
        inp[:, max(0,x - side) : x + side, max(0,y - side) : y + side, max(0, z - side) : z + side, :] = block.clone()
        return inp

    def validation_step(self, batch, batch_idx):
        x, target = batch
        x = self(x, steps=randint(self.min_step, self.max_step+1), fire_rate=self.hparams.fire_rate)
        #pdb.set_trace()
        damaged = self.take_cube(x.detach().clone()) #self.make_cube_damage(x.detach().clone())
        reconstructed = damaged.clone()
        reconstructed = self(reconstructed, steps=randint(self.min_step, self.max_step+1), fire_rate=self.hparams.fire_rate)
        loss = F.mse_loss(x[:,:,:,:,:4], target[:,:,:,:,:4])
        loss_reconstruction = F.mse_loss(reconstructed[:,:,:,:,:4], x[:,:,:,:,:4])
        self.log("loss_from_seed", loss)
        self.log("loss_reconstructed", loss_reconstruction)
        if self.trainer.current_epoch % 2 == 0:
            x = x.squeeze()
            damaged = damaged.squeeze()
            reconstructed = reconstructed.squeeze()
            self.trainer.logger.experiment.log({"DAMAGED - before and after" : visualizeImprovements(x, damaged)})
            self.trainer.logger.experiment.log({"DAMAGED - Growing improvements " : visualizeImprovements(reconstructed, damaged)})
        return {"loss_from_seed":loss, "out":x, "damaged":damaged, "reconstructed":reconstructed, "loss_reconstructed":loss_reconstruction}



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"], betas=self.hparams["betas"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams["lr_gamma"])
        return {"optimizer":optimizer, "lr_scheduler":scheduler}

    def configure_callbacks(self) -> List[pl.Callback]:
        callbacks = [PoolSamplerCallback()]
        return callbacks