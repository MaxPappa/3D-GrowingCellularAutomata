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
from utils import take_cube

class CAModel(pl.LightningModule):
    def __init__(self, hparams: Union[Dict, ModelConfig], min_step:int = 50, max_step: int = 95):
        super(CAModel, self).__init__()

        self.save_hyperparameters(asdict(hparams) if not isinstance(hparams, Dict) else hparams)
        self.min_step = min_step
        self.max_step = max_step
        self.fc0 = weight_norm(nn.Linear((self.hparams["n_channels"]*4), self.hparams["hidden_size"]))
        self.fc1 = nn.Linear(self.hparams["hidden_size"], self.hparams["n_channels"], bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()
        self.automatic_optimization = True

        kernel = torch.tensor([[[-1,  0,  1],
                [-2,  0,  2],
                [-1,  0,  1]],

                [[-2,  0, 2],
                [-4,  0,  4],
                [-2,  0,  2]],

                [[-1,  0,  1],
                [-2,  0,  2],
                [-1,  0,  1]]]) / 26
        
        self.register_buffer("kernel_x", kernel[None,None,...].repeat(self.hparams.n_channels,1,1,1,1))

        self.register_buffer("kernel_y", kernel.transpose(1,2)[None,None,...].repeat(self.hparams.n_channels,1,1,1,1))

        self.register_buffer("kernel_z", kernel.transpose(0,2)[None,None,...].repeat(self.hparams.n_channels,1,1,1,1))


    def alive(self, x):
        return F.max_pool3d(x[:, 3:4, :, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x):

        def _perceive_with(x, kernel):
            return F.conv3d(x, kernel, padding=1, groups=self.hparams.n_channels)

        y1 = _perceive_with(x,self.kernel_x)
        y2 = _perceive_with(x,self.kernel_y)
        y3 = _perceive_with(x,self.kernel_z)

        y = torch.cat((x,y1,y2,y3),1)
        return y

    def update(self, x:torch.tensor, fire_rate:float) -> torch.tensor:
        x = x.transpose(1,4)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x)
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

    def forward(self, x:torch.tensor, steps:int=1, fire_rate:float=0.0, path:str=None) -> torch.tensor:
        for step in range(steps):
            x = self.update(x, fire_rate)
            if path:
                fig = visualizeGO(x[0], True)
                fig.write_image(f"{path}_{step}.png")
        return x.detach()

    def training_step(self, train_batch:torch.tensor, batch_idx:int):
        x, target = train_batch
        num_step = randint(self.min_step, self.max_step)
        for step in range(0, num_step):
            x = self.update(x, self.hparams['fire_rate'])
        loss = F.mse_loss(x[:, :, :, :, :4], target[:,:,:,:,:4])
        self.log("train_loss", loss)
        return {"loss":loss, "out":x.detach()}

    def make_cube_damage(self, inp):
        lungh = len(torch.where(inp[:,:,:,:,3:4]>0.1)[0])
        if lungh <=10:
            return inp
        num = sample(range(0,lungh-1), k=1)[0]
        coords = torch.where(inp[:,:,:,:,3:4]>0.1)
        x,y,z = coords[1][num],coords[2][num],coords[3][num]
        r = randint(4,8)
        inp[:,max(0,x-r):x+r, max(0,y-r):y+r, max(0,z-r):z+r,:] = 0
        return inp

    def perturbate(self, inp, perc):
        indices = (inp[0,:,:,:,3:4]>0.1)[:,:,:,0].squeeze().nonzero(as_tuple=False)
        mask = torch.rand(indices.shape[0]) <= perc
        indices = indices[mask,:]
        noise = torch.rand(list(inp.shape[1:4])+[16], dtype=inp.dtype, device=self.device)
        inp[0, indices[:,0], indices[:,1], indices[:,2],:] = noise[indices[:,0],indices[:,1],indices[:,2],:]
        return inp

    def validation_step(self, batch, batch_idx):
        x, target = batch
        x = self(x, steps=self.max_step, fire_rate=self.hparams.fire_rate)
        damaged = take_cube(x.detach().clone())
        reconstructed = damaged.clone()
        reconstructed = self(reconstructed, steps=self.max_step, fire_rate=self.hparams.fire_rate)
        loss = F.mse_loss(x[:,:,:,:,:4], target[:,:,:,:,:4])
        loss_reconstruction = F.mse_loss(reconstructed[:,:,:,:,:4], target[:,:,:,:,:4])
        perturbated = self.perturbate(x.detach().clone(), 0.05)
        pertOut = perturbated.detach().clone()
        pertOut = self(pertOut, steps=self.max_step, fire_rate=self.hparams.fire_rate)
        lossPert = F.mse_loss(pertOut[:,:,:,:,:4], target[:,:,:,:,:4])
        self.log("loss_from_seed", loss)
        self.log("loss_reconstructed", loss_reconstruction)
        self.log("loss_pert", lossPert)
        if self.trainer.current_epoch == 0:
            self.trainer.logger.experiment.log({"Target" : visualizeGO(target.squeeze(), target.squeeze().shape[:4])})
        if self.trainer.current_epoch % self.hparams.log_val_plot_every_n_epoch == 0:
            x = x.squeeze()
            damaged = damaged.squeeze()
            reconstructed = reconstructed.squeeze()
            pertOut = pertOut.squeeze()
            perturbated = perturbated.squeeze()
            self.trainer.logger.experiment.log({"DAMAGED - before and after" : visualizeImprovements(x, damaged)})
            self.trainer.logger.experiment.log({"DAMAGED - Growing improvements " : visualizeImprovements(reconstructed, damaged)})
            self.trainer.logger.experiment.log({"Perturbation - before and after some steps" : visualizeImprovements(perturbated, pertOut)})
        return {"loss_from_seed":loss, "out":x, "damaged":damaged, "reconstructed":reconstructed, "loss_reconstructed":loss_reconstruction}



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"], betas=self.hparams["betas"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=self.hparams.lr_gamma, last_epoch=-1)
        return {"optimizer":optimizer, "lr_scheduler":scheduler}

    def configure_callbacks(self) -> List[pl.Callback]:
        callbacks = [PoolSamplerCallback()]
        return callbacks