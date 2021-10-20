import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from visualizationUtils import visualizeGO

class LitCAModel(pl.LightningModule):
    def __init__(self, channel_n, fire_rate, hidden_size=128):
        super(LitCAModel, self).__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.fc0 = nn.Linear(channel_n*4, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()
        self.automatic_optimization = True

    def alive(self, x):
        return F.max_pool3d(x[:, 3:4, :, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, kernel):
            kernel = kernel[None,None,...].repeat(self.channel_n, 1, 1, 1, 1)
            return F.conv3d(x, kernel, padding=1, groups=self.channel_n)

        kernel_x = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                             [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
                             [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]) / 26
        kernel_y = kernel_x.transpose(0,1)
        kernel_z = kernel_x.transpose(1,2)

        y1 = _perceive_with(x,kernel_x)
        y2 = _perceive_with(x,kernel_y)
        y3 = _perceive_with(x,kernel_z)
        y = torch.cat((x,y1,y2,y3),1)
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,4)
        mask = x[:, 3:4, :, :, :]
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,4)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,4)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0, path=None):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
            if path:
                fig = visualizeGO(x[0], True)
                fig.write_image(f"{path}_{step}.png")
        return x

    def training_step(self, train_batch, batch_idx):
        x, targ = train_batch
        loss = F.mse_loss(x[:, :, :, :, :4], targ[:,:,:,:,:4])
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        betas = (0.5, 0.5)
        lr = 2e-3
        lr_gamma = 0.9999
        optimizer = torch.optim.Adam(self.parameters, lr=lr, betas=betas)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)
        return optimizer, scheduler