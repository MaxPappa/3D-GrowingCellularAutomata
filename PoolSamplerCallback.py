import pytorch_lightning as pl
import torch
from random import sample

class PoolSamplerCallback(pl.Callback):

    def on_train_batch_end(self, trainer:pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        #idx = sample(range(0,len(trainer.datamodule.pool)), len(batch))
        val = len(trainer.datamodule.pool)//trainer.num_training_batches#len(batch) * np.array(batch)
        idx = range(batch_idx * val, batch_idx*val+val)
        new_batch = []
        for i in range(0,len(idx)):
            #new_batch += [trainer.datamodule.pool[idx[i]]]
            trainer.datamodule.pool[idx[i]] = outputs[i]
        #pl_module.dataset.batch = torch.tensor(new_batch)

    # def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
    #     loss_rank = torch.mean(
    #         torch.pow(
    #             self.pl_module.dataset.batch[..., :4]-self.pl_module.dataset.target, 2
    #             ), [-2,-3,-4,-1]
    #     ).argsort()[::-1]
    #     pl_module.dataset.batch[loss_rank[0]] = torch.from_numpy(pl_module.dataset.seed)