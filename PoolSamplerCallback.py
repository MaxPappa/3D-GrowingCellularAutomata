import pytorch_lightning as pl
import torch
from random import sample

class PoolSamplerCallback(pl.Callback):

    def on_train_batch_end(self, trainer:pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        val = len(trainer.datamodule.pool)//trainer.num_training_batches
        idx = range(batch_idx * val, batch_idx*val+val)
        for i in range(0,len(idx)):
            trainer.datamodule.pool[idx[i]] = outputs["out"][i]