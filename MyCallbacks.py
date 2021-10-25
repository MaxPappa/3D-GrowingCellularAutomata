import pytorch_lightning as pl
import torch
from visualizationUtils import visualizeImprovements
from random import randint

class PoolSamplerCallback(pl.Callback):
    def on_train_batch_end(self, trainer:pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        val = len(trainer.datamodule.pool)//trainer.num_training_batches
        idx = range(batch_idx * val, batch_idx*val+val)
        for i in range(0,len(idx)):
            trainer.datamodule.pool[idx[i]] = outputs["out"][i]


class GrowImprovementPerEpochLogger(pl.Callback):
    def __init__(self, seed, target):
        self.seed = seed
        self.target = target

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused=None) -> None:
        out = pl_module(self.seed[None,...], steps=randint(pl_module.min_step, pl_module.max_step+1))[0]
        trainer.logger.experiment.log(
            {"Growing improvements" : visualizeImprovements(out, self.target)},
            on_step=False, on_epoch=True
        )