import pytorch_lightning as pl
import torch
from visualizationUtils import visualizeImprovements, visualizePatterns
from random import randint, sample

class PoolSamplerCallback(pl.Callback):
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: torch.tensor, batch: torch.tensor, batch_idx: int, dataloader_idx: int):
        with torch.no_grad():
            val = len(trainer.datamodule.pool)//trainer.num_training_batches
            idx = range(batch_idx * val, batch_idx*val+val)
            trainer.datamodule.pool[idx] = outputs["out"].detach().clone()


class PoolPatternSampleCallback(pl.Callback):
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused=None) -> None:
        if trainer.current_epoch % 5 != 0:
            return
        with torch.no_grad():
            val = len(trainer.datamodule.pool)
            idx = torch.tensor(sample(range(val), 4))
            sampled_patterns = trainer.datamodule.pool[idx]
            trainer.logger.experiment.log({"Pool patterns" : visualizePatterns(sampled_patterns, 4)})

class GrowImprovementPerEpochLogger(pl.Callback):
    def __init__(self, seed, target):
        self.seed = seed
        self.target = target

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused=None) -> None:
        with torch.no_grad():
            out = pl_module(self.seed[None,...], steps=randint(pl_module.min_step, pl_module.max_step+1), fire_rate=0.5)[0]
            trainer.logger.experiment.log({"Growing improvements" : visualizeImprovements(out, self.target)})