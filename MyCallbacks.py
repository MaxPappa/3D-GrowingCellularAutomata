import pytorch_lightning as pl
import torch
from visualizationUtils import visualizeImprovements, visualizePatterns
from random import randint, sample
import math

class PoolSamplerCallback(pl.Callback):
    ''' After each train step ends, this callback is called to push computed and updated examples into
        the pool of examples that is sampled before the start of each training step
    '''    
    def on_train_batch_end(self, trainer:pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        with torch.no_grad():
            val = math.ceil(len(trainer.datamodule.pool)/trainer.num_training_batches)
            idx = range(batch_idx * val, batch_idx*val+val)
            if val < len(idx):
                val = len(idx)
            if idx[-1]+1 > trainer.datamodule.pool.size(0):
                idx = range(idx[0], trainer.datamodule.pool.size(0))
            trainer.datamodule.pool[idx] = outputs["out"].detach().cpu().clone()

class PoolPatternSampleCallback(pl.Callback):
    '''
        This callback is used to see how the examples are evolving in the pool during training
    '''    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused=None) -> None:
        if trainer.current_epoch % 5 != 0:
            return
        with torch.no_grad():
            val = len(trainer.datamodule.pool)
            idx = torch.tensor(sample(range(val), 4))
            sampled_patterns = trainer.datamodule.pool[idx]
            trainer.logger.experiment.log({"Pool patterns" : visualizePatterns(sampled_patterns, 4)})

class GrowImprovementPerEpochLogger(pl.Callback):
    '''
        Log improvements of a seed at the end of each 5 epochs
    '''    
    def __init__(self, seed, target):
        self.seed = seed
        self.target = target

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused=None) -> None:
        if trainer.current_epoch % 5 != 0:
            return
        seed = trainer.datamodule.seed.detach().clone().to(self.device)
        out = pl_module(seed[None,...], steps=randint(pl_module.min_step, pl_module.max_step+1), fire_rate=pl_module.hparams.fire_rate)[0]
        trainer.logger.experiment.log({"Growing improvements" : visualizeImprovements(out, self.target)})