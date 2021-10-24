from pytorch_lightning.trainer import model_hooks
import CAModel
from pytorch_lightning import Trainer
import Config
import VoxelDataModule as vdm

if __name__ == '__main__':
    cfgModel = Config.ModelConfig()
    cfgData = Config.DataModuleConfig()

    ca_model = CAModel.CAModel(cfgModel)
    data_module = vdm.VoxelDataModule(cfgData)
    
    trainer = Trainer(devices=1, accelerator="gpu", log_every_n_steps=1)
    trainer.fit(ca_model, datamodule=data_module)
    #trainer = Trainer(ca_model, reload_dataloaders_every_n_epoch=1)