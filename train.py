import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import conf_utils
from training import trainer

conf = conf_utils.get_config(path='configs/conf.yaml')

if __name__ == '__main__':
    pl_module = trainer.SRTrainer(conf)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_psnr",
        dirpath='./ckpts/',
        filename=pl_module.network._get_name() + "_{epoch:02d}_{val_psnr:.2f}",
        save_top_k=1,
        mode="max",
    )
    logger = TensorBoardLogger('./logs')
    callbacks = [lr_monitor, checkpoint_callback]

    if conf['load_pretrained']:
        pretrained_path = conf['pretrained_path']
        ext = os.path.splitext(pretrained_path)[1]
        if ext == '.pth':
            pl_module.network.load_state_dict(torch.load(pretrained_path), strict=conf['strict_load'])
        elif ext == '.ckpt':
            pl_module.load_from_checkpoint(pretrained_path, conf, strict=conf['strict_load'])

    trainer = pl.Trainer(gpus=[0],
                         accelerator="ddp",
                         plugins=DDPPlugin(find_unused_parameters=False),
                         max_epochs=conf['trainer']['num_epochs'],
                         callbacks=callbacks,
                         logger=logger,
                         track_grad_norm=-1,
                         profiler=None,
                         check_val_every_n_epoch=conf['trainer']['check_val_every_n_epoch'],
                         replace_sampler_ddp=True)

    trainer.fit(pl_module)
