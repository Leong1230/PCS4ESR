import os
import hydra
import pytorch_lightning as pl
from hybridpc.callback import *
from importlib import import_module
from hybridpc.data.data_module import DataModule
from pytorch_lightning.callbacks import LearningRateMonitor


def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.model.checkpoint_monitor)
    gpu_cache_clean_monitor = GPUCacheCleanCallback()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_monitor, gpu_cache_clean_monitor, lr_monitor]


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.global_train_seed, workers=True)

    output_path = os.path.join(cfg.exp_output_root_path, "training")
    os.makedirs(output_path, exist_ok=True)

    print("==> initializing data ...")
    data_module = DataModule(cfg)

    print("==> initializing logger ...")
    logger = hydra.utils.instantiate(cfg.model.logger, save_dir=output_path)

    print("==> initializing monitor ...")
    callbacks = init_callbacks(cfg)

    print("==> initializing trainer ...")
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **cfg.model.trainer)

    print("==> initializing model ...")
    model = getattr(import_module("hybridpc.model"), cfg.model.network.module)(cfg)

    #Load the model parameters from the checkpoint and set it to the model
    if os.path.isfile(cfg.model.stage1_ckpt_path):
        print("==> loading model from checkpoint...")
        model = model.load_from_checkpoint(cfg.model.stage1_ckpt_path, cfg)

    print("==> start training ...")
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.model.ckpt_path)

if __name__ == '__main__':
    main()