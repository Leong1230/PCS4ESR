import os
import hydra
import torch
from importlib import import_module
import pytorch_lightning as pl
from hybridpc.data.data_module import DataModule


def init_model(cfg):
    return getattr(import_module("hybridpc.model"), cfg.model.model.module) \
        (cfg.model.model, cfg.data, cfg.model.optimizer, cfg.model.lr_decay, cfg.model.inference)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    # fix the seed
    pl.seed_everything(cfg.global_test_seed, workers=True)

    print("=> initializing trainer...")
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1, logger=False)

    output_path = os.path.join(cfg.exp_output_root_path, "inference", cfg.model.inference.split, "udf_visualizations")
    os.makedirs(output_path, exist_ok=True)

    print("==> initializing data ...")
    data_module = DataModule(cfg)

    print("=> initializing model...")
    model = getattr(import_module("hybridpc.model"), cfg.model.network.module)(cfg)

    print("=> start inference...")
    checkpoint = torch.load(cfg.model.ckpt_path)
    # trainer.fit_loop.epoch_progress.current.completed = checkpoint["epoch"]  # TODO
    trainer.test(model=model, datamodule=data_module, ckpt_path=cfg.model.ckpt_path)


if __name__ == '__main__':
    main()
