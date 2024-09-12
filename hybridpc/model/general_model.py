from pyexpat import features
from sklearn.metrics import jaccard_score
import torch
import time
import os
import torch.nn as nn
import numpy as np
import math
import torchmetrics
import open3d as o3d
import importlib
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pl_bolts
import hydra
from collections import OrderedDict
from typing import Mapping, Any, Optional
from pycg import exp, image
from hybridpc.optimizer.optimizer import cosine_lr_decay
from hybridpc.model.module import Backbone, Dense_Generator
from torch.nn import functional as F
from lightning.pytorch.utilities import grad_norm


class GeneralModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.training_stage = cfg.model.training_stage
        self.val_test_step_outputs = []
        # For recording test information
        # step -> log_name -> log_value (list of ordered-dict)
        self.test_logged_values = []
        self.record_folder = None
        self.record_headers = []
        self.record_data_cache = {}
        self.last_test_valid = False

    def configure_optimizers(self):
        # params_to_optimize = list(self.encoder.parameters()) + list(self.mask_decoder.parameters()) + list(self.udf_decoder.parameters())
        # if hasattr(self, 'mask_decoder'):
        #     params_to_optimize += list(self.mask_decoder.parameters())
    
        params_to_optimize = self.parameters()
        
        if self.hparams.model.optimizer.name == "SGD":
            optimizer = torch.optim.SGD(
                params_to_optimize,
                lr=self.hparams.model.optimizer.lr,
                momentum=0.9,
                weight_decay=1e-4,
            )
            scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=int(self.hparams.model.optimizer.warmup_steps_ratio * self.hparams.model.trainer.max_steps),
                max_epochs=self.hparams.model.trainer.max_steps,
                eta_min=0,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }

        elif self.hparams.model.optimizer.name == 'Adam':
            optimizer = torch.optim.Adam(
                params_to_optimize,
                lr=self.hparams.model.optimizer.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-4,
            )
            return optimizer

        else:
            logging.error('Optimizer type not supported')

    def training_step(self, data_dict):
        pass 

    # def on_before_optimizer_step(self, optimizer, optimizer_idx):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     udf_norms = grad_norm(self.udf_decoder, norm_type=2)
    #     mask_norms = grad_norm(self.mask_decoder, norm_type=2)
    #     encoder_norms = grad_norm(self.encoder, norm_type=2)
    #     # self.log_dict("train/udf_norms", udf_norms)
    #     # self.log_dict("train/mask_norms", mask_norms)
    #     # self.log_dict("train/encoder_norms", encoder_norms)

    def on_train_epoch_end(self):
        if self.hparams.model.optimizer.name == 'Adam':
            # Update the learning rates for Adam optimizers
            cosine_lr_decay(
                self.trainer.optimizers[0], self.hparams.model.optimizer.lr, self.current_epoch,
                self.hparams.model.lr_decay.decay_start_epoch, self.hparams.model.trainer.max_epochs, 1e-6
            )

    def validation_step(self, data_dict, idx):
        pass

    def validation_epoch_end(self, outputs):
        metrics_to_log = ['chamfer-L1', 'f-score', 'f-score-20']
        if outputs:
            avg_metrics = {metric: np.mean([x[metric] for x in outputs]) for metric in metrics_to_log if metric in outputs[0]}
            for key, value in avg_metrics.items():
                self.log(f"val_reconstruction/{key}", value, logger=True)



    def test_step(self, data_dict, idx):
        pass

    # def log(self, name: str, value: Any, *args, **kwargs):
    #     """
    #         Override PL's default log function, to better support test-time or overfit-time no-logger logging
    #     (using various types including string).
    #     """
    #     # For overfitting, we leave the log to the log_dict monkey_patch.
    #     # if isinstance(self.trainer, omegaconf.dictconfig.DictConfig):
    #     #     return
    #     if self.trainer.testing:
    #         if name == 'batch-idx':
    #             self.test_logged_values.append(OrderedDict([(name, value)]))
    #         else:
    #             if isinstance(value, torch.Tensor):
    #                 value = value.item()
    #             self.test_logged_values[-1][name] = value
    #     else:
    #         super().log(name=name, value=value, *args, **kwargs)

    def log_dict_prefix(
        self,
        prefix: str,
        dictionary: Mapping[str, Any],
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None
    ):
        """
        This overrides fixes if dict key is not a string...
        """
        dictionary = {
            prefix + "/" + str(k): v for k, v in dictionary.items()
        }
        self.log_dict(dictionary=dictionary,
                      prog_bar=prog_bar,
                      logger=logger, on_step=on_step, on_epoch=on_epoch)

    def log_image(self, name: str, img: np.ndarray):
        if self.trainer.logger is not None:
            # if self.logger_type == 'tb':
            #     if img.shape[2] <= 4:
            #         # WHC -> CWH
            #         img = np.transpose(img, (2, 0, 1))
            #     self.trainer.logger.experiment.add_image(name, img, self.trainer.global_step)
            # elif self.logger_type == 'wandb':
            self.trainer.logger.log_image(key=name, images=[img])

    # def log_plot(self, name: str, fig: matplotlib.figure.Figure, close: bool = True):
    #     img = image.from_mplot(fig, close)
    #     self.log_image(name, img)
    #     if close:
    #         plt.close(fig)

    def log_geometry(self, name: str, geom, draw_color: bool = False):
        if self.trainer.logger is None:
            return
        if isinstance(geom, o3d.geometry.TriangleMesh):
            try:
                from pycg import render
                mv_img = render.multiview_image(
                    [geom], viewport_shading='LIT' if draw_color else 'NORMAL', backend='filament')
                # mv_img = render.multiview_image(
                    # [geom], viewport_shading='LIT' if draw_color else 'NORMAL', backend='opengl')
                self.log_image("mesh" + name, mv_img)
            except Exception:
                exp.logger.warning("Not able to render mesh during training.")
        else:
            raise NotImplementedError

    def test_log_data(self, data_dict: dict):
        # Output artifact data only when there is no focus.
        # if self.record_folder is None or self.hparams.focus == "none":
        #     return
        self.record_data_cache.update(data_dict)