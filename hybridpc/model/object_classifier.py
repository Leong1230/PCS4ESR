from pyexpat import features
import torch
import time
import os
import torch.nn as nn
import numpy as np
import math
import torchmetrics
import pytorch_lightning as pl
from module import Backbone, ImplicitDecoder

class ObjectClassifier(pl.LightningModule):
    def __init__(self, model, data, optimizer, lr_decay, inference=None):
        super().__init__()
        self.save_hyperparameters()

        self.seg_decoder = ImplicitDecoder(
            model.latent_dim,
            model.seg_decoder.input_dim,
            model.seg_decoder.hidden_dim,
            model.seg_decoder.num_hidden_layers_before_skip,
            model.seg_decoder.num_hidden_layers_after_skip,
            data.category_num,
        )
        self.functa_decoder = ImplicitDecoder(
            model.latent_dim,
            model.functa_decoder.input_dim,
            model.functa_decoder.hidden_dim,
            model.functa_decoder.num_hidden_layers_before_skip,
            model.functa_decoder.num_hidden_layers_after_skip,
            1,
        )