# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import pandas as pd
import attrs
import math
from typing import Callable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.axes
import pytorch_lightning as pl

@attrs.define
class MLPParameters:
    input_dim: int
    output_dim: int
    hidden_layer_sizes: list[int]
    activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.SiLU
    learning_rate: str = attrs.field(default="adaptive")
    @learning_rate.validator
    def _check_learning_rate(self, _, value):
        if value not in ["constant", "adaptive"]:
            raise ValueError(f"learning_rate must be 'constant' or 'adaptive', not {value}")

    learning_rate_init: float = 0.001
    learning_rate_patience:int = 10
    learning_rate_factor:float = 0.1



class MLP(pl.LightningModule):
    "basic multi-layer perceptron for pyTorch Lightning"

    def __init__(
        self,
        settings: MLPParameters,
    ):
        "initialize the model"
        super().__init__()
        self.settings = settings
        self.loss_fn = nn.MSELoss()
        self.train_losses: list[float] = []
        self.validation_losses: list[float] = []

        self.model = self._create_layers()


    def _create_layers(self):
        "create the layers of the model"
        layers = []
        input_dim = self.settings.input_dim
        for hidden_dim in self.settings.hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.settings.activation())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, self.settings.output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        "run the forward pass"
        return self.model(x)

    #batch_idx is part of the interface and needs to be here even if not used
    def predict_step(self, batch, batch_idx):
        "predict the output"
        x, = batch
        return self(x)

    def configure_optimizers(self):
        "configure the optimizers with automatic learning rate reduction"
        settings = {}
        optimizer = torch.optim.Adam(self.parameters(), lr=self.settings.learning_rate_init)

        settings['optimizer'] = optimizer

        if self.settings.learning_rate == "adaptive":
            scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                      factor=self.settings.learning_rate_factor,
                                      patience=self.settings.learning_rate_patience,
                                      verbose=True)
            settings['lr_scheduler'] = scheduler
            settings['monitor'] = 'val_log_loss'
        return settings

    #batch_idx is part of the interface and needs to be here even if not used
    def training_step(self, batch, batch_idx):
        "single training step"
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    #batch_idx is part of the interface and needs to be here even if not used
    def validation_step(self, batch, batch_idx):
        "single validation step"
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        log_loss = math.log10(loss.item())

        self.log('val_log_loss', log_loss)

    def on_train_epoch_end(self) -> None:
        train_loss = float(self.trainer.logged_metrics['train_loss'])
        val_loss = float(self.trainer.logged_metrics['val_loss'])

        self.train_losses.append(train_loss)
        self.validation_losses.append(val_loss)
        return super().on_train_epoch_end()

    def plot_loss_curves(self) -> matplotlib.axes._axes.Axes:
        "plot loss curves"
        df = pd.DataFrame({'train_loss': self.train_losses, 'validation_loss': self.validation_losses})
        return df.plot(logy=True, figsize=(10, 10))