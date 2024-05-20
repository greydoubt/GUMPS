# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import random
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import matplotlib

from gumps.solvers.regressors.pytorch_utils.models import MLP, MLPParameters

def seed_everything(seed: int):
    "based on  https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964 in order to make CI deterministic"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class TestTorchMLP(unittest.TestCase):
    "test the torch MLP model, this is the core regressor"

    def setUp(self):
        seed_everything(0)
        self.batch_size = 32
        self.num_samples = 100

        self.MLPParameters = MLPParameters(input_dim=10,
                                            output_dim=2,
                                            hidden_layer_sizes=[64, 32, 16],
                                            activation=torch.nn.SiLU,
                                            learning_rate="adaptive",
                                            learning_rate_init=0.001,
                                            learning_rate_patience=10,
                                            learning_rate_factor=0.1)

        # Create dummy input and target tensors
        self.inputs = torch.randn(self.num_samples, self.MLPParameters.input_dim)
        self.targets = torch.randn(self.num_samples, self.MLPParameters.output_dim)

        # Create dummy validation input and target tensors
        self.val_inputs = torch.randn(self.num_samples, self.MLPParameters.input_dim)
        self.val_targets = torch.randn(self.num_samples, self.MLPParameters.output_dim)

        # Create a dummy dataset and dataloader for testing
        dataset = TensorDataset(self.inputs, self.targets)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Create a dummy validation dataset and dataloader for testing
        val_dataset = TensorDataset(self.val_inputs, self.val_targets)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        #Create a dummy prediction dataset and dataloader for testing
        predict_dataset = TensorDataset(self.inputs)
        self.predict_dataloader = DataLoader(predict_dataset, batch_size=self.batch_size, shuffle=False)

    def test_initialize_constant(self):
        "test the initialization of the sequential model"
        self.MLPParameters.learning_rate = "constant"
        model = MLP(self.MLPParameters)
        self.assertEqual(model.settings.input_dim, self.MLPParameters.input_dim)
        self.assertEqual(model.settings.output_dim, self.MLPParameters.output_dim)
        self.assertEqual(model.settings.hidden_layer_sizes, self.MLPParameters.hidden_layer_sizes)
        self.assertEqual(model.settings.activation, self.MLPParameters.activation)
        self.assertEqual(model.settings.learning_rate_init, self.MLPParameters.learning_rate_init)
        self.assertIsInstance(model.loss_fn, torch.nn.MSELoss)
        self.assertEqual(len(model.model), 7)


    def test_initialize_bad_learning_rate(self):
        "test the initialization of the sequential model"
        with self.assertRaises(ValueError):
            self.MLPParameters.learning_rate = "foo"


    def test_initialize(self):
        "test the initialization of the model"
        model = MLP(self.MLPParameters)
        self.assertEqual(model.settings.input_dim, self.MLPParameters.input_dim)
        self.assertEqual(model.settings.output_dim, self.MLPParameters.output_dim)
        self.assertEqual(model.settings.hidden_layer_sizes, self.MLPParameters.hidden_layer_sizes)
        self.assertEqual(model.settings.activation, self.MLPParameters.activation)
        self.assertEqual(model.settings.learning_rate_init, self.MLPParameters.learning_rate_init)
        self.assertIsInstance(model.loss_fn, torch.nn.MSELoss)
        self.assertEqual(len(model.model), 7)


    def test_forward_pass(self):
        "test the forward pass"
        model = MLP(self.MLPParameters)
        inputs = torch.randn(self.batch_size, self.MLPParameters.input_dim)
        outputs = model.forward(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.MLPParameters.output_dim))


    def test_predict_step(self):
        "test the predict step"
        model = MLP(self.MLPParameters)
        batch = next(iter(self.predict_dataloader))
        outputs = model.predict_step(batch, 0)
        self.assertEqual(outputs.shape, (self.batch_size, self.MLPParameters.output_dim))


    def test_training_step(self):
        "test the training step"
        model = MLP(self.MLPParameters)
        batch = next(iter(self.dataloader))
        loss = model.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)


    def test_validation_step(self):
        """The validation step can't be tested in isolation from what I can figure out.
        It needs to be run from inside a training loop.  This is a limitation of pytorch lightning.
        This method is here just to document this limitation and so it is obvious that testing
        was not skipped. This method is tested as part of the test_training method."""


    def test_training(self):
        "test that the model can be trained for a few steps, this test just verifies that the training loop runs"
        model = MLP(self.MLPParameters)
        trainer = pl.Trainer(
            max_epochs=10,
            logger=False,
            enable_checkpointing=False
        )

        trainer.fit(model, self.dataloader, self.val_dataloader)

    def test_plot_loss_curves(self):
        "test that the matplotlib is generated and that train and validation have the right number of entries"
        model = MLP(self.MLPParameters)
        trainer = pl.Trainer(
            max_epochs=10,
            logger=False,
            enable_checkpointing=False
        )

        trainer.fit(model, self.dataloader, self.val_dataloader)

        axes = model.plot_loss_curves()
        self.assertIsInstance(axes, matplotlib.axes._axes.Axes)
        self.assertEqual(len(model.train_losses), 10)
        self.assertEqual(len(model.validation_losses), 10)


    def test_configure_optimizers(self):
        "test that the optimizer is configured correctly"
        model = MLP(self.MLPParameters)
        optimizer = model.configure_optimizers()
        self.assertIsInstance(optimizer['optimizer'], torch.optim.Adam)


    def test_create_layers(self):
        "test the create layers method"
        model = MLP(self.MLPParameters)
        layers = model._create_layers()
        self.assertEqual(len(layers), 7)


