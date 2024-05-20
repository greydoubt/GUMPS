# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import attrs
from gumps.solvers.regressors.regressor_data import Split

@attrs.define
class DataModuleParameters:
    batch_size: int
    shuffle: bool

def get_core_count() -> int:
    "return the number of cores to use"
    if torch.cuda.is_available():
        #don't use more than 2 cores, it slows down the training
        #regression problems don't have enough data to benefit from
        #more cores for loading
        return 1
    else:
        return 0

def get_persistent_workers() -> bool:
    "return whether or not to use persistent workers"
    if torch.cuda.is_available():
        return True
    return False

def get_multiprocessing_context() -> None|str:
    "return the multiprocessing context"
    if get_core_count() > 0:
        return "spawn"
    return None

class DataModule(pl.LightningDataModule):
    "pytorch lightning data module for training and validation"
    def __init__(self, *, scaled_split: Split,
                 parameters:DataModuleParameters):
        "initialize the data module"
        super().__init__()
        self.scaled_split = scaled_split
        self.parameters = parameters

    def setup(self, stage=None):
        "setup the data module"
        split = self.scaled_split

        train_input_tensor = torch.tensor(split.train_input.values)
        validation_input_tensor = torch.tensor(split.validation_input.values)
        train_output_tensor = torch.tensor(split.train_output.values)
        validation_output_tensor = torch.tensor(split.validation_output.values)

        self.train_dataset = TensorDataset(train_input_tensor, train_output_tensor)
        self.val_dataset = TensorDataset(validation_input_tensor, validation_output_tensor)

    def train_dataloader(self):
        "return the training dataloader"
        return DataLoader(self.train_dataset,
                          batch_size=self.parameters.batch_size,
                          shuffle=self.parameters.shuffle,
                          multiprocessing_context=get_multiprocessing_context(),
                          persistent_workers=get_persistent_workers(),
                          num_workers=get_core_count())

    def val_dataloader(self):
        "return the validation dataloader"
        return DataLoader(self.val_dataset,
                          batch_size=self.parameters.batch_size,
                          shuffle=False,
                          multiprocessing_context=get_multiprocessing_context(),
                          persistent_workers=get_persistent_workers(),
                          num_workers=get_core_count())

class DataModulePredict(pl.LightningDataModule):
    "pytorch lightning data module for training and validation"
    def __init__(self, *, input_data: pd.DataFrame,
                 parameters:DataModuleParameters):
        "initialize the data module"
        super().__init__()
        self.input_data = input_data
        self.parameters = parameters

    def setup(self, stage=None):
        "setup the data module"
        input_tensor = torch.tensor(self.input_data.values)
        self.predict_dataset = TensorDataset(input_tensor,)

    def predict_dataloader(self):
        "return the dataloader"
        return DataLoader(self.predict_dataset,
                          batch_size=self.parameters.batch_size,
                          shuffle=False,
                          multiprocessing_context=get_multiprocessing_context(),
                          persistent_workers=get_persistent_workers(),
                          num_workers=get_core_count())
