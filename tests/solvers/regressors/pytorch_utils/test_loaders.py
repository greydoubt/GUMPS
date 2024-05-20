# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import patch
from gumps.solvers.regressors.pytorch_utils import loaders
import pandas as pd
import numpy as np
from gumps.solvers.regressors.regressor_data import DataSettings, DataRegression
from gumps.scalers.null_scaler import NullScaler

def test_get_core_count_cuda_available():
    with patch('torch.cuda.is_available') as mock_cuda_available:
        # Set up the mocks
        mock_cuda_available.return_value = True

        result = loaders.get_core_count()

        # Check the result
        assert result == 1

def test_get_core_count_cuda_not_available():
    with patch('torch.cuda.is_available') as mock_cuda_available:
        # Set up the mocks
        mock_cuda_available.return_value = False

        result = loaders.get_core_count()

        # Check the result
        assert result == 0

def test_get_persistent_workers_cuda_available():
    with patch('torch.cuda.is_available') as mock_cuda_available:
        # Set up the mocks
        mock_cuda_available.return_value = True

        assert loaders.get_persistent_workers() == True

def test_get_persistent_workers_cuda_not_available():
    with patch('torch.cuda.is_available') as mock_cuda_available:
        # Set up the mocks
        mock_cuda_available.return_value = False

        assert loaders.get_persistent_workers() == False


def test_get_multiprocessing_context_cuda_available():
    with patch('torch.cuda.is_available') as mock_cuda_available:
        # Set up the mocks
        mock_cuda_available.return_value = True

        assert loaders.get_multiprocessing_context() == 'spawn'

def test_get_multiprocessing_context_cuda_not_available():
    with patch('torch.cuda.is_available') as mock_cuda_available:
        # Set up the mocks
        mock_cuda_available.return_value = False

        assert loaders.get_multiprocessing_context() == None

def test_create_datamodule():
    "test creating the datamodule"
    input_data = pd.DataFrame({"x": np.linspace(0, 1, 5)})
    output_data = pd.DataFrame({"y": np.linspace(0, 1, 5)})
    parameters = loaders.DataModuleParameters(batch_size=32,
                                              shuffle=True)
    settings = DataSettings(input_data=input_data,
                            output_data=output_data)
    data = DataRegression(settings, NullScaler(), NullScaler())


    datamodule = loaders.DataModule(scaled_split=data.scaled_split,
                                    parameters=parameters)

    pd.testing.assert_frame_equal(input_data, datamodule.scaled_split.full_input)
    pd.testing.assert_frame_equal(output_data, datamodule.scaled_split.full_output)
    assert datamodule.parameters == parameters


def test_setup_predict():
    "test setting up the predict data module"
    input_data = pd.DataFrame({"x": np.linspace(0, 1, 5)})
    parameters = loaders.DataModuleParameters(batch_size=32,
                                              shuffle=True)

    datamodule = loaders.DataModulePredict(input_data = input_data,
                                    parameters=parameters)

    datamodule.setup(stage="predict")

    assert len(datamodule.predict_dataset) == 5

def test_setup_train_test_split_random():
    "test setting up the train test split"
    input_data = pd.DataFrame({"x": np.linspace(0, 1, 5)})
    output_data = pd.DataFrame({"y": np.linspace(0, 1, 5)})
    parameters = loaders.DataModuleParameters(batch_size=32,
                                              shuffle=True)

    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="random",
                            validation_fraction=0.2)
    data = DataRegression(settings, NullScaler(), NullScaler())

    datamodule = loaders.DataModule(scaled_split=data.scaled_split,
                                    parameters=parameters)

    datamodule.setup()

    assert len(datamodule.train_dataset) == 4
    assert len(datamodule.val_dataset) == 1

def test_setup_train_test_split_sequential():
    "test setting up the train test split"
    input_data = pd.DataFrame({"x": np.linspace(0, 1, 5)})
    output_data = pd.DataFrame({"y": np.linspace(0, 1, 5)})
    parameters = loaders.DataModuleParameters(batch_size=32,
                                              shuffle=True)

    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="sequential",
                            validation_fraction=0.2)
    data = DataRegression(settings, NullScaler(), NullScaler())

    datamodule = loaders.DataModule(scaled_split=data.scaled_split,
                                    parameters=parameters)

    datamodule.setup()

    assert len(datamodule.train_dataset.tensors[0]) == 4
    assert len(datamodule.val_dataset.tensors[0]) == 1
    assert len(datamodule.train_dataset.tensors[1]) == 4
    assert len(datamodule.val_dataset.tensors[1]) == 1

def test_setup_train_test_split_manual():
    "test setting up the train test split"
    input_data = pd.DataFrame({"x": np.linspace(0, 1, 5)})
    output_data = pd.DataFrame({"y": np.linspace(0, 1, 5)})
    parameters = loaders.DataModuleParameters(batch_size=32,
                                              shuffle=True)

    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="manual",
                            train_indices=[0, 1, 2],
                            validation_indices=[3, 4])
    data = DataRegression(settings, NullScaler(), NullScaler())

    datamodule = loaders.DataModule(scaled_split=data.scaled_split,
                                    parameters=parameters)

    datamodule.setup()

    assert len(datamodule.train_dataset.tensors[0]) == 3
    assert len(datamodule.val_dataset.tensors[0]) == 2
    assert len(datamodule.train_dataset.tensors[1]) == 3
    assert len(datamodule.val_dataset.tensors[1]) == 2



def test_train_dataloader():
    "test the train dataloader"
    input_data = pd.DataFrame({"x": np.linspace(0, 1, 5)})
    output_data = pd.DataFrame({"y": np.linspace(0, 1, 5)})
    parameters = loaders.DataModuleParameters(batch_size=32,
                                              shuffle=True)

    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="random",
                            validation_fraction=0.2)
    data = DataRegression(settings, NullScaler(), NullScaler())

    datamodule = loaders.DataModule(scaled_split=data.scaled_split,
                                    parameters=parameters)

    datamodule.setup()

    dataloader = datamodule.train_dataloader()

    assert len(dataloader) == 1
    for batch in dataloader:
        assert len(batch) == 2
        assert len(batch[0]) == 4
        assert len(batch[1]) == 4

def test_val_dataloader():
    "test the validation dataloader"
    input_data = pd.DataFrame({"x": np.linspace(0, 1, 5)})
    output_data = pd.DataFrame({"y": np.linspace(0, 1, 5)})
    parameters = loaders.DataModuleParameters(batch_size=32,
                                              shuffle=True)

    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="random",
                            validation_fraction=0.2)
    data = DataRegression(settings, NullScaler(), NullScaler())

    datamodule = loaders.DataModule(scaled_split=data.scaled_split,
                                    parameters=parameters)

    datamodule.setup()

    dataloader = datamodule.val_dataloader()

    assert len(dataloader) == 1
    for batch in dataloader:
        assert len(batch) == 2
        assert len(batch[0]) == 1
        assert len(batch[1]) == 1

def test_predict_dataloader():
    "test the predict dataloader"
    input_data = pd.DataFrame({"x": np.linspace(0, 1, 5)})
    parameters = loaders.DataModuleParameters(batch_size=32,
                                              shuffle=True)

    datamodule = loaders.DataModulePredict(input_data = input_data,
                                    parameters=parameters)

    datamodule.setup(stage="predict")

    dataloader = datamodule.predict_dataloader()

    assert len(dataloader) == 1
    for batch in dataloader:
        assert len(batch) == 1
        assert len(batch[0]) == 5

