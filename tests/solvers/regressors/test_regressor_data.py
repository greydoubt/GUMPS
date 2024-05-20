# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#test the regressor_data module using pytest

from gumps.solvers.regressors.regressor_data import DataSettings, Split, DataRegression, keep_unique
from gumps.scalers.null_scaler import NullScaler
import pandas as pd
import numpy as np

def test_Split():
    "Test the Split class"
    train_input = pd.DataFrame({"a":[1,2,3], "b":[4,5,6]})
    validation_input = pd.DataFrame({"a":[7,8,9], "b":[10,11,12]})
    train_output = pd.DataFrame({"c":[13,14,15]})
    validation_output = pd.DataFrame({"c":[16,17,18]})
    full_input = pd.concat([train_input, validation_input])
    full_output = pd.concat([train_output, validation_output])
    split = Split(train_input=train_input,
                  validation_input=validation_input,
                  train_output=train_output,
                  validation_output=validation_output,
                  full_input=full_input,
                  full_output=full_output)
    pd.testing.assert_frame_equal(split.train_input, train_input)
    pd.testing.assert_frame_equal(split.validation_input, validation_input)
    pd.testing.assert_frame_equal(split.train_output, train_output)
    pd.testing.assert_frame_equal(split.validation_output, validation_output)

def test_DataSettings():
    "Test the DataSettings class"
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            validation_fraction=0.1)
    pd.testing.assert_frame_equal(settings.input_data, input_data)
    pd.testing.assert_frame_equal(settings.output_data, output_data)
    assert settings.validation_fraction == 0.1
    assert settings.random_state is None
    assert settings.train_test_split == "random"
    assert settings.train_indices is not None
    assert settings.validation_indices is not None
    assert len(settings.train_indices) == 9
    assert len(settings.validation_indices) == 1

def test_DataSettings_sequential_split():
    "Test the DataSettings class with a sequential split"
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="sequential",
                            validation_fraction=0.1)
    pd.testing.assert_frame_equal(settings.input_data, input_data)
    pd.testing.assert_frame_equal(settings.output_data, output_data)
    assert settings.validation_fraction == 0.1
    assert settings.random_state is None
    assert settings.train_test_split == "sequential"
    assert settings.train_indices is not None
    assert settings.validation_indices is not None
    assert all(settings.train_indices == np.array([0,1,2,3,4,5,6,7,8]))
    assert all(settings.validation_indices == np.array([9]))
    assert len(settings.train_indices) == 9
    assert len(settings.validation_indices) == 1

def test_DataSettings_random_split():
    "Test the DataSettings class with a random split"
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="random",
                            random_state=42,
                            validation_fraction=0.1)
    pd.testing.assert_frame_equal(settings.input_data, input_data)
    pd.testing.assert_frame_equal(settings.output_data, output_data)
    assert settings.validation_fraction == 0.1
    assert settings.random_state == 42
    assert settings.train_test_split == "random"
    assert settings.train_indices is not None
    assert settings.validation_indices is not None
    assert all(settings.train_indices == np.array([1, 5, 0, 7, 2, 9, 4, 3, 6]))
    assert all(settings.validation_indices == np.array([8]))
    assert len(settings.train_indices) == 9
    assert len(settings.validation_indices) == 1

def test_DataSettings_manual_split():
    "Test the DataSettings class with a manual split"
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="manual",
                            train_indices=np.array([1,2,3,4,5,6,7,8,9]),
                            validation_indices=np.array([0]))
    pd.testing.assert_frame_equal(settings.input_data, input_data)
    pd.testing.assert_frame_equal(settings.output_data, output_data)
    assert settings.validation_fraction == 0.1
    assert settings.random_state is None
    assert settings.train_test_split == "manual"
    assert settings.train_indices is not None
    assert settings.validation_indices is not None
    assert all(settings.train_indices == np.array([1,2,3,4,5,6,7,8,9]))
    assert all(settings.validation_indices == np.array([0]))
    assert len(settings.train_indices) == 9
    assert len(settings.validation_indices) == 1

def test_DataSetting_validation_fraction_less_than_0():
    "Test the DataSettings class with a validation fraction less than 0"
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    try:
        settings = DataSettings(input_data=input_data,
                                output_data=output_data,
                                validation_fraction=-0.1)
    except ValueError as e:
        assert str(e) == "validation_fraction must be between 0 and 1, not -0.1"

def test_DataSetting_validation_fraction_greater_than_1():
    "Test the DataSettings class with a validation fraction greater than 1"
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    try:
        settings = DataSettings(input_data=input_data,
                                output_data=output_data,
                                validation_fraction=1.1)
    except ValueError as e:
        assert str(e) == "validation_fraction must be between 0 and 1, not 1.1"

def test_DataSetting_validation_fraction_1():
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    try:
        settings = DataSettings(input_data=input_data,
                                output_data=output_data,
                                validation_fraction=1.0)
    except ValueError as e:
        assert str(e) == "validation_fraction must be between 0 and 1, not 1.0"

def test_DataSettings_validation_fraction_0():
    "Test the DataSettings class with a validation fraction of 0"
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            validation_fraction=0)
    pd.testing.assert_frame_equal(settings.input_data, input_data)
    pd.testing.assert_frame_equal(settings.output_data, output_data)
    assert settings.validation_fraction == 0
    assert settings.random_state is None
    assert settings.train_test_split == "random"
    assert settings.train_indices is not None
    assert settings.validation_indices is not None
    assert all(settings.train_indices == np.array([0,1,2,3,4,5,6,7,8,9]))
    assert all(settings.validation_indices == np.array([]))
    assert len(settings.train_indices) == 10
    assert len(settings.validation_indices) == 0

def test_DataSetting_calculate_validation_fraction_manual():
    "Test the calculate validation fraction method with a manual split"
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="manual",
                            train_indices=np.array([1,2,3,4,5,6,7,8,9]),
                            validation_indices=np.array([0]))
    assert settings.calculate_validation_fraction() == 0.1

def test_DataSetting_calculate_validation_fraction_0():
    "Test the calculate validation fraction method with a validation fraction of 0"
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            validation_fraction=0)
    assert settings.calculate_validation_fraction() == 0

def test_DataSettings_regenerate_indices_random():
    "Test the regenerate indices method with a random split"
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="random",
                            random_state=42,
                            validation_fraction=0.1)
    train_indices, validation_indices = settings.regenerate_indices()
    assert all(train_indices == np.array([1, 5, 0, 7, 2, 9, 4, 3, 6]))
    assert all(validation_indices == np.array([8]))

def test_DataSettings_regenerate_indices_sequential():
    "Test the regenerate indices method with a sequential split"
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="sequential",
                            validation_fraction=0.1)
    train_indices, validation_indices = settings.regenerate_indices()
    assert all(train_indices == np.array([0,1,2,3,4,5,6,7,8]))
    assert all(validation_indices == np.array([9]))

def test_DataSettings_regenerate_indices_manual():
    "Test the regenerate indices method with a manual split"
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="manual",
                            train_indices=np.array([1,2,3,4,5,6,7,8,9]),
                            validation_indices=np.array([0]))
    try:
        settings.regenerate_indices()
    except ValueError as e:
        assert str(e) == "Cannot regenerate indices when train_test_split is 'manual'"

def test_DataSettings_regenerate_indices_validation_fraction_0():
    "Test the calculate validation fraction method with a validation fraction of 0"
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            validation_fraction=0)
    train_indices, validation_indices = settings.regenerate_indices()
    assert all(train_indices == np.array([0,1,2,3,4,5,6,7,8,9]))
    assert all(validation_indices == np.array([]))

def test_DataRegression():
    input_scaler = NullScaler()
    output_scaler = NullScaler()
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data)

    data_regression = DataRegression(settings, input_scaler, output_scaler)

    assert data_regression.settings == settings
    assert data_regression.input_scaler == input_scaler
    assert data_regression.output_scaler == output_scaler
    assert data_regression.split is not None
    assert data_regression.scaled_split is not None

def test_DataRegression_split_data():
    input_scaler = NullScaler()
    output_scaler = NullScaler()
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="manual",
                            train_indices=np.array([0,1,2,3,4,5,6,7,8]),
                            validation_indices=np.array([9]))

    data_regression = DataRegression(settings, input_scaler, output_scaler)

    assert data_regression.settings == settings
    assert data_regression.input_scaler == input_scaler
    assert data_regression.output_scaler == output_scaler
    assert data_regression.split is not None
    assert data_regression.scaled_split is not None
    pd.testing.assert_frame_equal(data_regression.split.train_input, pd.DataFrame({"a":np.linspace(0, 1,10)[:9], "b":np.linspace(0, 1,10)[:9]}))
    pd.testing.assert_frame_equal(data_regression.split.validation_input, pd.DataFrame({"a":[1.0], "b":[1.0]}, index=[9]))
    pd.testing.assert_frame_equal(data_regression.split.train_output, pd.DataFrame({"c":np.linspace(0, 1,10)[:9]}))
    pd.testing.assert_frame_equal(data_regression.split.validation_output, pd.DataFrame({"c":[1.0]}, index=[9]))
    pd.testing.assert_frame_equal(data_regression.scaled_split.train_input, pd.DataFrame({"a":np.linspace(0, 1,10)[:9], "b":np.linspace(0, 1,10)[:9]}))
    pd.testing.assert_frame_equal(data_regression.scaled_split.validation_input, pd.DataFrame({"a":[1.0], "b":[1.0]}, index=[9]))
    pd.testing.assert_frame_equal(data_regression.scaled_split.train_output, pd.DataFrame({"c":np.linspace(0, 1,10)[:9]}))
    pd.testing.assert_frame_equal(data_regression.scaled_split.validation_output, pd.DataFrame({"c":[1.0]}, index=[9]))

def test_DataRegression_fit_scaler():
    input_scaler = NullScaler()
    output_scaler = NullScaler()
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="manual",
                            train_indices=np.array([0,1,2,3,4,5,6,7,8]),
                            validation_indices=np.array([9]))

    data_regression = DataRegression(settings, input_scaler, output_scaler)

    data_regression.fit_scaler()

    assert data_regression.input_scaler.columns == ["a", "b"]
    assert data_regression.output_scaler.columns == ["c"]

def test_DataRegression_scale_data():
    input_scaler = NullScaler()
    output_scaler = NullScaler()
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="manual",
                            train_indices=np.array([0,1,2,3,4,5,6,7,8]),
                            validation_indices=np.array([9]))

    data_regression = DataRegression(settings, input_scaler, output_scaler)

    data_regression.fit_scaler()

    scaled_split = data_regression.scale_data()

    pd.testing.assert_frame_equal(scaled_split.train_input, data_regression.split.train_input)
    pd.testing.assert_frame_equal(scaled_split.validation_input, data_regression.split.validation_input)
    pd.testing.assert_frame_equal(scaled_split.train_output, data_regression.split.train_output)
    pd.testing.assert_frame_equal(scaled_split.validation_output, data_regression.split.validation_output)

def test_keep_unique_no_duplicates():
    "test the keep unique method"
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})

    unique_input_data, unique_output_data = keep_unique(input_data, output_data)

    pd.testing.assert_frame_equal(unique_input_data, input_data)
    pd.testing.assert_frame_equal(unique_output_data, output_data)

def test_keep_unique_duplicates():
    "test the keep unique method"
    input_data = pd.DataFrame({"x": [1,2,3,3,3]})
    output_data = pd.DataFrame({"y": [1,2,3,3,4]})

    correct_input_data = pd.DataFrame({"x": [1,2,3,3]})
    correct_output_data = pd.DataFrame({"y": [1,2,3,4]})

    new_input_data, new_output_data = keep_unique(input_data, output_data)

    pd.testing.assert_frame_equal(new_input_data, correct_input_data)
    pd.testing.assert_frame_equal(new_output_data, correct_output_data)

def test_DataRegression_update_data():
    input_scaler = NullScaler()
    output_scaler = NullScaler()
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="random",
                            validation_fraction=0.1)

    data_regression = DataRegression(settings, input_scaler, output_scaler)

    new_input_data = pd.DataFrame({"a":np.linspace(2, 3, 10), "b":np.linspace(2, 3, 10)})
    new_output_data = pd.DataFrame({"c":np.linspace(2, 3, 10)})

    data_regression.update_data(new_input_data, new_output_data)

    pd.testing.assert_frame_equal(data_regression.settings.input_data, pd.concat([input_data, new_input_data]).reset_index(drop=True))
    pd.testing.assert_frame_equal(data_regression.settings.output_data, pd.concat([output_data, new_output_data]).reset_index(drop=True))
    assert len(data_regression.settings.train_indices) == 18
    assert len(data_regression.settings.validation_indices) == 2
    assert len(data_regression.split.train_input) == 18
    assert len(data_regression.split.validation_input) == 2
    assert len(data_regression.scaled_split.train_input) == 18
    assert len(data_regression.scaled_split.validation_input) == 2

def test_DataRegression_update_data_manual_exception():
    input_scaler = NullScaler()
    output_scaler = NullScaler()
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="manual",
                            train_indices=np.array([0,1,2,3,4,5,6,7,8]),
                            validation_indices=np.array([9]))

    data_regression = DataRegression(settings, input_scaler, output_scaler)

    new_input_data = pd.DataFrame({"a":np.linspace(2, 3, 10), "b":np.linspace(2, 3, 10)})
    new_output_data = pd.DataFrame({"c":np.linspace(2, 3, 10)})

    try:
        data_regression.update_data(new_input_data, new_output_data)
    except ValueError as e:
        assert str(e) == "Cannot update data when train_test_split is 'manual' use update_split instead"

def test_DataRegression_reset_split():
    input_scaler = NullScaler()
    output_scaler = NullScaler()
    input_data = pd.DataFrame({"a":np.linspace(0, 1, 10), "b":np.linspace(0, 1, 10)})
    output_data = pd.DataFrame({"c":np.linspace(0, 1, 10)})
    settings = DataSettings(input_data=input_data,
                            output_data=output_data,
                            train_test_split="random")

    data_regression = DataRegression(settings, input_scaler, output_scaler)

    new_input_data = pd.DataFrame({"a":np.linspace(2, 3, 20), "b":np.linspace(2, 3, 20)})
    new_output_data = pd.DataFrame({"c":np.linspace(2, 3, 20)})

    new_train_input = new_input_data[:10]
    new_validation_input = new_input_data[10:]
    new_train_output = new_output_data[:10]
    new_validation_output = new_output_data[10:]

    new_split = Split(train_input=new_train_input,
                      validation_input=new_validation_input,
                      train_output=new_train_output,
                      validation_output=new_validation_output,
                      full_input=new_input_data,
                      full_output=new_output_data)

    data_regression.reset_split(new_split)

    pd.testing.assert_frame_equal(data_regression.settings.input_data, new_input_data)
    pd.testing.assert_frame_equal(data_regression.settings.output_data, new_output_data)
    assert len(data_regression.settings.train_indices) == 10
    assert len(data_regression.settings.validation_indices) == 10
    assert len(data_regression.split.train_input) == 10
    assert len(data_regression.split.validation_input) == 10
    assert len(data_regression.scaled_split.train_input) == 10
    assert len(data_regression.scaled_split.validation_input) == 10
