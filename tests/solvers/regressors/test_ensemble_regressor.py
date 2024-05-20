# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#create tests for the EnsembleRegressor class using pytest

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from gumps.solvers.regressors.ensemble_regressor import EnsembleParameters, EnsembleRegressor
from gumps.solvers.regressors.trivial_regressor import TrivialLinearRegressor
from gumps.solvers.regressors.regression_solver import RegressionParameters, OptunaParameters
from gumps.scalers.scaler import AbstractScaler
from gumps.solvers.regressors.pytorch_utils.loaders import Split
from gumps.solvers.regressors.xgboost_regressor import XGBoostRegressor, XGBoostParameters

def get_input_data() -> pd.DataFrame:
    "Return input data"
    input_data = pd.DataFrame({"x": np.linspace(0, 1, 10), "y": np.linspace(1,2,10)})
    return input_data

def get_output_data() -> pd.DataFrame:
    "Return output data"
    output_data = pd.DataFrame({"z": np.linspace(2, 3, 10), 'w': np.linspace(0, 1, 10)**2})
    return output_data

def get_regressor() -> TrivialLinearRegressor:
    "Return regressor"
    parameters = RegressionParameters(input_data=get_input_data(), output_data=get_output_data())
    regressor = TrivialLinearRegressor(parameters)
    return regressor

def test_EnsembleParameters_number_regressors():
    with pytest.raises(ValueError):
        EnsembleParameters(regressor=get_regressor(), number_regressors=0, input_data=get_input_data(), output_data=get_output_data())

    with pytest.raises(ValueError):
        EnsembleParameters(regressor=get_regressor(), number_regressors=1.1, input_data=get_input_data(), output_data=get_output_data())

    with pytest.raises(ValueError):
        EnsembleParameters(regressor=get_regressor(), number_regressors=-1, input_data=get_input_data(), output_data=get_output_data())

def test_EnsembleParameters_invalid_validation_fraction():
    with pytest.raises(ValueError):
        EnsembleParameters(regressor=get_regressor(), number_regressors=1, validation_fraction=0, input_data=get_input_data(), output_data=get_output_data())

    with pytest.raises(ValueError):
        EnsembleParameters(regressor=get_regressor(), number_regressors=1, validation_fraction=1, input_data=get_input_data(), output_data=get_output_data())

    with pytest.raises(ValueError):
        EnsembleParameters(regressor=get_regressor(), number_regressors=1, validation_fraction=1.1, input_data=get_input_data(), output_data=get_output_data())

    with pytest.raises(ValueError):
        EnsembleParameters(regressor=get_regressor(), number_regressors=1, validation_fraction=-0.1, input_data=get_input_data(), output_data=get_output_data())

def test_EnsembleParameters():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=1, input_data=get_input_data(), output_data=get_output_data())
    assert isinstance(parameters.regressor, TrivialLinearRegressor)
    assert parameters.number_regressors == 1
    assert parameters.validation_fraction == 0.1
    assert parameters.random_state is None
    assert parameters.weight_ensemble == True

def test_EnsembleRegressor():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=1, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)
    assert isinstance(regressor, EnsembleRegressor)
    assert isinstance(regressor.parameters, EnsembleParameters)
    assert isinstance(regressor.splits, list)
    assert isinstance(regressor.regressors, list)
    assert regressor.weights is None
    assert isinstance(regressor._get_scalers()[0], AbstractScaler)
    assert isinstance(regressor._get_scalers()[1], AbstractScaler)
    assert isinstance(regressor._get_regressor(), TrivialLinearRegressor)
    assert isinstance(regressor._create_splits()[0], Split)
    assert isinstance(regressor._create_regressor()[0], TrivialLinearRegressor)

def test_EnsembleRegressor_get_scalers():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=1, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)
    assert isinstance(regressor._get_scalers()[0], AbstractScaler)
    assert isinstance(regressor._get_scalers()[1], AbstractScaler)

def test_EnsembleRegressor_get_regressor():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=1, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)
    assert isinstance(regressor._get_regressor(), TrivialLinearRegressor)

def test_EnsembleRegressor_create_splits():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)
    assert isinstance(regressor._create_splits()[0], Split)
    assert len(regressor._create_splits()[0].train_input) == 9
    assert len(regressor._create_splits()[0].validation_input) == 1
    assert len(regressor._create_splits()) == 4

def test_EnsembleRegressor_create_regressor():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)
    assert isinstance(regressor._create_regressor()[0], TrivialLinearRegressor)
    assert len(regressor._create_regressor()) == 4

def test_EnsembleRegressor_fit():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)
    regressor.fit()
    assert isinstance(regressor.weights, pd.DataFrame)
    assert len(regressor.weights) == 4
    pd.testing.assert_series_equal(regressor.weights.sum(), pd.Series({'z':1.0, 'w':1.0}))

def test_Ensemble_create_error():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)
    regressor.fit()
    error = regressor._create_error()
    assert isinstance(error, pd.DataFrame)
    assert len(error) == 4
    assert len(error.columns) == 2
    assert 'z' in error.columns
    assert 'w' in error.columns

def test_Ensemble_create_weights_error():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)
    regressor.fit()
    weights = regressor._create_weights_error()
    assert isinstance(weights, pd.DataFrame)
    assert len(weights) == 4
    assert len(weights.columns) == 2
    assert 'z' in weights.columns
    assert 'w' in weights.columns
    pd.testing.assert_series_equal(weights.sum(), pd.Series({'z':1.0, 'w':1.0}))

def test_Ensemble_create_weights_uniform():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)
    regressor.fit()
    weights = regressor._create_weights_uniform()
    assert isinstance(weights, pd.DataFrame)
    assert len(weights) == 4
    assert len(weights.columns) == 2
    assert 'z' in weights.columns
    assert 'w' in weights.columns
    pd.testing.assert_series_equal(weights.sum(), pd.Series({'z':1.0, 'w':1.0}))

def test_Ensemble_create_weights_weight_ensemble_true():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)
    regressor.fit()
    weights = regressor._create_weights()
    assert isinstance(weights, pd.DataFrame)
    assert len(weights) == 4
    assert len(weights.columns) == 2
    assert 'z' in weights.columns
    assert 'w' in weights.columns
    pd.testing.assert_series_equal(weights.sum(), pd.Series({'z':1.0, 'w':1.0}))

def test_Ensemble_create_weights_weight_ensemble_false():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=get_input_data(), output_data=get_output_data(), weight_ensemble=False)
    regressor = EnsembleRegressor(parameters)
    regressor.fit()
    weights = regressor._create_weights()
    assert isinstance(weights, pd.DataFrame)
    assert len(weights) == 4
    assert len(weights.columns) == 2
    assert 'z' in weights.columns
    assert 'w' in weights.columns
    pd.testing.assert_series_equal(weights.sum(), pd.Series({'z':1.0, 'w':1.0}))

def test_Ensemble_clone():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)
    clone = regressor.clone(parameters)
    assert isinstance(clone, EnsembleRegressor)
    assert isinstance(clone.parameters, EnsembleParameters)
    assert isinstance(clone.splits, list)
    assert isinstance(clone.regressors, list)
    assert clone.weights is None
    assert isinstance(clone._get_scalers()[0], AbstractScaler)
    assert isinstance(clone._get_scalers()[1], AbstractScaler)
    assert isinstance(clone._get_regressor(), TrivialLinearRegressor)
    assert isinstance(clone._create_splits()[0], Split)
    assert isinstance(clone._create_regressor()[0], TrivialLinearRegressor)
    assert clone.parameters == parameters

def test_Ensemble_predict():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)
    regressor.fit()
    prediction = regressor.predict(get_input_data())
    assert isinstance(prediction, pd.DataFrame)
    assert len(prediction) == 10
    assert len(prediction.columns) == 2
    assert 'z' in prediction.columns
    assert 'w' in prediction.columns

def test_Ensemble_predict_weights_enabled_false():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=get_input_data(), output_data=get_output_data(), weight_ensemble=False)
    regressor = EnsembleRegressor(parameters)
    regressor.fit()
    prediction = regressor.predict(get_input_data())
    assert isinstance(prediction, pd.DataFrame)
    assert len(prediction) == 10
    assert len(prediction.columns) == 2
    assert 'z' in prediction.columns
    assert 'w' in prediction.columns

def test_Ensemble_predict_with_uncertainty():
    input_data = get_input_data()
    output_data = get_output_data()
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=input_data, output_data=output_data)
    regressor = EnsembleRegressor(parameters)
    regressor.fit()
    prediction = regressor.predict_with_uncertainty(input_data)
    assert isinstance(prediction.prediction, pd.DataFrame)
    assert isinstance(prediction.minimum, pd.DataFrame)
    assert isinstance(prediction.maximum, pd.DataFrame)
    assert isinstance(prediction.standard_deviation, pd.DataFrame)
    assert prediction.prediction.shape == output_data.shape
    assert prediction.minimum.shape == output_data.shape
    assert prediction.maximum.shape == output_data.shape
    assert prediction.standard_deviation.shape == output_data.shape
    assert (prediction.prediction >= prediction.minimum).all().all()
    assert (prediction.prediction <= prediction.maximum).all().all()
    assert (prediction.minimum <= prediction.maximum).all().all()

def test_Ensemble_update_data():
    input_data = get_input_data()
    output_data = get_output_data()
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=input_data, output_data=output_data)
    regressor = EnsembleRegressor(parameters)
    regressor.fit()

    new_input_data = pd.DataFrame({"x": np.linspace(2, 3, 10), "y": np.linspace(3,4,10)})
    new_output_data = pd.DataFrame({"z": np.linspace(4, 5, 10), 'w': np.linspace(2, 3, 10)**2})

    regressor.update_data(new_input_data, new_output_data)
    assert len(regressor.parameters.input_data) == 20
    assert len(regressor.parameters.output_data) == 20

    for reg in regressor.regressors:
        assert len(reg.data_regression.split.full_input) == 20
        assert len(reg.data_regression.split.full_output) == 20

def test_Ensemble_save():
    input_data = get_input_data()
    output_data = get_output_data()
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=input_data, output_data=output_data)
    regressor = EnsembleRegressor(parameters)
    regressor.fit()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        regressor.save(temp_dir)

        assert (temp_dir / "data").exists()
        assert (temp_dir / "error_scalar.joblib").exists()
        assert (temp_dir / "error_vector.joblib").exists()
        assert (temp_dir / "parameters.joblib").exists()
        assert (temp_dir / "splits.joblib").exists()
        assert (temp_dir / "regressor").exists()
        assert (temp_dir / "weights.joblib").exists()
        assert (temp_dir / "regressor.txt").exists()

        for i in range(4):
            assert (temp_dir / "regressors" / f"{i}").exists()

def test_Ensemble_save_not_fitted_exception():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        with pytest.raises(RuntimeError):
            regressor.save(temp_dir)

def test_Ensemble_load():
    input_data = get_input_data()
    output_data = get_output_data()
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=input_data, output_data=output_data)
    regressor = EnsembleRegressor(parameters)
    regressor.fit()

    before_save = regressor.predict(input_data)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        regressor.save(temp_dir)

        new_regressor = EnsembleRegressor.load(temp_dir)

        after_load = new_regressor.predict(input_data)
        pd.testing.assert_frame_equal(before_save, after_load)
        assert isinstance(new_regressor, EnsembleRegressor)
        assert isinstance(new_regressor.parameters, EnsembleParameters)
        assert isinstance(new_regressor.splits, list)
        assert isinstance(new_regressor.regressors, list)
        assert isinstance(new_regressor.weights, pd.DataFrame)

def test_Ensemble_auto_tune():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)

    with pytest.raises(NotImplementedError):
        regressor.auto_tune(None)

def test_Ensemble_auto_tune_xgboost():
    "this should run without generating an error"
    input_data = get_input_data()
    output_data = get_output_data()

    parameters = XGBoostParameters(input_data=input_data, output_data=output_data)
    internal_regressor = XGBoostRegressor(parameters)

    parameters = EnsembleParameters(regressor=internal_regressor, number_regressors=4, input_data=input_data, output_data=output_data)
    regressor = EnsembleRegressor(parameters)

    settings = OptunaParameters(number_of_trials=5)

    regressor.auto_tune(settings)

def test_Ensemble_get_tuned_parameters():
    parameters = EnsembleParameters(regressor=get_regressor(), number_regressors=4, input_data=get_input_data(), output_data=get_output_data())
    regressor = EnsembleRegressor(parameters)

    with pytest.raises(NotImplementedError):
        regressor.get_tuned_parameters()

def test_Ensemble_get_tuned_parameters_xgboost():
    input_data = get_input_data()
    output_data = get_output_data()

    parameters = XGBoostParameters(input_data=input_data, output_data=output_data)
    internal_regressor = XGBoostRegressor(parameters)

    parameters = EnsembleParameters(regressor=internal_regressor, number_regressors=4, input_data=input_data, output_data=output_data)
    regressor = EnsembleRegressor(parameters)

    assert isinstance(regressor.get_tuned_parameters(), dict)
