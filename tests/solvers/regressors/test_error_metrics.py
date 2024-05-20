# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#create tests for Error and ErrorMetrics class in error_metrics.py using pyTest

import pandas as pd
import numpy as np
from gumps.solvers.regressors.error_metrics import ErrorScalar, ErrorVector, ErrorMetrics
import tempfile
from pathlib import Path

def test_ErrorScalar():
    r_squared = 0.5
    mean_squared_error = 0.25
    root_mean_squared_error = 0.5
    normalized_mean_squared_error = 0.125
    normalized_root_mean_squared_error = 0.3535533905932738
    error = ErrorScalar(r_squared, mean_squared_error, root_mean_squared_error, normalized_mean_squared_error, normalized_root_mean_squared_error)
    assert error.r_squared == r_squared
    assert error.mean_squared_error == mean_squared_error
    assert error.root_mean_squared_error == root_mean_squared_error
    assert error.normalized_mean_squared_error == normalized_mean_squared_error
    assert error.normalized_root_mean_squared_error == normalized_root_mean_squared_error


def test_ErrorScalar_series_output():
    r_squared = 0.5
    mean_squared_error = 0.25
    root_mean_squared_error = 0.5
    normalized_mean_squared_error = 0.125
    normalized_root_mean_squared_error = 0.3535533905932738
    error = ErrorScalar(r_squared, mean_squared_error, root_mean_squared_error, normalized_mean_squared_error, normalized_root_mean_squared_error)
    pd.testing.assert_series_equal(error.series_output(),
                                   pd.Series({"score": r_squared,
                                              'r2_score': r_squared,
                                              'mean_squared_error': mean_squared_error,
                                              'mse': mean_squared_error,
                                              'root_mean_squared_error': root_mean_squared_error,
                                              'rmse': root_mean_squared_error,
                                              'normalized_mean_squared_error': normalized_mean_squared_error,
                                              'nmse': normalized_mean_squared_error,
                                              'normalized_root_mean_squared_error': normalized_root_mean_squared_error,
                                              'nrmse': normalized_root_mean_squared_error}))

def test_ErrorVector():
    r_squared = pd.Series([0.5, 0.5], index=["a", "b"])
    mean_squared_error = pd.Series([0.25, 0.25], index=["a", "b"])
    root_mean_squared_error = pd.Series([0.5, 0.5], index=["a", "b"])
    normalized_mean_squared_error = pd.Series([0.125, 0.125], index=["a", "b"])
    normalized_root_mean_squared_error = pd.Series([0.3535533905932738, 0.3535533905932738], index=["a", "b"])
    error = ErrorVector(r_squared, mean_squared_error, root_mean_squared_error, normalized_mean_squared_error, normalized_root_mean_squared_error)
    pd.testing.assert_series_equal(error.r_squared, r_squared)
    pd.testing.assert_series_equal(error.mean_squared_error, mean_squared_error)
    pd.testing.assert_series_equal(error.root_mean_squared_error, root_mean_squared_error)
    pd.testing.assert_series_equal(error.normalized_mean_squared_error, normalized_mean_squared_error)
    pd.testing.assert_series_equal(error.normalized_root_mean_squared_error, normalized_root_mean_squared_error)


def test_ErrorVector_series_output():
    r_squared = pd.Series([0.5, 0.5], index=["a", "b"])
    mean_squared_error = pd.Series([0.25, 0.25], index=["a", "b"])
    root_mean_squared_error = pd.Series([0.5, 0.5], index=["a", "b"])
    normalized_mean_squared_error = pd.Series([0.125, 0.125], index=["a", "b"])
    normalized_root_mean_squared_error = pd.Series([0.3535533905932738, 0.3535533905932738], index=["a", "b"])
    error = ErrorVector(r_squared, mean_squared_error, root_mean_squared_error, normalized_mean_squared_error, normalized_root_mean_squared_error)
    pd.testing.assert_series_equal(error.series_output(),
                                   pd.Series({"score": r_squared.to_numpy().mean(),
                                              'r2_score': r_squared.to_numpy().mean(),
                                              'mean_squared_error': mean_squared_error.to_numpy(),
                                              'mse': mean_squared_error.to_numpy(),
                                              'root_mean_squared_error': root_mean_squared_error.to_numpy(),
                                              'rmse': root_mean_squared_error.to_numpy(),
                                              'normalized_mean_squared_error': normalized_mean_squared_error.to_numpy(),
                                              'nmse': normalized_mean_squared_error.to_numpy(),
                                              'normalized_root_mean_squared_error': normalized_root_mean_squared_error.to_numpy(),
                                              'nrmse': normalized_root_mean_squared_error.to_numpy()}))


def test_ErrorVector_frame_output():
    r_squared = pd.Series([0.5, 0.5], index=["a", "b"])
    mean_squared_error = pd.Series([0.25, 0.25], index=["a", "b"])
    root_mean_squared_error = pd.Series([0.5, 0.5], index=["a", "b"])
    normalized_mean_squared_error = pd.Series([0.125, 0.125], index=["a", "b"])
    normalized_root_mean_squared_error = pd.Series([0.3535533905932738, 0.3535533905932738], index=["a", "b"])
    error = ErrorVector(r_squared, mean_squared_error, root_mean_squared_error, normalized_mean_squared_error, normalized_root_mean_squared_error)
    pd.testing.assert_frame_equal(error.frame_output(),
                                   pd.DataFrame({"score": r_squared,
                                                 'r2_score': r_squared,
                                                 'mean_squared_error': mean_squared_error,
                                                 'mse': mean_squared_error,
                                                 'root_mean_squared_error': root_mean_squared_error,
                                                 'rmse': root_mean_squared_error,
                                                 'normalized_mean_squared_error': normalized_mean_squared_error,
                                                 'nmse': normalized_mean_squared_error,
                                                 'normalized_root_mean_squared_error': normalized_root_mean_squared_error,
                                                 'nrmse': normalized_root_mean_squared_error}).T)


def test_ErrorMetrics():
    y_true = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    y_pred = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    error_metrics = ErrorMetrics(y_true, y_pred)
    assert error_metrics.error_scalar.r_squared == 1.0
    assert error_metrics.error_scalar.mean_squared_error == 0.0
    assert error_metrics.error_scalar.root_mean_squared_error == 0.0
    assert error_metrics.error_scalar.normalized_mean_squared_error == 0.0
    assert error_metrics.error_scalar.normalized_root_mean_squared_error == 0.0
    pd.testing.assert_series_equal(error_metrics.error_vector.r_squared,
                                          pd.Series([1.0, 1.0], index=["a", "b"]))
    pd.testing.assert_series_equal(error_metrics.error_vector.mean_squared_error,
                                          pd.Series([0.0, 0.0], index=["a", "b"]))
    pd.testing.assert_series_equal(error_metrics.error_vector.root_mean_squared_error,
                                          pd.Series([0.0, 0.0], index=["a", "b"]))
    pd.testing.assert_series_equal(error_metrics.error_vector.normalized_mean_squared_error,
                                          pd.Series([0.0, 0.0], index=["a", "b"]))
    pd.testing.assert_series_equal(error_metrics.error_vector.normalized_root_mean_squared_error,
                                          pd.Series([0.0, 0.0], index=["a", "b"]))

def test_ErrorMetricsNoArgs():
    error_metrics = ErrorMetrics()
    assert error_metrics.error_scalar is None
    assert error_metrics.error_vector is None

def test_ErrorMetrics_set_data():
    y_true = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    y_pred = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    error_metrics = ErrorMetrics()
    error_metrics.set_data(y_true, y_pred)
    assert error_metrics.error_scalar.r_squared == 1.0
    assert error_metrics.error_scalar.mean_squared_error == 0.0
    assert error_metrics.error_scalar.root_mean_squared_error == 0.0
    assert error_metrics.error_scalar.normalized_mean_squared_error == 0.0
    assert error_metrics.error_scalar.normalized_root_mean_squared_error == 0.0
    pd.testing.assert_series_equal(error_metrics.error_vector.r_squared,
                                          pd.Series([1.0, 1.0], index=["a", "b"]))
    pd.testing.assert_series_equal(error_metrics.error_vector.mean_squared_error,
                                          pd.Series([0.0, 0.0], index=["a", "b"]))
    pd.testing.assert_series_equal(error_metrics.error_vector.root_mean_squared_error,
                                          pd.Series([0.0, 0.0], index=["a", "b"]))
    pd.testing.assert_series_equal(error_metrics.error_vector.normalized_mean_squared_error,
                                          pd.Series([0.0, 0.0], index=["a", "b"]))
    pd.testing.assert_series_equal(error_metrics.error_vector.normalized_root_mean_squared_error,
                                          pd.Series([0.0, 0.0], index=["a", "b"]))

def test_ErrorMetrics_get_scalar_error():
    y_true = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    y_pred = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    error_metrics = ErrorMetrics()
    error = error_metrics.get_scalar_error(y_true, y_pred)
    assert error.r_squared == 1.0
    assert error.mean_squared_error == 0.0
    assert error.root_mean_squared_error == 0.0
    assert error.normalized_mean_squared_error == 0.0
    assert error.normalized_root_mean_squared_error == 0.0


def test_ErrorMetrics_get_vector_error():
    y_true = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    y_pred = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    error_metrics = ErrorMetrics()
    error = error_metrics.get_vector_error(y_true, y_pred)
    pd.testing.assert_series_equal(error.r_squared,
                                          pd.Series([1.0, 1.0], index=["a", "b"]))
    pd.testing.assert_series_equal(error.mean_squared_error,
                                          pd.Series([0.0, 0.0], index=["a", "b"]))
    pd.testing.assert_series_equal(error.root_mean_squared_error,
                                          pd.Series([0.0, 0.0], index=["a", "b"]))
    pd.testing.assert_series_equal(error.normalized_mean_squared_error,
                                          pd.Series([0.0, 0.0], index=["a", "b"]))
    pd.testing.assert_series_equal(error.normalized_root_mean_squared_error,
                                          pd.Series([0.0, 0.0], index=["a", "b"]))


def test_ErrorMetrics_save():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        y_true = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
        y_pred = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])

        error_metrics = ErrorMetrics(y_true, y_pred)
        error_metrics.save(temp_dir)

        assert (temp_dir / "error_scalar.joblib").exists()
        assert (temp_dir / "error_vector.joblib").exists()

def test_ErrorMetrics_save_no_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        error_metrics = ErrorMetrics()
        try:
            error_metrics.save(temp_dir)
        except ValueError as e:
            assert str(e) == "ErrorMetrics has not been set with data"

def test_ErrorMetrics_load():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        y_true = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
        y_pred = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])

        error_metrics_save = ErrorMetrics(y_true, y_pred)
        error_metrics_save.save(temp_dir)

        error_metrics_load = ErrorMetrics()
        error_metrics_load.load(temp_dir)

        assert error_metrics_load.error_scalar is not None
        assert error_metrics_load.error_vector is not None

        assert error_metrics_load.error_scalar.r_squared == error_metrics_save.error_scalar.r_squared
        assert error_metrics_load.error_scalar.mean_squared_error == error_metrics_save.error_scalar.mean_squared_error
        assert error_metrics_load.error_scalar.root_mean_squared_error == error_metrics_save.error_scalar.root_mean_squared_error
        assert error_metrics_load.error_scalar.normalized_mean_squared_error == error_metrics_save.error_scalar.normalized_mean_squared_error
        assert error_metrics_load.error_scalar.normalized_root_mean_squared_error == error_metrics_save.error_scalar.normalized_root_mean_squared_error

        pd.testing.assert_series_equal(error_metrics_load.error_vector.r_squared, error_metrics_save.error_vector.r_squared)
        pd.testing.assert_series_equal(error_metrics_load.error_vector.mean_squared_error, error_metrics_save.error_vector.mean_squared_error)
        pd.testing.assert_series_equal(error_metrics_load.error_vector.root_mean_squared_error, error_metrics_save.error_vector.root_mean_squared_error)
        pd.testing.assert_series_equal(error_metrics_load.error_vector.normalized_mean_squared_error, error_metrics_save.error_vector.normalized_mean_squared_error)
        pd.testing.assert_series_equal(error_metrics_load.error_vector.normalized_root_mean_squared_error, error_metrics_save.error_vector.normalized_root_mean_squared_error)

def test_ErrorMetrics_has_metrics_true():
    y_true = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    y_pred = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    error_metrics = ErrorMetrics(y_true, y_pred)
    assert error_metrics.has_metrics() == True

def test_ErrorMetrics_has_metrics_false():
    error_metrics = ErrorMetrics()
    assert error_metrics.has_metrics() == False

def test_ErrorMetrics_get_metrics_multioutput_raw_values():
    y_true = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    y_pred = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    error_metrics = ErrorMetrics(y_true, y_pred)
    pd.testing.assert_series_equal(error_metrics.get_metrics("raw_values"),
                                   pd.Series({"score": 1.0,
                                              'r2_score': 1.0,
                                              'mean_squared_error': [0.0, 0.0],
                                              'mse': [0.0, 0.0],
                                              'root_mean_squared_error': [0.0, 0.0],
                                              'rmse': [0.0, 0.0],
                                              'normalized_mean_squared_error': [0.0, 0.0],
                                              'nmse': [0.0, 0.0],
                                              'normalized_root_mean_squared_error': [0.0, 0.0],
                                              'nrmse': [0.0, 0.0]}))

def test_ErrorMetrics_get_metrics_multioutput_uniform_average():
    y_true = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    y_pred = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    error_metrics = ErrorMetrics(y_true, y_pred)
    pd.testing.assert_series_equal(error_metrics.get_metrics("uniform_average"),
                                   pd.Series({"score": 1.0,
                                              'r2_score': 1.0,
                                              'mean_squared_error': 0.0,
                                              'mse': 0.0,
                                              'root_mean_squared_error': 0.0,
                                              'rmse': 0.0,
                                              'normalized_mean_squared_error': 0.0,
                                              'nmse': 0.0,
                                              'normalized_root_mean_squared_error': 0.0,
                                              'nrmse': 0.0}))

def test_ErrorMetrics_get_metrics_multioutput_raw_values_no_data():
    error_metrics = ErrorMetrics()
    try:
        error_metrics.get_metrics("raw_values")
    except ValueError as e:
        assert str(e) == "ErrorMetrics has not been set with data"

def test_ErrorMetrics_get_metrics_multioutput_bad_input():
    y_true = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    y_pred = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    error_metrics = ErrorMetrics(y_true, y_pred)
    try:
        error_metrics.get_metrics("bad_input")
    except ValueError as e:
        assert str(e) == "multioutput must be 'raw_values' or 'uniform_average'"

def test_ErrorMetrics_get_metrics_frame():
    y_true = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    y_pred = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    error_metrics = ErrorMetrics(y_true, y_pred)
    pd.testing.assert_frame_equal(error_metrics.get_metrics_frame(),
                                   pd.DataFrame({"score": pd.Series([1.0, 1.0], index=["a", "b"]),
                                                 'r2_score': pd.Series([1.0, 1.0], index=["a", "b"]),
                                                 'mean_squared_error': pd.Series([0.0, 0.0], index=["a", "b"]),
                                                 'mse': pd.Series([0.0, 0.0], index=["a", "b"]),
                                                 'root_mean_squared_error': pd.Series([0.0, 0.0], index=["a", "b"]),
                                                 'rmse': pd.Series([0.0, 0.0], index=["a", "b"]),
                                                 'normalized_mean_squared_error': pd.Series([0.0, 0.0], index=["a", "b"]),
                                                 'nmse': pd.Series([0.0, 0.0], index=["a", "b"]),
                                                 'normalized_root_mean_squared_error': pd.Series([0.0, 0.0], index=["a", "b"]),
                                                 'nrmse': pd.Series([0.0, 0.0], index=["a", "b"])}).T)

def test_ErrorMetrics_get_metrics_frame_no_data():
    error_metrics = ErrorMetrics()
    try:
        error_metrics.get_metrics_frame()
    except ValueError as e:
        assert str(e) == "ErrorMetrics has not been set with data"

