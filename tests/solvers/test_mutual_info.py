# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import scipy.stats.qmc
import pandas as pd
import numpy as np

from gumps.solvers.mutual_info import MutualSettings, MutualInfoRegression
from gumps.scalers.log_combo_scaler import LogComboScaler

def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    input_data = scipy.stats.qmc.Sobol(4, scramble=False).random(n=1000)
    input_data = pd.DataFrame(input_data, columns=['a', 'b', 'c', 'd'])

    output_data = pd.DataFrame({'w':np.log(input_data["a"]+1e-6) * input_data['b'],
                                'x':np.exp(input_data['b']),
                                'y':input_data['c']**2,
                                'z':input_data['d'] * input_data['b']})
    return input_data, output_data

def test_MutualSettings():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data, n_neighbors=5, random_state=42)
    assert settings.input_data is input_data
    assert settings.output_data is output_data
    assert settings.n_neighbors == 5
    assert settings.random_state == 42

def test_MutualSettings_auto():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data, n_neighbors='auto')
    assert settings.input_data is input_data
    assert settings.output_data is output_data
    assert settings.n_neighbors == 'auto'
    assert settings.random_state == None

def test_MutualSettings_exception():
    input_data, output_data = get_data()
    with pytest.raises(ValueError):
        settings = MutualSettings(input_data, output_data, n_neighbors=-1)

def test_MutualSettings_exception_str():
    input_data, output_data = get_data()
    with pytest.raises(ValueError):
        settings = MutualSettings(input_data, output_data, n_neighbors='invalid')

def test_MutualInfoRegression_init():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    assert mutual_info.settings is settings
    assert mutual_info.input_scaler is not None
    assert mutual_info.output_scaler is not None
    assert mutual_info.scaled_input is not None
    assert mutual_info.scaled_output is not None
    assert mutual_info.fitted == False
    assert mutual_info.mutual_info_scaled is None

def test_MutualInfoRegression_get_scaler():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    input_scaler, output_scaler = mutual_info._get_scaler()
    assert isinstance(input_scaler, LogComboScaler)
    assert isinstance(output_scaler, LogComboScaler)

def test_MutualInfoRegression_scale_values():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    scaled_input_data, scaled_output_data = mutual_info._scale_values()
    assert isinstance(scaled_input_data, pd.DataFrame)
    assert isinstance(scaled_output_data, pd.DataFrame)
    assert scaled_input_data.shape == input_data.shape
    assert scaled_output_data.shape == output_data.shape

def test_MutualInfoRegression_fit():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    mutual_info.fit()
    assert mutual_info.fitted == True
    assert mutual_info.mutual_info_scaled is not None
    assert mutual_info.mutual_info_scaled.shape == (4, 4)
    assert mutual_info.mutual_info_scaled.columns.tolist() == ['w', 'x', 'y', 'z']
    assert mutual_info.mutual_info_scaled.index.tolist() == ['a', 'b', 'c', 'd']

def test_MutualInfo_regression__fit():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    df = mutual_info._fit(input_data, output_data)
    assert df is not None
    assert df.shape == (4, 4)
    assert df.columns.tolist() == ['w', 'x', 'y', 'z']
    assert df.index.tolist() == ['a', 'b', 'c', 'd']

def test_MutualInfoRegression_score():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    mutual_info.fit()
    df = mutual_info.score()
    assert df is not None
    assert df.shape == (4, 4)
    assert df.columns.tolist() == ['w', 'x', 'y', 'z']
    assert df.index.tolist() == ['a', 'b', 'c', 'd']

def test_MutualInfoRegression_score_exception():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    with pytest.raises(RuntimeError):
        df = mutual_info.score()

def test_MutualInfoRegression_score_scaled():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    mutual_info.fit()
    df = mutual_info.score_scaled()
    assert df is not None
    assert df.shape == (4, 4)
    assert df.columns.tolist() == ['w', 'x', 'y', 'z']
    assert df.index.tolist() == ['a', 'b', 'c', 'd']

def test_MutualInfoRegression_score_scaled_exception():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    with pytest.raises(RuntimeError):
        df = mutual_info.score_scaled()

def test_MutualInfoRegression_plot():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    mutual_info.fit()
    mutual_info.plot()
    assert True

def test_MutualInfoRegression_plot_exception():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    with pytest.raises(RuntimeError):
        mutual_info.plot()

def test_MutualInfoRegression_plot_scaled():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    mutual_info.fit()
    mutual_info.plot_scaled()
    assert True

def test_MutualInfoRegression_plot_scaled_exception():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    with pytest.raises(RuntimeError):
        mutual_info.plot_scaled()

def test_MutualInfoRegression_get_elbow_sample():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    df = mutual_info.get_elbow_sample()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 29

def test_MutualInfoRegression_get_elbow_point():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    elbow_point = mutual_info.get_elbow_point()
    assert elbow_point == 4

def test_MutualInfoRegression_update_elbow_point():
    input_data, output_data = get_data()
    settings = MutualSettings(input_data, output_data)
    mutual_info = MutualInfoRegression(settings)
    mutual_info.update_elbow_point()
    assert mutual_info.settings.n_neighbors == 4
