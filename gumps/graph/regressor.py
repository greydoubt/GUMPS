# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create response plots for a regressor. This requires an already fitted regressor, a starting point, and a search range.
This should be used with caution for high dimensional problems as it shows a 2d slice of the search space for each feature.

It is importants that GUMPS use the matplotlib object oriented interface to create plots.
The plt interface is stateful and keeps plot references in memory. This can cause memory leaks and other issues."""

import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import attrs
import gumps.solvers.regressors.regression_solver
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@attrs.define
class RegressorPlotParameters:
    regressor: gumps.solvers.regressors.regression_solver.AbstractRegressor
    start: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    points: int = 100
    font_size: int = 20
    row_width: int = 6
    column_height: int = 6

@attrs.define
class FigureData:
    input_data: pd.DataFrame
    output_data: pd.DataFrame
    predicted_output_data: pd.DataFrame
    response_input_data: dict[str, pd.DataFrame]
    response_output_data: dict[str, pd.DataFrame]

    def get_response_data(self, feature: str) -> dict[str, pd.DataFrame]:
        "get the response data for a given feature"
        response = {}
        for key in self.response_input_data:
            x = self.response_input_data[key][key]
            y = self.response_output_data[key][feature]
            response[key] = pd.DataFrame({'x':x, 'y':y})
        return response

class RegressorPlot:

    def __init__(self, params: RegressorPlotParameters):
        self.params = params

    def get_plot(self) -> matplotlib.figure.Figure:
        "get the plot as a matplotlib figure"
        input_data, output_data, predicted_output_data = self._get_prediction_data()
        response_input_data, response_output_data = self._get_response_data()
        data = FigureData(input_data, output_data, predicted_output_data, response_input_data, response_output_data)
        fig = self._create_figure(data)
        return fig

    def get_plot_scaled(self) -> matplotlib.figure.Figure:
        "get the plot as a matplotlib figure"
        input_data, output_data, predicted_output_data = self._get_prediction_data_scaled()
        response_input_data, response_output_data = self._get_response_data_scaled()
        data = FigureData(input_data, output_data, predicted_output_data, response_input_data, response_output_data)
        fig = self._create_figure(data)
        return fig

    def plot(self) -> None:
        fig = self.get_plot()
        plt.show()
        plt.close(fig)

    def plot_scaled(self) -> None:
        fig = self.get_plot_scaled()
        plt.show()
        plt.close(fig)

    def save_plot(self, path:Path):
        "save the plot"
        fig = self.get_plot()
        fig.savefig(path)
        plt.close(fig)

    def save_scaled_plot(self, path:Path):
        "save the plot"
        fig = self.get_plot_scaled()
        fig.savefig(path)
        plt.close(fig)

    def _get_subplot_size(self) -> dict[str, int]:
        "get the number of rows and columns for the figure, we need 1 figure per input variable and 1 more for the prediction plot"
        return {'nrows':len(self.params.regressor.parameters.output_data.columns), 'ncols':1 + len(self.params.start)}

    def _create_figure(self, data:FigureData) -> matplotlib.figure.Figure:
        subplot_size = self._get_subplot_size()
        width = self.params.row_width * subplot_size['ncols']
        height = self.params.column_height * subplot_size['nrows']
        fig, axes = plt.subplots(**subplot_size, figsize=[width, height], squeeze=False)

        for idx, response_label in enumerate(self.params.regressor.parameters.output_data.columns):
            ax_left = axes[idx, 0]
            axs_right = axes[idx, 1:]
            self._create_prediction_subplot(ax_left, response_label, data.output_data, data.predicted_output_data)
            self._create_response_subplot(axs_right, response_label, data.get_response_data(response_label))
        fig.tight_layout()
        return fig

    def _get_prediction_data(self):
        input_data = self.params.regressor.parameters.input_data
        output_data = self.params.regressor.parameters.output_data
        predicted_output_data = self.params.regressor.predict(input_data)
        return input_data, output_data, predicted_output_data

    def _get_prediction_data_scaled(self):
        input_data_scaled = self.params.regressor.data_regression.scaled_split.full_input
        output_data_scaled = self.params.regressor.data_regression.scaled_split.full_output

        _, _, predicted_output_data = self._get_prediction_data()

        predicted_output_data_scaled = self.params.regressor.data_regression.output_scaler.transform(predicted_output_data)
        predicted_output_data_scaled = pd.DataFrame(predicted_output_data_scaled, columns=output_data_scaled.columns)
        return input_data_scaled, output_data_scaled, predicted_output_data_scaled

    def _scale_input(self, input_data:pd.DataFrame) -> pd.DataFrame:
        "scale the input data"
        return self.params.regressor.data_regression.input_scaler.transform(input_data)

    def _scale_output(self, output_data:pd.DataFrame) -> pd.DataFrame:
        "scale the input data"
        return self.params.regressor.data_regression.output_scaler.transform(output_data)

    def _get_response_data_scaled(self) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        "get the response data for each input variable, one simulation needs to be run per input variable in order to simplify splitting"
        response_data_input, response_data_output = self._get_response_data()
        for label in response_data_input:
            response_data_input[label] = self._scale_input(response_data_input[label])
            response_data_output[label] = self._scale_output(response_data_output[label])
        return response_data_input, response_data_output

    def _get_response_data(self) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        "get the response data for each input variable, one simulation needs to be run per input variable in order to simplify splitting"
        response_data_input = {}
        response_data_output = {}
        for label in self.params.start.keys():
            input_data, output_data = self._get_response_data_for_label(label)
            response_data_input[label] = input_data
            response_data_output[label] = output_data
        return response_data_input, response_data_output

    def _get_response_data_for_label(self, label:str) -> tuple[pd.DataFrame, pd.DataFrame]:
        "get the response data for a single input variable"
        param_data = {}
        for param in self.params.start.keys():
            if param == label:
                param_data[param] = np.linspace(self.params.lower_bound[param], self.params.upper_bound[param], self.params.points)
            else:
                param_data[param] = np.repeat(self.params.start[param], self.params.points)
        input_data = pd.DataFrame(param_data)
        output_data = self.params.regressor.predict(input_data)
        return input_data, output_data

    def _set_font_size(self, ax) -> None:
        "set the font size for the figure"
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(self.params.font_size)

    def _create_prediction_subplot(self, ax, response_label:str, output_data:pd.DataFrame, predicted_output_data:pd.DataFrame) -> None:
        ax.scatter(predicted_output_data[response_label], output_data[response_label])
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Data")
        ax.set_title(f"Prediction vs Data for {response_label}")

        min_value = min(predicted_output_data[response_label].min(), output_data[response_label].min())
        max_value = max(predicted_output_data[response_label].max(), output_data[response_label].max())
        ax.plot([min_value, max_value], [min_value, max_value], color='red')
        ax.set_xlim(min_value, max_value)
        ax.set_ylim(min_value, max_value)
        self._set_font_size(ax)

    def _create_response_subplot(self, axs, response_label:str, response_data:dict[str, pd.DataFrame]) -> None:
        for idx, param in enumerate(response_data):
            ax = axs[idx]
            ax.plot(response_data[param].x, response_data[param].y)
            ax.set_xlabel(param)
            ax.set_ylabel(response_label)
            ax.set_title(f"Response for {response_label} vs {param}")
            self._set_font_size(ax)
