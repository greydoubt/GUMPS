# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#create a response surface app that uses the response sampler and then has plotting routines to visualize the data

import logging
from pathlib import Path
from typing import Callable, Iterator

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import seaborn as sns
from matplotlib.figure import Figure

from gumps.common.hdf5 import H5
from gumps.common.app_utils import run_batch_iterator
from gumps.common.parallel import AbstractParallelPool
from gumps.solvers.response_sampler import (ResponseSampler,
                                            ResponseSamplerParameters)
from gumps.studies.batch_study import AbstractBatchStudy

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ResponseSurface:
    "Create the response surface app"

    def __init__(self, *, parameters: ResponseSamplerParameters,
                 batch: AbstractBatchStudy,
                 parallel: AbstractParallelPool,
                 directory: Path | None=None,
                 processing_function: Callable | None,
                 pre_processing_function : Callable | None=None):
        "initialize the response surface app"
        self.parameters = parameters
        self.solver = ResponseSampler(solver_settings=self.parameters)
        self.directory = directory
        self.parallel = parallel

        self.batch = batch
        self.processing_function = processing_function
        self.pre_processing_function = pre_processing_function


    def save(self):
        "save the response surface app"
        if self.directory is None:
            raise ValueError("directory must be set to save results")

        if self.solver.request is None or self.solver.response is None:
            raise ValueError("Response Surface must be run before saving")

        filename = self.directory / "response_surface.h5"

        data = H5(filename=filename.as_posix())

        data.root.lower_bound = self.parameters.lower_bound.to_numpy()
        data.root.lower_bound_names = self.parameters.lower_bound.index.to_numpy()

        data.root.upper_bound = self.parameters.upper_bound.to_numpy()
        data.root.upper_bound_names = self.parameters.upper_bound.index.to_numpy()

        data.root.baseline = self.parameters.baseline.to_numpy()
        data.root.baseline_names = self.parameters.baseline.index.to_numpy()

        data.root.points_1d = self.parameters.points_1d
        data.root.points_2d_per_dimension = self.parameters.points_2d_per_dimension

        if self.directory is not None:
            data.root.directory = self.directory.as_posix()

        if self.solver.response is not None:
            data.root.response = self.solver.response.to_numpy()
            data.root.response_names = self.solver.response.columns

        if self.solver.request is not None:
            data.root.request = self.solver.request.to_numpy()
            data.root.request_names = self.solver.request.columns

        data.save()


    @classmethod
    def load(cls, *, directory: Path, parallel: AbstractParallelPool):
        "load the response surface app"

        filename = directory / "response_surface.h5"

        data = H5(filename=filename.as_posix())
        data.load()

        lower_bound = pd.Series(data.root.lower_bound, index=data.root.lower_bound_names)
        upper_bound = pd.Series(data.root.upper_bound, index=data.root.upper_bound_names)
        baseline = pd.Series(data.root.baseline, index=data.root.baseline_names)
        points_1d = data.root.points_1d
        points_2d_per_dimension = data.root.points_2d_per_dimension

        directory = Path(data.root.directory)

        parameters = ResponseSamplerParameters(lower_bound=lower_bound,
                                                  upper_bound=upper_bound,
                                                  baseline=baseline,
                                                  points_1d=points_1d,
                                                  points_2d_per_dimension=points_2d_per_dimension)

        response = pd.DataFrame(data.root.response, columns=data.root.response_names.astype(str).tolist())

        request = pd.DataFrame(data.root.request, columns=data.root.request_names.astype(str).tolist())

        instance = cls.__new__(cls)
        instance.parameters = parameters
        instance.solver = ResponseSampler(solver_settings=parameters)
        instance.directory = directory
        instance.parallel = parallel
        instance.batch = None
        instance.processing_function = None
        instance.pre_processing_function = None

        instance.solver.response = response
        instance.solver.request = request

        return instance


    def run(self):
        "run the response surface app"

        with self.batch:
            while self.solver.has_next():
                request = self.solver.ask()

                request, output_data = run_batch_iterator(self.batch, request, self.processing_function, self.pre_processing_function)

                self.solver.tell(output_data, request)


    def request_variables(self) -> list[str]:
        "get the request variables"
        return list(self.solver.request.columns)


    def response_variables(self) -> list[str]:
        "get the response variables"
        return list(self.solver.response.columns)


    def request(self) -> pd.DataFrame:
        "get the request"
        return self.solver.request


    def response(self) -> pd.DataFrame:
        "get the response"
        return self.solver.response


    def split_response(self)-> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        "split the variables"
        return self.solver.split_response()

    @staticmethod
    def clean_nan_values(df: pd.DataFrame) -> pd.DataFrame:
        "clean the nan values and log any removed values"
        if df.isna().any().any():
            nan_rows = df[df.isna().any(axis=1)].index.tolist()
            logger.warning(f"NaN values found in data, removing rows {nan_rows}")

        new = df.dropna().reset_index(drop=True)

        if new.empty:
            raise ValueError("All data is NaN")
        return new

    @staticmethod
    def _generate_1d_plot(x: tuple[str, pd.DataFrame, str, float, float]) -> tuple[str, Figure]:
        "designed to be called from a parallel pool"
        key, data, response_name, lower_bound, upper_bound = x

        fig, ax = plt.subplots()

        sns.lineplot(data=data, x=key, y=response_name, ax=ax)
        ax.set_title(f'{key} vs {response_name}')
        ax.set_xlabel(key)
        ax.set_ylabel(response_name)
        ax.set_xlim(lower_bound, upper_bound)

        ax.legend([response_name])

        fig.tight_layout()

        return key, fig


    def get_1d_plots(self, response_name: str) -> Iterator[tuple[str, Figure]]:
        "get the 1d plot"
        with self.parallel as map_function:
            yield from self._get_1d_plots(response_name=response_name, map_function=map_function)

    def _get_1d_plots(self, response_name:str, map_function:Callable) -> Iterator[tuple[str, Figure]]:
        "internal version of get_1d_plots that takes a map function to simplify get_all versions"
        data_1d = self.solver.get_response_1d()

        data_1d = {key: self.clean_nan_values(data) for key, data in data_1d.items()}

        lb = self.parameters.lower_bound
        ub = self.parameters.upper_bound

        runs = ((key, data, response_name, lb[key], ub[key]) for key, data in data_1d.items())

        for key, fig in map_function(self._generate_1d_plot, runs):
            yield key, fig


    def show_1d_plots(self, response_name: str):
        """This is intended to be used in a jupyter notebook and that has problems with
        generators creating plots. So convert to a list first and then show the plots.
        If you have a simulation that creates many plots this could require a LOT of
        memory. So be careful."""
        for _, figure in list(self.get_1d_plots(response_name=response_name)):
            plt.show()
            plt.close(figure)


    def save_1d_plots(self, response_name: str):
        "save the 1d plots"
        with self.parallel as map_function:
            self._save_1d_plots(response_name=response_name, map_function=map_function)


    def _save_1d_plots(self, response_name:str, map_function:Callable):
        "internal version of save_1d_plots that takes a map function to simplify save_all versions"
        if self.directory is None:
            raise ValueError("directory must be set to save plots")

        save_dir = self.directory / response_name / "1D"

        save_dir.mkdir(parents=True, exist_ok=True)

        for key, figure in self._get_1d_plots(response_name=response_name, map_function=map_function):
            figure.savefig(save_dir / f"1D_{key}_{response_name}.png")
            plt.close(figure)


    def save_all_1d_plots(self):
        "save all 1d plots"
        with self.parallel as map_function:
            for response_name in self.response_variables():
                self._save_1d_plots(response_name=response_name, map_function=map_function)


    def get_2d_bounds(self, key:str) -> tuple[float, float, float, float]:
        "get the 2d bounds"
        lower_bound = self.parameters.lower_bound
        upper_bound = self.parameters.upper_bound

        lb_x = lower_bound[key.split('___')[0]]
        ub_x = upper_bound[key.split('___')[0]]

        lb_y = lower_bound[key.split('___')[1]]
        ub_y = upper_bound[key.split('___')[1]]

        return lb_x, ub_x, lb_y, ub_y


    @staticmethod
    def get_scale(data: pd.Series) -> tuple[str, matplotlib.colors.Normalize | matplotlib.colors.LogNorm]:
        "get the scale"
        min_data = data.min()
        max_data = data.max()

        if min_data > 0 and max_data/min_data > 100:
            return "log", matplotlib.colors.LogNorm(vmin=min_data,
                                                    vmax=max_data)
        else:
            return "linear", matplotlib.colors.Normalize(vmin=min_data,
                                                         vmax=max_data)


    @staticmethod
    def _generate_2d_plot(x) -> tuple[str, Figure]:
        "designed to be called from a parallel pool"
        key, data, response_name, lb_x, ub_x, lb_y, ub_y, show_points = x

        fig, ax = plt.subplots()

        x = np.linspace(lb_x, ub_x, 100)
        y = np.linspace(lb_y, ub_y, 100)

        x_name, y_name = key.split('___')

        X, Y = np.meshgrid(x, y)
        Z = scipy.interpolate.griddata((data[x_name], data[y_name]), data[response_name], (X, Y), method='cubic')

        scale_name, scaler = ResponseSurface.get_scale(data[response_name])

        im = ax.imshow(Z, extent=[lb_x, ub_x, lb_y, ub_y],
                    origin='lower', norm=scaler, aspect='auto')

        if show_points:
            ax.scatter(data[x_name], data[y_name], marker='.', c='black', s=1)

        ax.set_title(f'{x_name} vs {y_name} vs {response_name} {scale_name} scaling')
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

        fig.colorbar(im, ax=ax)

        ax.set_xlim(lb_x, ub_x)
        ax.set_ylim(lb_y, ub_y)

        fig.tight_layout()

        return key, fig

    def get_2d_plots(self, response_name:str, show_points=True) -> Iterator[tuple[str, Figure]]:
        "get the 2d plots"
        with self.parallel as map_function:
            yield from self._get_2d_plots(response_name=response_name, map_function=map_function, show_points=show_points)

    def _get_2d_plots(self, response_name:str, map_function:Callable, show_points=True) -> Iterator[tuple[str, Figure]]:
        "internal version of get_2d_plots that takes a map function to simplify get_all versions"
        data_2d = self.solver.get_response_2d()

        data_2d = {key: self.clean_nan_values(data) for key, data in data_2d.items()}

        runs = ((key, data, response_name, *self.get_2d_bounds(key), show_points) for key, data in data_2d.items())

        for key, fig in map_function(self._generate_2d_plot, runs):
            yield key, fig


    def show_2d_plots(self, response_name:str, show_points=True):
        """This is intended to be used in a jupyter notebook and that has problems with
        generators creating plots. So convert to a list first and then show the plots.
        If you have a simulation that creates many plots this could require a LOT of
        memory. So be careful."""
        for _, figure in list(self.get_2d_plots(response_name=response_name, show_points=show_points)):
            plt.show()
            plt.close(figure)


    def save_2d_plots(self, response_name:str):
        "save the 2d plots"
        with self.parallel as map_function:
            self._save_2d_plots(response_name=response_name, map_function=map_function)


    def _save_2d_plots(self, response_name:str, map_function:Callable):
        "internal version of save_2d_plots that takes a map function to simplify save_all versions"
        if self.directory is None:
            raise ValueError("directory must be set to save plots")

        save_dir = self.directory / response_name / '2D'

        save_dir.mkdir(parents=True, exist_ok=True)

        for key, figure in self._get_2d_plots(response_name=response_name, map_function=map_function):
            figure.savefig(save_dir / f"2D_{key}_{response_name}.png")
            plt.close(figure)


    def save_all_2d_plots(self):
        "save all 2d plots, for efficient parallelization this can't just call save_2d_plots"
        with self.parallel as map_function:
            for response_name in self.response_variables():
                self._save_2d_plots(response_name=response_name, map_function=map_function)


    def save_all_plots(self):
        "save all plots"
        self.save_all_1d_plots()
        self.save_all_2d_plots()
        self.save_all_tornado_plots()


    def save_all_response_plot(self, response_name:str):
        "save all response plots"
        self.save_1d_plots(response_name=response_name)
        self.save_2d_plots(response_name=response_name)
        self.save_tornado_plot(response_name=response_name)


    @staticmethod
    def get_data_bounds(data_1d: dict[str, pd.DataFrame], response_name:str) -> tuple[float, float]:
        "get the upper and lower bounds of the data set"
        max_values = [data[response_name].max() for data in data_1d.values()]
        min_values = [data[response_name].min() for data in data_1d.values()]

        min_value = min(min_values)
        max_value = max(max_values)
        return min_value, max_value


    def rescale_data(self, response_name:str) -> tuple[dict[str, pd.DataFrame], str]:
        "rescale the data"
        data_1d = self.solver.get_response_1d()

        data_1d = {key: self.clean_nan_values(data) for key, data in data_1d.items()}

        baseline = self.solver.get_baseline_response()

        min_value, max_value = self.get_data_bounds(data_1d=data_1d, response_name=response_name)

        mean_value = baseline[response_name][0]


        if min_value > 0:
            scale_name = f"Percent Change from {mean_value:.2e} for {response_name}"
            name_scaler = scipy.interpolate.interp1d([min_value, mean_value, max_value], [min_value/mean_value, 1, max_value/mean_value])
        else:
            scale_name = f"Absolute Change from {mean_value:.2e} for {response_name} with Linear scaling"
            name_scaler = scipy.interpolate.interp1d([min_value, mean_value, max_value], [-1, 0, 1])

        data_1d_rescale = {}

        lower_bound = self.parameters.lower_bound
        upper_bound = self.parameters.upper_bound
        baseline = self.parameters.baseline

        for key, data in data_1d.items():
            new_df = {}

            interp = scipy.interpolate.interp1d([lower_bound[key], baseline[key], upper_bound[key]], [0, 0.5, 1])

            new_df[key] = interp(data[key])
            new_df[response_name] = name_scaler(data[response_name])

            data_1d_rescale[key] = pd.DataFrame(new_df)
        return data_1d_rescale, scale_name


    @staticmethod
    def gradient_bar(ax, data:pd.DataFrame, x_column:str, color_column:str, vertical_position:float, height=0.5):
        "generate a gradient bar"
        ax.scatter(data[x_column], data[color_column] * height + vertical_position - height/2,
                   marker='.', c=data[color_column], s=4,
                   cmap=matplotlib.colormaps['RdYlGn'])
        ax.axhline(vertical_position, color='black', linewidth=0.5)


    @staticmethod
    def get_bar_order(data_1d: dict[str, pd.DataFrame], response_name:str) -> list[str]:
        "get the bar order"
        widths = []
        for key, data in data_1d.items():
            widths.append( (data[response_name].max() - data[response_name].min(), key) )
        widths.sort()
        return [key for _, key in widths]


    def get_tornado_plot(self, response_name:str) -> plt.Figure:
        "get the tornado plot"
        data_1d_rescale, rescale_name = self.rescale_data(response_name=response_name)

        xlim = self.get_data_bounds(data_1d_rescale, response_name=response_name)

        key_order = self.get_bar_order(data_1d_rescale, response_name=response_name)

        max_height = 0.1 * len(key_order) + 0.1

        fig, ax = plt.subplots(figsize=[5, int(0.15 * len(key_order))+4])
        ax.set(xlim=xlim, ylim=(0, max_height))

        for idx, key in enumerate(key_order):
            self.gradient_bar(ax, data_1d_rescale[key], response_name, key, idx*0.1 + 0.1, height=0.05)

        ax.set_yticks([idx*0.1 + 0.1 for idx in range(len(key_order))], labels=key_order)

        ax.set_title(f"Response Plot for {response_name}")
        ax.set_xlabel(rescale_name)
        ax.set_ylabel("Parameter")

        # add colorbar
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=matplotlib.colormaps['RdYlGn']),
                            ticks=[0, 0.5, 1],
                            ax=ax, orientation='horizontal', pad=0.2)
        cbar.ax.set_xticklabels(['Low', 'Baseline', 'High'])

        fig.tight_layout()

        return fig


    def show_tornado_plot(self, response_name:str):
        "show the tornado plot"
        figure = self.get_tornado_plot(response_name=response_name)

        plt.show()
        plt.close(figure)


    def save_tornado_plot(self, response_name:str):
        "save the tornado plot"
        if self.directory is None:
            raise ValueError("directory must be set to save plots")

        save_dir = self.directory / "tornado"

        save_dir.mkdir(parents=True, exist_ok=True)

        figure = self.get_tornado_plot(response_name=response_name)
        figure.savefig(save_dir / f"tornado_{response_name}.png")
        plt.close(figure)


    def save_all_tornado_plots(self):
        "save all tornado plots"
        for response_name in self.response_variables():
            self.save_tornado_plot(response_name=response_name)
