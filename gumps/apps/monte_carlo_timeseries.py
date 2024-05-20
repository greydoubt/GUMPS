# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Time series version of the monte carlo app for uncertainty quantification"

import itertools
from typing import Callable, Optional

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use('AGG')

import logging
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt

import gumps.solvers.monte_carlo_timeseries_solver
from gumps.common import hdf5
from gumps.common.app_utils import run_batch_time_iterator
from gumps.studies.batch_time_study import AbstractBatchTimeStudy

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MonteCarloTimeSeriesApp:
    "create a monte carlo app"

    def __init__(self, *, parameters: gumps.solvers.monte_carlo_timeseries_solver.MonteCarloParameters,
        processing_function: Callable, directory:Optional[Path],
        batch: AbstractBatchTimeStudy, pre_processing_function: Callable|None=None):
        "initialize the monte carlo app"
        self.parameters = parameters
        self.processing_function = processing_function
        self.pre_processing_function = pre_processing_function
        self.batch = batch
        self.next_index:int = 0

        self.chain: Optional[pd.DataFrame] = None
        self.scores: Optional[xr.Dataset] = None

        self.directory = directory
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def update_dataset(data: Optional[xr.Dataset], new_data: xr.Dataset):
        "update data with new_data"
        if data is None:
            return new_data
        else:
            return xr.concat([data, new_data], dim="index")

    @staticmethod
    def update_dataframe(data: Optional[pd.DataFrame], new_data: pd.DataFrame):
        "update data with new_data"
        if data is None:
            return new_data
        else:
            return pd.concat([data, new_data])

    def answer(self) -> xr.Dataset:
        "return the final answer as a dictionary"
        if self.scores is None:
            raise RuntimeError("No scores have been calculated yet")
        return self.loss(self.scores)

    def save_data_hdf5(self):
        "save all the data to hdf5"
        if self.directory is None:
            raise RuntimeError("No directory specified")

        if self.chain is None or self.scores is None:
            raise RuntimeError("No data to save")

        data = hdf5.H5( (self.directory / "data.h5").as_posix() )
        data.root.chain = self.chain.to_numpy()
        data.root.chain_columns = self.chain.columns.to_numpy()

        data.root.scores = self.scores.to_array().to_numpy()
        data.root.scores_index = self.scores.index.to_numpy()
        data.root.scores_time = self.scores.time.to_numpy()
        data.root.scores_variables = list(self.scores)

        data.root.probability = self.parameters.target_probability
        data.save()


    def load_data_hdf5(self):
        "load data from hdf5 for graphing or for further refinement"
        if self.directory is None:
            raise RuntimeError("No directory specified")

        data = hdf5.H5( (self.directory / "data.h5").as_posix())
        data.load()

        self.chain = pd.DataFrame(data.root.chain, columns=data.root.chain_columns)

        scores = xr.DataArray(
            data = data.root.scores,
            coords={
                'variable': [i.decode() for i in data.root.scores_variables],
                'index': data.root.scores_index,
                'time': data.root.scores_time
            }
        )

        self.scores = scores.to_dataset('variable')

        self.parameters.target_probability = data.root.probability

    def get_plots(self) -> dict[str, plt.Figure]:
        "return all the plots as a dictionary"
        plots = {}

        if self.scores is None or self.chain is None:
            raise RuntimeError("No data to plot")

        probability = self.loss(self.scores)

        quantiles = [i for i in probability.coords['quantile'].to_numpy()]
        time = [i for i in probability.coords['time'].to_numpy()]
        min_quantile = min(quantiles)
        max_quantile = max(quantiles)

        for key,value in probability.items():
            fig = plt.figure()

            for quantile in quantiles:
                plt.plot(time, value.sel(quantile=quantile), label=f"{quantile:.2f}")
            plt.fill_between(time, value.sel(quantile=min_quantile),  value.sel(quantile=max_quantile), alpha=0.2)

            plt.legend()
            plt.title(key)
            plt.tight_layout()
            plots[str(key)] = fig

        dataset = az.convert_to_inference_data(self.chain.to_dict(orient="list"))

        fig = plt.figure()
        az.plot_pair(
            dataset,
            kind="kde",
            textsize=22,
            marginals=True,
            kde_kwargs={
                'bw':'scott',
                "hdi_probs": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "contourf_kwargs": {"cmap": "Greys"}},
        marginal_kwargs={'bw':'scott'}
        )
        plt.title("Pairwise KDE")
        plt.tight_layout()
        plots['corner'] = fig

        return plots

    def create_plots(self):
        "create plots"
        for key, value in self.get_plots().items():
            if self.directory is not None:
                value.savefig(self.directory / f"{key}.png")
            value.show()


    def display_results(self, count:int, convergence: np.ndarray):
        "display the results  (change to logging library once moved to gumps)"
        flat = convergence.flatten()
        true_count = np.count_nonzero(flat)
        temp  = [f"Gen= {count} Convergence= {true_count}/{len(flat)} = {true_count/len(flat):.1%}"]
        logger.info(' '.join(temp))


    def loss(self, distribution: xr.Dataset) -> xr.Dataset:
        "calculate the loss for this distribution"
        probabilities = np.atleast_1d(self.parameters.target_probability)
        quantiles = distribution.quantile(probabilities, dim='index')
        return quantiles


    def run(self):
        "run the monte carlo solver"
        solver = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloTimeSeriesSolver(solver_settings=self.parameters)

        chain = None
        chain_scores = None

        counter = itertools.count()

        with self.batch:
            while solver.has_next():
                count = next(counter)
                input_data = solver.ask()

                input_data, distribution = run_batch_time_iterator(self.batch, input_data, self.processing_function, self.pre_processing_function)

                #correct all the indexes
                input_data = input_data.reset_index(drop=True)
                input_data.index = input_data.index + self.next_index
                distribution = distribution.assign_coords(index=input_data.index)

                self.next_index += len(input_data)

                chain_scores = self.update_dataset(chain_scores, distribution)
                chain = self.update_dataframe(chain,input_data)

                score = self.loss(chain_scores)

                solver.tell(score)

                self.display_results(count, solver.get_convergence())

        self.chain = chain
        self.scores = chain_scores
