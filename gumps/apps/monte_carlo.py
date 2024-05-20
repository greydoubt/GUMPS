# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Monte carlo app for uncertainty quantification"

import itertools
from typing import Callable, Optional

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use('AGG')

import logging
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt

import gumps.solvers.monte_carlo_solver
from gumps.common import hdf5
from gumps.common.app_utils import run_batch_iterator
from gumps.studies.batch_study import AbstractBatchStudy

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MonteCarloApp:
    "create a monte carlo app"

    def __init__(self, *, parameters: gumps.solvers.monte_carlo_solver.MonteCarloParameters,
        processing_function: Callable, directory:Optional[Path]=None,
        batch: AbstractBatchStudy, pre_processing_function: Callable|None=None):
        "initialize the monte carlo app"
        self.parameters = parameters
        self.processing_function = processing_function
        self.pre_processing_function = pre_processing_function
        self.batch = batch

        self.chain: Optional[pd.DataFrame] = None
        self.scores: Optional[pd.DataFrame] = None

        self.directory = directory
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def update(data: Optional[pd.DataFrame], new_data: pd.DataFrame):
        "update data with new_data"
        if data is None:
            return new_data
        else:
            return pd.concat([data, new_data])

    def answer(self):
        "return the final answer as a dictionary"
        return self.loss(self.scores)

    def save_data_hdf5(self):
        "save all the data to hdf5"
        data = hdf5.H5(self.directory / "data.h5")
        data.root.chain = self.chain
        data.root.scores = self.scores
        data.root.probability = self.parameters.target_probability
        data.save()

    def find_limits(self) -> tuple[dict[str, float], dict[str, float]]:
        "find the plotting limits to center the plot without cutting out chosen targets"
        lower_limit = np.min(self.parameters.target_probability)
        upper_limit = np.min(1-np.array(self.parameters.target_probability))
        min_size = min(lower_limit, upper_limit)
        probability_points = np.array([min_size, 1 - min_size])
        extremes = np.atleast_1d(np.percentile(self.scores, probability_points * 100, axis=0))
        headers = list(self.scores)
        diff = extremes[1] - extremes[0]
        lower_bound = dict(zip(headers, extremes[0] - 0.1 * diff))
        upper_bound = dict(zip(headers, extremes[1] + 0.1 * diff))
        return lower_bound, upper_bound

    def create_plots(self):
        "create plots"
        probability = self.loss(self.scores)

        lower_bound, upper_bound = self.find_limits()

        for key,value in self.scores.items():
            plt.figure()
            sns.histplot(value, stat="density")
            for join_name, value in probability.items():
                name, prob = join_name.rsplit('_', maxsplit=1)
                if name == key:
                    plt.axvline(value, color='r', label=f"P({prob})={value:.2e}")
            plt.xlim(lower_bound[key], upper_bound[key])
            plt.legend()
            plt.tight_layout()

            if self.directory is not None:
                plt.savefig(self.directory / f"{key}_scores.png")

        dataset = az.convert_to_inference_data(self.chain.to_dict(orient="list"))

        plt.figure()
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

        plt.tight_layout()

        if self.directory is not None:
            plt.savefig(self.directory / "az_corner.png")

    def display_results(self, counter, scores: pd.Series):
        "display the results  (change to logging library once moved to gumps)"
        temp  = [f"Gen= {next(counter)}"]
        for key, value in scores.items():
            name, prob = key.rsplit('_', maxsplit=1)
            temp.append(f"P[{name}]({prob})={value:.2e}")
        logger.info(' '.join(temp))

    def loss(self, distribution: pd.DataFrame) -> pd.Series:
        "calculate the loss for this distribution"
        probabilities = np.atleast_1d(self.parameters.target_probability)
        headers = list(distribution)

        result: dict[str, float] = {}
        for probability in probabilities:
            points = np.percentile(distribution, probability * 100, axis=0)
            for header, point in zip(headers, points):
                result[f"{header}_{probability}"] = point

        return pd.Series(result)

    def run(self):
        "run the monte carlo solver"
        solver = gumps.solvers.monte_carlo_solver.MonteCarloSolver(solver_settings=self.parameters)

        chain = None
        chain_scores = None

        counter = itertools.count()

        with self.batch:
            while solver.has_next():
                input_data = solver.ask()

                input_data, distribution = run_batch_iterator(self.batch, input_data, self.processing_function, self.pre_processing_function)

                chain_scores = self.update(chain_scores, distribution)
                chain = self.update(chain,input_data)

                score = self.loss(chain_scores)

                self.display_results(counter, score)

                solver.tell(score)

        self.chain = chain
        self.scores = chain_scores
