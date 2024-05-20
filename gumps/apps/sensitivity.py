# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"create a sensitivity app using SALib for sensitivity analysis"

import logging
from pathlib import Path
from typing import Callable, Optional

import matplotlib.axes
import numpy as np
import pandas as pd
import SALib.util.results

import gumps.solvers.sensitivity
from gumps.common import hdf5
from gumps.studies.batch_study import AbstractBatchStudy

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class SensitivityApp:
    "Create a sensitivity app"

    def __init__(self, *, parameters: gumps.solvers.sensitivity.SensitivitySolverParameters,
        processing_function: Callable, directory:Optional[Path],
        batch: AbstractBatchStudy,pre_processing_function: Callable|None=None):
        "initialize the sensitivity app"

        self.parameters = parameters
        self.processing_function = processing_function
        self.pre_processing_function = pre_processing_function
        self.batch = batch

        self.directory = directory
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

        self.input_data: pd.DataFrame | None = None
        self.output_data: pd.DataFrame | None = None
        self.analysis: SALib.util.results.ResultDict|None = None
        self.solver = gumps.solvers.sensitivity.SensitivitySolver(solver_settings=self.parameters)

    def results(self) -> dict[str, pd.DataFrame|None]:
        "return the final result"
        if self.analysis is None:
            raise RuntimeError("The sensitivity app has not been run yet")

        if self.parameters.calc_second_order:
            ST, S1, S2 = self.analysis.to_df()
        else:
            ST, S1 = self.analysis.to_df()
            S2 = None
        return {'ST': ST, 'S1': S1, 'S2': S2}

    def save_data_hdf5(self):
        "save all the data to hdf5"
        if self.input_data is None or self.output_data is None:
            raise RuntimeError("The sensitivity app has not been run yet")


        if self.directory is None:
            raise RuntimeError("No directory specified")


        data = hdf5.H5( (self.directory / "data.h5").as_posix() )
        data.root.input_data = self.input_data.to_numpy()
        data.root.output_data = self.output_data.to_numpy()
        data.root.input_names = self.input_data.columns
        data.root.output_names = self.output_data.columns

        results = self.results()

        for key, value in results.items():
            if value is not None:
                data.root[key] = value.to_numpy()
                data.root[f"{key}_names"] = value.columns
        data.save()

    def run(self):
        "run the sensitivity app"
        with self.batch:
            while self.solver.has_next():
                input_data = self.solver.ask()

                input_data = pd.DataFrame(input_data, columns=self.parameters.lower_bound.index.tolist())

                if self.pre_processing_function is not None:
                    input_data_processed = self.pre_processing_function(input_data)
                else:
                    input_data_processed = input_data

                temp_output: list[dict] = []
                keep_indexes: list[int] = []

                for idx, _, data in self.batch.iter_run(input_data_processed, self.processing_function):
                    keep_indexes.append(idx)
                    temp_output.append(data)

                output_data = pd.DataFrame(temp_output)
                input_data = input_data.iloc[keep_indexes]

                if output_data.shape[1] != 1:
                    raise RuntimeError("The sensitivity app only supports one output column")

                self.solver.tell(output_data.to_numpy().squeeze())

                self.input_data = input_data
                self.output_data = output_data
                self.analysis = self.solver.analysis


    def plot(self)-> np.ndarray[matplotlib.axes._axes.Axes]:
        "plot the results"
        if self.analysis is None:
            raise RuntimeError("The sensitivity app has not been run yet")

        return self.solver.plot()
