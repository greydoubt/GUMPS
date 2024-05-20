# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""This is an adaptive sampler that uses a Gaussian regressor to sample the next points.
The system is based on pure exploration to reduce uncertainty in the model. In order
to generate a batch of points the highest uncertainty is sampled first. The model is then
recalculated with that point set to its mean value and the next highest uncertainty is sampled.
This process is repeated until the batch is full. The batch size is set by the user. The
number of points to add is also set by the user.

This algorithm is expensive to calculate and depending on the complex it can take seconds
to minutes to calculate the next batch of points. This should only be used when a simulation
is much slower than the time it takes to calculate the next batch of points. Otherwise just use
a normal sampler with many points.
"""

import copy

import attrs
import numpy as np
import pandas as pd

from gumps.solvers.batch_solver import AbstractBatchSolver
from gumps.solvers.regressors.gaussian_regressor import GaussianRegressor
from gumps.solvers.pymoo_solvers import PyMooSolver, PyMooSolverParameters

from sklearn.preprocessing import MinMaxScaler


@attrs.define
class AdaptiveSamplerParameters:
    "This is the parameters for the adaptive sampler"
    points_to_add : int
    batch_size: int = 1
    max_iterations: int = 100
    population_size: int = 100

@attrs.define
class WhatIf:
    input_data: pd.DataFrame
    output_data: pd.DataFrame
    uncertainty: pd.DataFrame
    regressor: GaussianRegressor

class GaussianRegressorAdaptiveSampler(AbstractBatchSolver):
    """Gaussian regressor adaptive sampler that uses a Gaussian regressor to sample the next points
    and it based on pure exploration to reduce uncertainty."""

    def __init__(self, regressor: GaussianRegressor,
                 solver_settings: AdaptiveSamplerParameters) -> None:
        "Initialize the adaptive sampler"
        self.regressor = regressor
        self.solver_settings = solver_settings
        self.points_to_add = solver_settings.points_to_add

        self.new_input_data : pd.DataFrame | None = None

        self.pymoo_scaler = self.get_input_scaler()

    def get_input_scaler(self) -> MinMaxScaler:
        "Get the input scaler"
        scaler = MinMaxScaler()
        scaler.fit(self.regressor.parameters.input_data)
        return scaler

    def has_next(self) -> bool:
        "Check if more points are available"
        return bool(self.points_to_add)

    def ask(self) -> pd.DataFrame:
        "Ask for the next batch of points"
        whatif = self.whatif()
        self.new_input_data = whatif.input_data
        return whatif.input_data

    def whatif(self) -> WhatIf:
        "return information for the next batch of points"
        points_input = []
        points_output = []
        points_uncertainty = []

        #the regressor is copied because we update the regressor
        #with the most probable value for each point
        #to select the next point
        regressor_copy = copy.deepcopy(self.regressor)

        for _ in range(self.solver_settings.batch_size):
            point, output, uncertainty = self._ask_one(regressor_copy)

            points_input.append(point)
            points_output.append(output)
            points_uncertainty.append(uncertainty)

        return WhatIf(input_data=pd.DataFrame(points_input),
                        output_data=pd.DataFrame(points_output),
                        uncertainty=pd.DataFrame({'uncertainty': points_uncertainty}),
                        regressor=regressor_copy)

    def get_pymoo_solver_settings(self) -> PyMooSolverParameters:
        "get the pymoo solver settings and return them"
        number_var = len(self.regressor.parameters.input_data.columns)
        solver_settings = PyMooSolverParameters(
            number_var=number_var,
            number_obj=1,
            lower_bound=np.zeros(number_var),
            upper_bound=np.ones(number_var),
            auto_transform=False,
            population_size=self.solver_settings.population_size,
            algorithm_name="unsga3",
            total_generations=self.solver_settings.max_iterations,
        )
        return solver_settings

    def _ask_one(self, regressor:GaussianRegressor) -> tuple[pd.Series, pd.Series, float]:
        "ask for one new point, the regressor is updated with the most probable value"
        pymoo_settings = self.get_pymoo_solver_settings()
        solver = PyMooSolver(solver_settings=pymoo_settings)

        next_point_input: pd.Series
        next_point_output: pd.Series
        next_point_uncertainty = 0

        while solver.has_next():
            population = solver.ask()
            population = self.pymoo_scaler.inverse_transform(population)
            population_df = pd.DataFrame(population, columns=self.regressor.parameters.input_data.columns)

            output_data, output_std = regressor.predict_uncertainty(population_df)

            #find the highest uncertainty volume
            output_std = output_std.product(axis=1)

            idx_max = output_std.argmax()
            if output_std.iloc[idx_max] > next_point_uncertainty:
                next_point_input = population_df.iloc[idx_max]
                next_point_output = output_data.iloc[idx_max]
                next_point_uncertainty = output_std.iloc[idx_max]

            solver.tell(-output_std.to_frame().to_numpy())

        #update regressor with the mean predicted value for the chosen point
        regressor.update_data(pd.DataFrame([next_point_input]), pd.DataFrame([next_point_output]))

        #remove the name, otherwise it is used as an index which creates problems elsewhere
        next_point_input.name = None
        next_point_output.name = None

        return next_point_input, next_point_output, next_point_uncertainty

    def tell(self, loss: pd.DataFrame, request: pd.DataFrame | None = None) -> None:
        "Tell the sampler the loss for the last batch of points"
        if self.new_input_data is None:
            raise RuntimeError("ask must be called before tell")

        if request is None:
            request = self.new_input_data

        if loss.shape[0] != request.shape[0]:
            raise ValueError("The loss and request must have the same number of rows")

        self.new_input_data = request

        self.points_to_add = max(0, self.points_to_add - len(loss))
        self.regressor.update_data(self.new_input_data, loss)