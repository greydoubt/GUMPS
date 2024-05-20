# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import numpy as np
import pandas as pd
import scipy.stats

import gumps.apps.monte_carlo
import gumps.kernels.ackley_complete_kernel
import gumps.solvers.monte_carlo_solver
import gumps.studies.ackley_batch_study
from gumps.common.parallel import Parallel
from gumps.studies.batch_study import BatchStudyMultiProcess
from gumps.studies.study import SimpleSimulationStudy


def processing_function(frame: pd.DataFrame|None) -> dict|None:
    "process the dataframe for the ackley function"
    if frame is None:
        return None
    return {'total':frame['total'].values[0]}

class TesAckleyUq(unittest.TestCase):
    "test the ackley batch studies with monte carlo solver"

    def test_ackley_batch_study_uq(self):
        "integration test for monte carlo solver"

        def pre_processing_function(input_data: pd.DataFrame):
            "pre-process the data"
            input_data_processed = {}
            input_data_processed['x'] = list(np.column_stack([input_data.x1, input_data.x2, input_data.x3, input_data.x4]))
            input_data_processed['a'] = input_data.a
            input_data_processed['b'] = input_data.b
            input_data_processed['c'] = input_data.c

            return pd.DataFrame(input_data_processed)

        def processing_function(frame: pd.DataFrame):
            "process the dataframe for the ackley function"
            return pd.DataFrame(frame['total'])

        distributions = {"x1":scipy.stats.uniform(0.0, 1),
                        "x2":scipy.stats.uniform(0.0, 1),
                        "x3":scipy.stats.uniform(0.0, 1),
                        "x4":scipy.stats.uniform(0.0, 1),
                        "a":scipy.stats.uniform(10, 30),
                        "b":scipy.stats.uniform(0.1, 0.5),
                        "c":scipy.stats.uniform(np.pi, 5*(np.pi))
                        }

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions,
                                                                        target_probability=[0.45],
                                                                        window=3,
                                                                        tolerance=1e-1,
                                                                        min_steps=3,
                                                                        sampler_seed=0,
                                                                        sampler_scramble=False)

        model_variables = {}

        batch = gumps.studies.ackley_batch_study.AckleyBatchStudy(model_variables=model_variables)

        app = gumps.apps.monte_carlo.MonteCarloApp(parameters=parameters,
            processing_function=processing_function,
            pre_processing_function=pre_processing_function,
            directory=None,
            batch=batch)
        app.run()

        answer = app.answer()

        known_value = pd.Series({'total_0.45':5.2})

        pd.testing.assert_series_equal(answer, known_value, rtol=1e-1)

    def test_ackley_complete_kernel_uq(self):
        "UQ test for ackley with complete kernel"
        def pre_processing_function(input_data: pd.DataFrame):
            "pre-process the data"
            input_data_processed = {}
            input_data_processed['x'] = list(np.column_stack([input_data.x1, input_data.x2, input_data.x3, input_data.x4]))
            input_data_processed['a'] = input_data.a
            input_data_processed['b'] = input_data.b
            input_data_processed['c'] = input_data.c

            return pd.DataFrame(input_data_processed)

        distributions = {"x1":scipy.stats.uniform(0.0, 1),
                        "x2":scipy.stats.uniform(0.0, 1),
                        "x3":scipy.stats.uniform(0.0, 1),
                        "x4":scipy.stats.uniform(0.0, 1),
                        "a":scipy.stats.uniform(10, 30),
                        "b":scipy.stats.uniform(0.1, 0.5),
                        "c":scipy.stats.uniform(np.pi, 5*(np.pi))
                        }

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions,
                                                                        target_probability=[0.45],
                                                                        window=3,
                                                                        tolerance=1e-1,
                                                                        min_steps=3,
                                                                        sampler_seed=0,
                                                                        sampler_scramble=False)


        kernel = gumps.kernels.ackley_complete_kernel.AckleyCompleteKernel()

        problem = {'x' : np.array([1,2]),'a' : 1,'b' : 2 ,'c' : 1}

        study = SimpleSimulationStudy(problem=problem, kernel=kernel)
        parallel = Parallel(poolsize=4)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        app = gumps.apps.monte_carlo.MonteCarloApp(parameters=parameters,
            processing_function=processing_function,
            pre_processing_function=pre_processing_function,
            directory=None,
            batch=batch)
        app.run()

        answer = app.answer()

        known_value = pd.Series({'total_0.45':5.2})

        pd.testing.assert_series_equal(answer, known_value, rtol=1e-1)

    def test_ackley_complete_kernel_alternate_uq(self):
        "UQ test for ackley with complete kernel"
        def pre_processing_function(input_data: pd.DataFrame):
            "pre-process the data"
            input_data_processed = {}
            input_data_processed['x'] = list(np.column_stack([input_data.x1, input_data.x2, input_data.x3, input_data.x4]))
            input_data_processed['a'] = input_data.a
            input_data_processed['b'] = input_data.b
            input_data_processed['c'] = input_data.c

            return pd.DataFrame(input_data_processed)

        distributions = {"x1":scipy.stats.uniform(0.0, 1),
                        "x2":scipy.stats.uniform(0.0, 1),
                        "x3":scipy.stats.uniform(0.0, 1),
                        "x4":scipy.stats.uniform(0.0, 1),
                        "a":scipy.stats.uniform(10, 30),
                        "b":scipy.stats.uniform(0.1, 0.5),
                        "c":scipy.stats.uniform(np.pi, 5*(np.pi))
                        }

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions,
                                                                        target_probability=[0.45],
                                                                        window=3,
                                                                        tolerance=1e-1,
                                                                        min_steps=3,
                                                                        sampler_seed=0,
                                                                        sampler_scramble=False)

        kernel = gumps.kernels.ackley_complete_kernel.AckleyCompleteKernelAlternate()

        problem = {'x' : np.array([1,2]),'a' : 1,'b' : 2 ,'c' : 1}

        study = SimpleSimulationStudy(problem=problem, kernel=kernel)
        parallel = Parallel(poolsize=4)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        app = gumps.apps.monte_carlo.MonteCarloApp(parameters=parameters,
            processing_function=processing_function,
            pre_processing_function=pre_processing_function,
            directory=None,
            batch=batch)
        app.run()

        answer = app.answer()

        known_value = pd.Series({'total_0.45':5.2})

        pd.testing.assert_series_equal(answer, known_value, rtol=1e-1)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TesAckleyUq)
    unittest.TextTestRunner(verbosity=2).run(suite)
