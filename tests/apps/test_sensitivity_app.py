# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the sensitivity app"

import tempfile
import unittest
from pathlib import Path

import matplotlib.axes
import pandas as pd

from gumps.apps.sensitivity import SensitivityApp
from gumps.solvers.sensitivity import SensitivitySolverParameters
from gumps.studies.batch_sphere_study import BatchLineStudy


def pre_processing_function(input_data: pd.DataFrame):
    "pre-process the data, this just makes sure the code path is hit"
    input_data_processed = input_data.copy()
    return input_data_processed

def processing_function(frame:pd.DataFrame) -> pd.DataFrame:
    "processing function to get the total from the dataframe"
    return pd.DataFrame(frame['total'])

class TestSensitivityApp(unittest.TestCase):
    "test the sensitivity app using the Ackley Batch Study"

    def setUp(self):
        "set up the test"
        self.lower_bound = pd.Series({'x1': -1, 'x2': -1, 'x3':-1, 'x4':-1})
        self.upper_bound = pd.Series({'x1': 1, 'x2': 1, 'x3':1, 'x4':1})
        self.parameters = SensitivitySolverParameters(lower_bound=self.lower_bound,
                                                upper_bound=self.upper_bound)

        model_variables = {'a1': 0.0, 'a2':0.0, 'a3':0, 'a4':0}
        self.batch = BatchLineStudy(model_variables=model_variables)

    def test_initialize(self):
        "test the initialization of the sensitivity app"
        app = SensitivityApp(parameters=self.parameters,
                                processing_function=processing_function,
                                directory=None,
                                batch=self.batch,
                                pre_processing_function=pre_processing_function)

        self.assertIsInstance(app.parameters, SensitivitySolverParameters)
        self.assertEqual(app.processing_function, processing_function)
        self.assertEqual(app.pre_processing_function, pre_processing_function)
        self.assertEqual(app.batch, self.batch)
        self.assertEqual(app.directory, None)
        self.assertEqual(app.input_data, None)
        self.assertEqual(app.output_data, None)
        self.assertEqual(app.analysis, None)

    def test_initalize_directory(self):
        "test the initialization of the sensitivity app with a directory"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir) / "foo" / "bar"

            SensitivityApp(parameters=self.parameters,
                                processing_function=processing_function,
                                directory=temp_dir,
                                batch=self.batch,
                                pre_processing_function=pre_processing_function)

            self.assertTrue(temp_dir.exists())

    def test_results_exception(self):
        "test that getting the results before running the code generates an exception"
        app = SensitivityApp(parameters=self.parameters,
                                processing_function=processing_function,
                                directory=None,
                                batch=self.batch,
                                pre_processing_function=pre_processing_function)

        with self.assertRaises(RuntimeError):
            app.results()

    def test_run_without_preprocessing(self):
        "test the run method"
        self.parameters.sample_power = 4

        app = SensitivityApp(parameters=self.parameters,
                                processing_function=processing_function,
                                directory=None,
                                batch=self.batch,
                                pre_processing_function=None)

        app.run()

        results = app.results()

        self.assertEqual(results['ST'].shape, (4, 2))
        self.assertEqual(results['S1'].shape, (4, 2))
        self.assertEqual(results['S2'].shape, (6, 2))

    def test_run_multi_output(self):
        "test that the run method fails if the output has more than one dimension"
        self.parameters.sample_power = 4

        def multi_processing_function(frame:pd.DataFrame) -> pd.DataFrame:
            "processing function to get the total from the dataframe"
            return pd.DataFrame(frame[['total', 'd1', 'd2', 'd3', 'd4']])

        app = SensitivityApp(parameters=self.parameters,
                                processing_function=multi_processing_function,
                                directory=None,
                                batch=self.batch,
                                pre_processing_function=pre_processing_function)

        with self.assertRaises(RuntimeError):
            app.run()

    def test_run(self):
        "test the run method"
        self.parameters.sample_power = 4

        app = SensitivityApp(parameters=self.parameters,
                                processing_function=processing_function,
                                directory=None,
                                batch=self.batch,
                                pre_processing_function=pre_processing_function)

        app.run()

        results = app.results()

        self.assertEqual(results['ST'].shape, (4, 2))
        self.assertEqual(results['S1'].shape, (4, 2))
        self.assertEqual(results['S2'].shape, (6, 2))

    def test_run_second_order_false(self):
        "test the run method"
        self.parameters.sample_power = 4
        self.parameters.calc_second_order = False

        app = SensitivityApp(parameters=self.parameters,
                                processing_function=processing_function,
                                directory=None,
                                batch=self.batch,
                                pre_processing_function=pre_processing_function)

        app.run()

        results = app.results()

        self.assertEqual(results['ST'].shape, (4, 2))
        self.assertEqual(results['S1'].shape, (4, 2))
        self.assertIsNone(results['S2'])

    def test_save_data_hdf5(self):
        "test the save_data_hdf5 method"
        self.parameters.sample_power = 4
        self.parameters.calc_second_order = True

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir) / "foo" / "bar"

            app = SensitivityApp(parameters=self.parameters,
                                    processing_function=processing_function,
                                    directory=temp_dir,
                                    batch=self.batch,
                                    pre_processing_function=pre_processing_function)
            app.run()
            app.save_data_hdf5()
            self.assertTrue((temp_dir / "data.h5").exists())


    def test_save_data_hdf5_second_order_false(self):
        "test the save_data_hdf5 when second_order is False"
        self.parameters.sample_power = 4
        self.parameters.calc_second_order = False

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir) / "foo" / "bar"

            app = SensitivityApp(parameters=self.parameters,
                                    processing_function=processing_function,
                                    directory=temp_dir,
                                    batch=self.batch,
                                    pre_processing_function=pre_processing_function)

            app.run()
            app.save_data_hdf5()
            self.assertTrue((temp_dir / "data.h5").exists())


    def test_save_data_hdf5_exception(self):
        "test the save_data_hdf5 when the analysis is None"
        self.parameters.sample_power = 4
        self.parameters.calc_second_order = False

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir) / "foo" / "bar"

            app = SensitivityApp(parameters=self.parameters,
                                    processing_function=processing_function,
                                    directory=temp_dir,
                                    batch=self.batch,
                                    pre_processing_function=pre_processing_function)

            with self.assertRaises(RuntimeError):
                app.save_data_hdf5()

    def test_save_data_hdf5_exception_no_directory(self):
        "test the save_data_hdf5 when the directory is None"
        self.parameters.sample_power = 4
        self.parameters.calc_second_order = False

        app = SensitivityApp(parameters=self.parameters,
                                    processing_function=processing_function,
                                    directory=None,
                                    batch=self.batch,
                                    pre_processing_function=pre_processing_function)

        app.input_data = pd.DataFrame()
        app.output_data = pd.DataFrame()

        with self.assertRaises(RuntimeError):
            app.save_data_hdf5()

    def test_plot(self):
        "test that plotting returns an axes"
        self.parameters.sample_power = 4
        self.parameters.calc_second_order = False

        app = SensitivityApp(parameters=self.parameters,
                                    processing_function=processing_function,
                                    directory=None,
                                    batch=self.batch,
                                    pre_processing_function=pre_processing_function)

        app.run()
        for ax in app.plot():
            self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_plot_exception(self):
        "test that plotting fails when the analysis is None"
        self.parameters.sample_power = 4
        self.parameters.calc_second_order = False

        app = SensitivityApp(parameters=self.parameters,
                                    processing_function=processing_function,
                                    directory=None,
                                    batch=self.batch,
                                    pre_processing_function=pre_processing_function)

        with self.assertRaises(RuntimeError):
            app.plot()


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSensitivityApp)
    unittest.TextTestRunner(verbosity=2).run(suite)
