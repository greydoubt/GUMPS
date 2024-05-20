# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the kernel principal component analysis solver"

import tempfile
import unittest
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import sklearn.decomposition
import matplotlib.axes

from gumps.scalers.log_combo_scaler import LogComboScaler
from gumps.scalers.null_scaler import NullScaler
from gumps.solvers.kernel_pca import KernelPCA, KernelPCASettings


class TestKernelPCA(unittest.TestCase):
    "test the PCA solver"

    def test_init(self):
        "test the initialization of the PCA solver"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)
        self.assertFalse(solver.fitted)
        self.assertIsInstance(solver.settings, KernelPCASettings)
        self.assertIsInstance(solver.pca, sklearn.decomposition.KernelPCA)
        self.assertIsInstance(solver.input_scaler, LogComboScaler)
        self.assertIsInstance(solver.scaled_input, pd.DataFrame)

    def test_bad_kernel(self):
        "test the initialization of the PCA solver with a bad kernel"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        with self.assertRaises(ValueError):
            KernelPCASettings(input_data, kernel="bad_kernel")

    def test_set_poly_kernel(self):
        "test the set_poly_kernel method"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})

        settings = KernelPCASettings(input_data, kernel="poly")
        solver = KernelPCA(settings)
        self.assertEqual(solver.pca.kernel, "poly")

    def test_get_pca(self):
        "test the get_pca method"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        pca = solver._get_pca()
        self.assertIsInstance(pca, sklearn.decomposition.KernelPCA)

    def test_get_scaler_auto_true(self):
        "test the get_scaler method"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data,
                               auto_scaling=True)
        solver = KernelPCA(settings)

        scaler = solver._get_scaler()
        self.assertIsInstance(scaler, LogComboScaler)

    def test_get_scaler_auto_false(self):
        "test the get_scaler method"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data,
                               auto_scaling=False)
        solver = KernelPCA(settings)

        scaler = solver._get_scaler()
        self.assertIsInstance(scaler, NullScaler)

    def test_scale_values(self):
        "test the scale_values method"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        scaled_values = solver._scale_values()
        self.assertEqual(scaled_values.shape, (3, 2))

    def test_fit(self):
        "test the fit method"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        solver.fit()

        self.assertTrue(solver.fitted)

    def test_fit_poly(self):
        "test the fit method with a poly kernel"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data, kernel="poly", degree=2)
        solver = KernelPCA(settings)

        solver.fit()

        self.assertTrue(solver.fitted)

    def test_fit_transform(self):
        "test the fit_transform method"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        transformed_data = solver.fit_transform()
        self.assertEqual(transformed_data.shape, (3, 1))

    def test_get_params(self):
        "test the get_params method"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)
        params = solver.get_params()
        self.assertEqual(params["n_components"], None)

    def test_inverse_transform(self):
        "test the inverse_transform method"
        input_data = pd.DataFrame({"x1": [1.0, 0.9, 1.0],
                                   "x2": [1.1, 1.0, 0.9],
                                   "x3": [0.9, 1, 1.1]})
        settings = KernelPCASettings(input_data, fit_inverse_transform=True)
        solver = KernelPCA(settings)

        solver.fit()

        transformed_data = solver.fit_transform()
        inverse_transformed_data = solver.inverse_transform(transformed_data)

        pd.testing.assert_frame_equal(input_data, inverse_transformed_data, atol=5e-2)

    def test_inverse_transform_exception(self):
        "test the inverse_transform method exception"
        input_data = pd.DataFrame({"x1": [1.0, 0.9, 1.0],
                                   "x2": [1.1, 1.0, 0.9],
                                   "x3": [0.9, 1, 1.1]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        with self.assertRaises(RuntimeError):
            solver.inverse_transform(input_data)

    def test_set_params(self):
        "test the set_params method"
        input_data = pd.DataFrame({"x1": [1.0, 0.9, 1.0, 0.9, 1.0, 1.1],
                                   "x2": [1.1, 1.0, 0.9, 0.9, 1.0, 1.1],
                                   "x3": [0.9, 1, 1.1, 0.9, 1.0, 1.1]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        solver.set_params(n_components=2)
        self.assertEqual(solver.pca.n_components, 2)

    def test_transform(self):
        "test the transform method"
        input_data = pd.DataFrame({"x1": [1.0, 0.9, 1.0, 0.9, 1.0, 1.1],
                                   "x2": [1.1, 1.0, 0.9, 0.9, 1.0, 1.1],
                                   "x3": [0.9, 1, 1.1, 0.9, 1.0, 1.1]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        solver.fit()

        transformed_data = solver.transform(solver.settings.input_data)
        self.assertEqual(transformed_data.shape, (6, 3))

    def test_transform_exception(self):
        "test the transform method exception"
        input_data = pd.DataFrame({"x1": [1.0, 0.9, 1.0, 0.9, 1.0, 1.1],
                                   "x2": [1.1, 1.0, 0.9, 0.9, 1.0, 1.1],
                                   "x3": [0.9, 1, 1.1, 0.9, 1.0, 1.1]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        with self.assertRaises(RuntimeError):
            solver.transform(solver.settings.input_data)

    def test_save(self):
        "test the save method"
        input_data = pd.DataFrame({"x1": [1.0, 0.9, 1.0, 0.9, 1.0, 1.1],
                                   "x2": [1.1, 1.0, 0.9, 0.9, 1.0, 1.1],
                                   "x3": [0.9, 1, 1.1, 0.9, 1.0, 1.1]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        solver.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            solver.save(temp_dir)

            self.assertTrue((temp_dir / "pca.joblib").exists())
            self.assertTrue((temp_dir / "input_scaler.joblib").exists())
            self.assertTrue((temp_dir / "settings.joblib").exists())
            self.assertTrue((temp_dir / "scaled_input.joblib").exists())

    def test_save_exception(self):
        "test the save method exception"
        input_data = pd.DataFrame({"x1": [1.0, 0.9, 1.0, 0.9, 1.0, 1.1],
                                   "x2": [1.1, 1.0, 0.9, 0.9, 1.0, 1.1],
                                   "x3": [0.9, 1, 1.1, 0.9, 1.0, 1.1]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)
        with self.assertRaises(RuntimeError):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                solver.save(temp_dir)

    def test_load(self):
        "test the load method"
        input_data = pd.DataFrame({"x1": [1.0, 0.9, 1.0, 0.9, 1.0, 1.1],
                                   "x2": [1.1, 1.0, 0.9, 0.9, 1.0, 1.1],
                                   "x3": [0.9, 1, 1.1, 0.9, 1.0, 1.1]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        solver.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            solver.save(temp_dir)

            solver2 = KernelPCA.load(temp_dir)

            np.testing.assert_almost_equal(solver.pca.eigenvalues_, solver2.pca.eigenvalues_)
            np.testing.assert_almost_equal(solver.pca.eigenvectors_, solver2.pca.eigenvectors_)
            np.testing.assert_almost_equal(solver.input_scaler.scaler.scaler.scale_,
                                   solver2.input_scaler.scaler.scaler.scale_)

            np.testing.assert_almost_equal(solver.input_scaler.scaler.scaler.var_,
                                   solver2.input_scaler.scaler.scaler.var_)

    def test_get_principal_components(self):
        "test the get_principal_components method"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        solver.fit()

        principal_components = solver.get_principal_components()
        self.assertEqual(principal_components.shape, (3, 1))

    def test_get_principal_components_exception(self):
        "test the get_principal_components method exception"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        with self.assertRaises(RuntimeError):
            solver.get_principal_components()

    def test_get_explained_variance(self):
        "test the get_explained_variance method"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        solver.fit()

        explained_variance = solver.get_explained_variance()
        self.assertEqual(explained_variance.shape, (1,1))

    def test_get_explained_variance_exception(self):
        "test the get_explained_variance method exception"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        with self.assertRaises(RuntimeError):
            solver.get_explained_variance()

    def test_get_explained_variance_plot(self):
        "test the get_explained_variance_plot method"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        solver.fit()

        explained_variance_plot = solver.get_explained_variance_plot()
        self.assertIsInstance(explained_variance_plot, matplotlib.axes.Axes)


    def test_get_explained_variance_plot_exception(self):
        "test the get_explained_variance_plot method exception"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data)
        solver = KernelPCA(settings)

        with self.assertRaises(RuntimeError):
            solver.get_explained_variance_plot()


    def test_rebuild_model(self):
        "rebuild the regressor"
        base_dir = Path(__file__).parent.parent / "data"
        regressor_dir = base_dir / "solver" / "kernel_pca"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertLogs('gumps.solvers.kernel_pca', level='WARNING') as cm:
                instance = KernelPCA.rebuild_model(temp_dir, auto_resave=False)

            self.assertIsInstance(instance, KernelPCA)

            self.assertTrue('WARNING:gumps.solvers.kernel_pca:PCA failed to load, rebuilding PCA.' in cm.output)


    def test_rebuild_model_resave(self):
        "rebuild the regressor and resave it"
        base_dir = Path(__file__).parent.parent / "data"
        regressor_dir = base_dir / "solver" / "kernel_pca"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertLogs('gumps.solvers.kernel_pca', level='WARNING') as cm:
                instance = KernelPCA.rebuild_model(temp_dir, auto_resave=True)

            self.assertIsInstance(instance, KernelPCA)

            self.assertTrue('WARNING:gumps.solvers.kernel_pca:PCA failed to load, rebuilding PCA.' in cm.output)
            self.assertTrue('WARNING:gumps.solvers.kernel_pca:PCA was rebuilt, resaving PCA.' in cm.output)


    def test_load_instance(self):
        "test loading the regressor"
        input_data = pd.DataFrame({"x1": [1, 2, 3],
                                   "x2": [4, 5, 6]})
        settings = KernelPCASettings(input_data)
        regressor = KernelPCA(settings)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)

            instance = KernelPCA.load_instance(temp_dir)

            self.assertIsInstance(instance, KernelPCA)


    def test_load_instance_exception(self):
        "test the load instance method with an exception"
        base_dir = Path(__file__).parent.parent / "data"
        regressor_dir = base_dir / "solver" / "kernel_pca"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertRaises(RuntimeError):
                KernelPCA.load_instance(temp_dir)


    def test_load_rebuild(self):
        "load the regressor with a different version of sklearn than it was saved with and auto-matically rebuild it"
        base_dir = Path(__file__).parent.parent / "data"
        regressor_dir = base_dir / "solver" / "kernel_pca"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertLogs('gumps.solvers.kernel_pca', level='WARNING') as cm:
                KernelPCA.load(temp_dir, auto_resave=False)
            self.assertTrue('WARNING:gumps.solvers.kernel_pca:PCA failed to load, rebuilding PCA.' in cm.output)

    def test_load_rebuild_resave(self):
        "load the regressor with a different version of sklearn than it was saved with and auto-matically rebuild it"
        base_dir = Path(__file__).parent.parent / "data"
        regressor_dir = base_dir / "solver" / "kernel_pca"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertLogs('gumps.solvers.kernel_pca', level='WARNING') as cm:
                KernelPCA.load(temp_dir, auto_resave=True, auto_rebuild=True)
            self.assertTrue('WARNING:gumps.solvers.kernel_pca:PCA failed to load, rebuilding PCA.' in cm.output)
            self.assertTrue('WARNING:gumps.solvers.kernel_pca:PCA was rebuilt, resaving PCA.' in cm.output)

    def test_load_exception(self):
        "load the regressor with a different version of sklearn and disallow auto-rebuild"
        base_dir = Path(__file__).parent.parent / "data"
        regressor_dir = base_dir / "solver" / "kernel_pca"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertRaises(RuntimeError):
                KernelPCA.load(temp_dir, auto_rebuild=False)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKernelPCA)
    unittest.TextTestRunner(verbosity=2).run(suite)
