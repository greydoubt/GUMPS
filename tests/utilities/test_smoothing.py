# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import pandas as pd
import numpy as np
import scipy.signal
import tempfile
from pathlib import Path

from gumps.utilities import smoothing
import gumps.common.exceptions

#Disable protected access for testing
# pylint: disable=W0212

class TestSmoothingInput(unittest.TestCase):
    "test the smoothing utility"

    def test_create_smoothing_input(self):
        "test the smoothing input class"
        df = pd.DataFrame({'a':np.linspace(0, 10, 10), 'b':np.linspace(0, 10, 10), 'c':np.linspace(0, 10, 10)})
        input = smoothing.SmoothingInput(df, 'a', 'b')

        pd.testing.assert_frame_equal(input.df, df)
        self.assertEqual(input.x_label, 'a')
        self.assertEqual(input.y_label, 'b')

    def test_sample_signal(self):
        "test the sampling of the signal"
        df = pd.DataFrame({'a':np.linspace(0, 10, 10), 'b':np.linspace(0, 10, 10), 'c':np.linspace(0, 10, 10)})
        input = smoothing.SmoothingInput(df, 'a', 'b')

        df_sampled = input._resample_signal()
        df_correct = pd.DataFrame({'times':df.a, 'values':df.b})

        pd.testing.assert_frame_equal(df_sampled, df_correct)

    def test_resample_too_many_samples(self):
        "test the resampling of the signal"
        df = pd.DataFrame({'a':np.linspace(0, 10, 10), 'b':np.linspace(0, 10, 10), 'c':np.linspace(0, 10, 10)})
        input = smoothing.SmoothingInput(df, 'a', 'b', max_samples=5)

        df_sampled = input._resample_too_many_samples()
        self.assertEqual(len(df_sampled), 5)

    def test_resample_time_step_size_error(self):
        "test the time step size error"
        df = pd.DataFrame({
            'a':np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b':np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10])})
        input = smoothing.SmoothingInput(df, 'a', 'b')

        error = input._time_step_size_error()

        self.assertEqual(error, 1.0)

    def test_resample_time_step_inconsistent(self):
        "test the resampling of the signal"
        df = pd.DataFrame({
            'a':np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b':np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10])})
        input = smoothing.SmoothingInput(df, 'a', 'b')

        df_sampled = input._resample_inconsistent_step_size()

        df_correct = pd.DataFrame({'times':np.linspace(0, 10, 11), 'values':np.linspace(0, 10, 11)})

        pd.testing.assert_frame_equal(df_sampled, df_correct)

    def test_resample_time_step_inconsistent_too_many(self):
        "test the resampling of the signal"
        df = pd.DataFrame({
            'a':np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b':np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10])})
        input = smoothing.SmoothingInput(df, 'a', 'b', max_samples=5)

        df_sampled = input._resample_inconsistent_step_size()

        self.assertEqual(len(df_sampled), 5)


class TestLPointProblem(unittest.TestCase):

    def test_smoothing_output(self):
        "test the smoothing output class"
        times = np.linspace(0, 10 ,10)
        signal = np.linspace(0, 10, 10)
        derivative = np.linspace(0, 10, 10)
        output = smoothing.SmoothingOutput(times=times, signal=signal, derivative=derivative)

        np.testing.assert_array_equal(output.signal, signal)
        np.testing.assert_array_equal(output.derivative, derivative)
        np.testing.assert_array_equal(output.times, times)

    def test_LPointProblem(self):
        "test the LPointProblem"
        x = np.linspace(1, 11, 11)
        y = np.linspace(11, 1, 11)

        lpp = smoothing.LPointProblem(x=x, y=y)

        np.testing.assert_almost_equal(lpp.x, x)
        np.testing.assert_almost_equal(lpp.y, y)
        np.testing.assert_almost_equal(lpp.x_min, x - 1)
        np.testing.assert_almost_equal(lpp.y_min, y - 1)
        np.testing.assert_almost_equal(lpp.first_point, np.array([0,1]))
        np.testing.assert_almost_equal(lpp.last_point, np.array([1,0]))

    def test_LPointProblem_distance(self):
        x = np.array([0, 0, 1])
        y = np.array([1, 0, 0])

        lpp = smoothing.LPointProblem(x=x, y=y)

        distance = lpp._distance()

        correct = np.array([0, (0.5**2 + 0.5**2)**(1/2), 0])

        np.testing.assert_almost_equal(distance, correct)

    def test_LPointProblem_LPoint(self):
        x = np.array([0, 0, 1])
        y = np.array([1, 0, 0])

        lpp = smoothing.LPointProblem(x=x, y=y)

        L_x, L_y = lpp.LPoint()

        self.assertEqual(L_x, 0)
        self.assertEqual(L_y, 0)

    def test_LPointProblem_LPoint_exception(self):
        x = np.array([0, 1, 1])
        y = np.array([1, 1, 0])

        lpp = smoothing.LPointProblem(x=x, y=y)

        with self.assertRaises(gumps.common.exceptions.NoLPointFoundError):
            lpp.LPoint()

    def test_LPointProblem_distance_diagonal(self):
        x = np.array([0, 0, 1])
        y = np.array([1, 0, 0])

        lpp = smoothing.LPointProblem(x=x, y=y)

        distance = lpp.distance_diagonal(np.array([1,1]).T)

        correct = -((0.5**2 + 0.5**2)**(1/2))
        self.assertAlmostEqual(distance, correct)

class TestMisc(unittest.TestCase):

    def test_signal_bessel(self):
        "test that a bessel function is returned"
        bessel = smoothing.signal_bessel(0.25, 1)
        self.assertIsInstance(bessel, np.ndarray)


class TestCriticalSamplingFrequencyOptimization(unittest.TestCase):

    def test_initialization(self):
        opt = smoothing.CriticalSamplingFrequencyOptimization(lb=0, ub=1, sse_target=1e-4,
                                                              values=np.linspace(0,1,10),
                                                              sampling_frequency=1)
        np.testing.assert_almost_equal(opt.xl, np.array([0.0]))
        np.testing.assert_almost_equal(opt.xu, np.array([1.0]))
        self.assertEqual(opt.sse_target, 1e-4)
        np.testing.assert_almost_equal(opt.values, np.linspace(0,1,10))
        self.assertEqual(opt.sampling_frequency, 1)

    def test_evaluate(self):
        opt = smoothing.CriticalSamplingFrequencyOptimization(lb=np.log10(1e-6),
                                                              ub=np.log10(1e-4/2),
                                                              sse_target=1e-4,
                                                              values=np.linspace(0,1,1000),
                                                              sampling_frequency=1/1000)
        out = {'F': None}
        opt._evaluate(x=np.array([-4.0]), out=out)

        self.assertAlmostEqual(out['F'], 1e-8)

class TestLPointOptimization(unittest.TestCase):

    def test_initialization(self):
        lpp = smoothing.LPointProblem(x=np.linspace(1,10,10), y=np.linspace(1,10,10))
        opt = smoothing.LPointOptimization(lpp, sampling_frequency=1, values=np.linspace(0,1,10))

        self.assertEqual(opt.lpp, lpp)
        np.testing.assert_almost_equal(opt.values, np.linspace(0,1,10))
        self.assertEqual(opt.sampling_frequency, 1)
        np.testing.assert_almost_equal(opt.xl, np.array([0.0]))
        np.testing.assert_almost_equal(opt.xu, np.array([1.0]))


    def test_evaluate(self):
        lpp = smoothing.LPointProblem(x=np.linspace(0,1,10), y=np.linspace(1,0,10))
        opt = smoothing.LPointOptimization(lpp, sampling_frequency=1/1000, values=np.linspace(0,1,1000))


        out = {'F': None}
        opt._evaluate(x=np.array([np.log10(1e-3/2)]), out=out)

        np.testing.assert_allclose(out['F'], -38.14, atol=0, rtol=1e-1)

class TestSmoothing(unittest.TestCase):

    def test_init(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        rmse = 1e-4
        sse = (1e-4)**2 * 1000

        smooth = smoothing.Smoothing(params=params, rmse_target=rmse)

        self.assertEqual(smooth.params, params)
        self.assertEqual(smooth.rmse_target, 1e-4)
        self.assertEqual(smooth.smoothing_factor, sse)

    def test_initialize(self):
        pass

    def test_create_spline(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        spline, factor = smooth._create_spline(params)

        self.assertIsInstance(spline, scipy.interpolate.UnivariateSpline)
        self.assertAlmostEqual(factor, 1)
        self.assertAlmostEqual(spline(0)/factor, 0, delta=1e-2)
        self.assertAlmostEqual(spline(1)/factor, 1, delta=1e-2)

    def test_smoothing_filter_signal(self):
        samples = 1000
        df = pd.DataFrame({'a':np.linspace(0, 1, samples), 'b':np.linspace(0, 1, samples), 'c':np.linspace(0, 1, samples)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        max_frequency = samples/2 - 1

        low_passed = smooth._smoothing_filter_signal(params.resampled_signal["times"], params.resampled_signal["values"], max_frequency)

        diff = np.linalg.norm(low_passed - params.resampled_signal["values"])

        self.assertAlmostEqual(diff, 0, delta=1e-4)

    def test_smoothing_filter_signal_none(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        low_passed = smooth._smoothing_filter_signal(params.resampled_signal["times"], params.resampled_signal["values"], None)

        np.testing.assert_array_almost_equal(low_passed, params.resampled_signal["values"])


    def test_find_critical_sampling_frequency(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        critical_sampling_frequency = smooth._find_critical_sampling_frequency(params.resampled_signal["values"])

        self.assertTrue(10 < critical_sampling_frequency < 500)

    def test_critical_sampling_frequency_LPoint(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.ones(1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        filters = np.array([0, 0.2, 1])
        sse = 10**np.array([1, 0.2, 0])
        critical_sampling_frequency = smooth._critical_sampling_frequency_LPoint(filters, sse)

        self.assertAlmostEqual(0.2, critical_sampling_frequency, delta=1e-2)

    def test_critical_sampling_frequency_LPoint_exception(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)


        filters = np.linspace(0, 1, 1000)
        sse = np.ones(1000)
        critical_sampling_frequency = smooth._critical_sampling_frequency_LPoint(filters, sse)

        self.assertIsNone(critical_sampling_frequency)

    def test_sample_critical_frequency(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        filters, sse = smooth._sample_critical_frequency(params.resampled_signal["values"])

        self.assertIsInstance(filters, np.ndarray)
        self.assertIsInstance(sse, np.ndarray)
        self.assertTrue(len(filters) == len(sse))
        self.assertTrue((filters > 1e-8).all())
        self.assertTrue((filters < 499).all())

    def test_refine_signal(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        filters, sse = smooth._sample_critical_frequency(params.resampled_signal["values"])

        critcal_frequence = smooth._refine_signal(np.linspace(0, 1, 1000), filters, np.log(sse), params.sampling_frequency)

        self.assertTrue(10 < critcal_frequence < 500)

    def test_find_max_signal(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        filters, _ = smooth._sample_critical_frequency(params.resampled_signal["values"])

        max_signal = smooth._find_max_signal(np.linspace(0, 1, 1000), filters)

        self.assertTrue(10 < max_signal < 500)

    def test_signal(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        signal = smooth.signal()

        self.assertIsInstance(signal, smoothing.SmoothingOutput)
        self.assertIsNone(signal.derivative)

    def test_signal_params(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        signal = smooth.signal(params)

        self.assertIsInstance(signal, smoothing.SmoothingOutput)
        self.assertIsNone(signal.derivative)

    def test_signal_derivative(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        signal = smooth.signal_derivative()

        self.assertIsInstance(signal, smoothing.SmoothingOutput)
        self.assertIsNotNone(signal.derivative)

    def test_signal_derivative_params(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        signal = smooth.signal_derivative(params)

        self.assertIsInstance(signal, smoothing.SmoothingOutput)
        self.assertIsNotNone(signal.derivative)

    def test_save_as_hdf5(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir) / "test.hdf5"
            smooth.save_as_hdf5(tmpfile)

            self.assertTrue(tmpfile.exists())

    def test_save_hdf5_exists(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir) / "test.hdf5"
            tmpfile.touch()
            with self.assertRaises(FileExistsError):
                smooth.save_as_hdf5(tmpfile)

    def test_load(self):
        df = pd.DataFrame({'a':np.linspace(0, 1, 1000), 'b':np.linspace(0, 1, 1000), 'c':np.linspace(0, 1, 1000)})
        params = smoothing.SmoothingInput(df, 'a', 'b')
        smooth = smoothing.Smoothing(params=params, rmse_target=1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir) / "test.hdf5"
            smooth.save_as_hdf5(tmpfile)

            smooth2 = smoothing.Smoothing.load_from_hdf5(tmpfile)

        pd.testing.assert_frame_equal(smooth.params.resampled_signal, smooth2.params.resampled_signal)
        self.assertEqual(smooth.rmse_target, smooth2.rmse_target)
        self.assertEqual(smooth.smoothing_factor, smooth2.smoothing_factor)
        self.assertEqual(smooth.critical_sampling_frequency, smooth2.critical_sampling_frequency)
        self.assertEqual(smooth.critical_sampling_frequency_derivative, smooth2.critical_sampling_frequency_derivative)

    def test_load_not_found(self):
        """Test that loading a file that does not exist raises an error"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir) / "test.hdf5"
            with self.assertRaises(FileNotFoundError):
                smoothing.Smoothing.load_from_hdf5(tmpfile)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSmoothingInput)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestLPointProblem)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestMisc)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestCriticalSamplingFrequencyOptimization)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestLPointOptimization)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestSmoothing)
    unittest.TextTestRunner(verbosity=2).run(suite)