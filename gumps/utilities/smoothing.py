# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipy.signal
from pathlib import Path
import pandas as pd
import attrs
import gumps.common.exceptions
from gumps.common import hdf5

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.optimize import minimize

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@attrs.define
class SmoothingInput:
    df : pd.DataFrame
    x_label : str
    y_label : str
    max_samples : int = 5000
    resampled_signal: pd.DataFrame = attrs.field(init=False)
    sampling_frequency: float = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.resampled_signal = self._resample_signal()
        self.sampling_frequency = 1.0 / (self.resampled_signal['times'].values[1] - self.resampled_signal['times'].values[0])

    def _resample_signal(self) -> pd.DataFrame:
        "Resample the signal to a uniform grid"
        if len(self.df[self.x_label].values) > self.max_samples:
            return self._resample_too_many_samples()

        per = self._time_step_size_error()

        if per > 0.01:
            return self._resample_inconsistent_step_size()
        return pd.DataFrame({'times':self.df[self.x_label].values, 'values':self.df[self.y_label].values})

    def _resample_too_many_samples(self):
        times = self.df[self.x_label].values
        values = self.df[self.y_label].values
        times_resample = np.linspace(times[0], times[-1], self.max_samples)
        spline_resample = scipy.interpolate.InterpolatedUnivariateSpline(times, values, k=5, ext=3)
        values_resample = spline_resample(times_resample)

        return pd.DataFrame({'times':times_resample, 'values':values_resample})

    def _time_step_size_error(self) -> float:
        times = self.df[self.x_label].values
        diff_times = np.diff(times)
        max_time = np.max(diff_times)
        min_time = np.min(diff_times)
        per = (max_time - min_time) / min_time
        return per

    def _resample_inconsistent_step_size(self) -> pd.DataFrame:
        times = self.df[self.x_label].values
        values = self.df[self.y_label].values

        min_time = np.min(np.diff(times))

        number_of_samples = int((times[-1] - times[0]) / min_time) + 1    #+1 is to include the last time point

        times_resample = np.linspace(times[0], times[-1], number_of_samples)

        if len(times_resample) > self.max_samples:
            times_resample = np.linspace(times[0], times[-1], self.max_samples)

        times_resample[-1] = times[-1]
        spline_resample = scipy.interpolate.InterpolatedUnivariateSpline(times, values, k=5, ext=3)
        values_resample = spline_resample(times_resample)

        return pd.DataFrame({'times':times_resample, 'values':values_resample})


@attrs.define
class SmoothingOutput:
    times: np.ndarray
    signal : np.ndarray
    derivative: np.ndarray | None


@attrs.define
class LPointProblem:
    "L-point problem for smoothing signal"
    x: np.ndarray
    y: np.ndarray
    x_min: np.ndarray = attrs.field(init=False)
    y_min: np.ndarray = attrs.field(init=False)
    first_point: np.ndarray = attrs.field(init=False)
    last_point: np.ndarray = attrs.field(init=False)
    diagonal_line: np.ndarray = attrs.field(init=False)
    factor: float = attrs.field(init=False)

    def __attrs_post_init__(self):
        sort_idx = np.argsort(self.x)
        self.x = self.x[sort_idx]
        self.y = self.y[sort_idx]

        #correct for offset in x and y
        self.y_min = self.y - np.min(self.y)
        self.x_min = self.x - np.min(self.x)

        self.diagonal_line = np.array([self.x_min, self.y_min]).T

        #normalize the data
        self.factor = np.max(self.diagonal_line, 0)
        self.diagonal_line = self.diagonal_line / self.factor

        #find the two points that are furthest apart
        self.first_point = self.diagonal_line[0, :]
        self.last_point = self.diagonal_line[-1, :]

    def distance_diagonal(self, diagonal_line:np.ndarray) -> np.ndarray:
        "this finds the distance of all points to a diagonal line connecting the first and last point"
        diagonal_line = (diagonal_line - np.array([np.min(self.x), np.min(self.y)]).T)/self.factor
        return self._distance(diagonal_line)

    def _distance(self, diagonal_line: np.ndarray | None = None) -> np.ndarray:
        "this finds the distance of all points to a diagonal line connecting the first and last point"
        if diagonal_line is None:
            diagonal_line = self.diagonal_line
        distance = np.cross(self.last_point - self.first_point, self.first_point - diagonal_line) / np.linalg.norm(self.last_point - self.first_point)
        return distance

    def LPoint(self) -> tuple[float, float]:
        "find the point that is furthest away from the line"
        distance = self._distance()
        max_idx = np.argmax(distance)
        max_d = distance[max_idx]
        l_x = self.x[max_idx]
        l_y = self.y[max_idx]

        if max_d <= 0 or np.isnan(max_d):
            raise gumps.common.exceptions.NoLPointFoundError("No L-point found")

        return l_x, l_y


def signal_bessel(critical_sampling_frequency: float, sampling_frequency: float) -> np.ndarray:
    "returns a numpy array of the bessel filter coefficients in second order sections format"
    return scipy.signal.bessel(3, critical_sampling_frequency, btype="lowpass", analog=False, fs=sampling_frequency, output="sos", norm="delay")

class CriticalSamplingFrequencyOptimization(ElementwiseProblem):

    def __init__(self, lb: float, ub: float, sse_target: float, values: np.ndarray, sampling_frequency: float):
        super().__init__(n_var=1, n_obj=1, n_constr=0, xl=lb, xu=ub)
        self.sse_target = sse_target
        self.values = values
        self.sampling_frequency = sampling_frequency

    def _evaluate(self, x: np.ndarray, out: dict[str, float], *args, **kwargs):
        critical_sampling_frequency = x[0]
        critical_sampling_frequency = 10**critical_sampling_frequency
        try:
            sos = signal_bessel(critical_sampling_frequency, self.sampling_frequency)
            low_passed = scipy.signal.sosfiltfilt(sos, self.values)
            sse = np.sum((low_passed - self.values) ** 2)

            error = (sse - self.sse_target)**2
        except ValueError:
            error = np.inf
        out["F"] = error

class LPointOptimization(ElementwiseProblem):

    def __init__(self, lpp: LPointProblem, sampling_frequency: float, values:np.ndarray):
        self.lpp = lpp
        self.values = values
        lb = np.log10(lpp.x[0])
        ub = np.log10(lpp.x[-1])
        super().__init__(n_var=1, n_obj=1, n_constr=0, xl=lb, xu=ub)
        self.sampling_frequency = sampling_frequency

    def _evaluate(self, x: np.ndarray, out: dict[str, float], *args, **kwargs):
        critical_sampling_frequency = x[0]
        critical_sampling_frequency = 10.0 ** critical_sampling_frequency
        try:
            sos = signal_bessel(critical_sampling_frequency, self.sampling_frequency)
        except ValueError:
            out['F'] = np.inf
            return

        try:
            low_passed = scipy.signal.sosfiltfilt(sos, self.values)
        except np.linalg.LinAlgError:
            out['F'] = np.inf
            return

        sse = np.sum((low_passed - self.values) ** 2)

        diagonal_line = np.array([critical_sampling_frequency, np.log(sse)])

        d = self.lpp.distance_diagonal(diagonal_line)

        out["F"] = -d

class Smoothing:
    def __init__(self, params:SmoothingInput, rmse_target:float=1e-4,
                 critical_sampling_frequency:float|None=None,
                 critical_sampling_frequency_derivative:float|None=None,
                 run_initialization:bool=True):
        """Initialize the smoothing object"""
        self.params = params
        self.rmse_target = rmse_target
        self.smoothing_factor = rmse_target**2 * len(params.df)
        self.critical_sampling_frequency: float | None = critical_sampling_frequency
        self.critical_sampling_frequency_derivative: float | None = critical_sampling_frequency_derivative

        if run_initialization:
            self.initialize()

    def initialize(self):
        "initialize the smoothing"
        values = self.params.resampled_signal['values']

        #normalize the values
        values = values * 1.0/np.max(values)

        self.critical_sampling_frequency = self._find_critical_sampling_frequency(values)

        if self.critical_sampling_frequency is None:
            logger.info("bessel filter disabled, no viable L point found")

        #to do
        spline, factor = self._create_spline(self.params)

        # run a quick butter pass to remove high frequency noise in the derivative (needed for some experimental data)
        values_filter = spline.derivative()(self.params.resampled_signal.times) / factor
        factor = 1.0/np.max(values_filter)
        values_filter = values_filter * factor
        self.critical_sampling_frequency_derivative = self._find_critical_sampling_frequency(values_filter)

    def _create_spline(self, params:SmoothingInput) -> tuple[scipy.interpolate.UnivariateSpline, float]:
        "create a spline for smoothing the signal"
        times = params.resampled_signal['times']
        values = params.resampled_signal['values']
        factor = 1.0 / np.max(values)
        values = values * factor
        values_filter = self._smoothing_filter_signal(times, values, self.critical_sampling_frequency)

        return (
            scipy.interpolate.UnivariateSpline(times, values_filter, s=self.smoothing_factor, k=5, ext=3),
            factor,
        )

    def _smoothing_filter_signal(self, times: np.ndarray, values: np.ndarray, critical_sampling_frequency : float | None) -> np.ndarray:
        "smooth a signal using a bessel filter"
        if critical_sampling_frequency is None:
            return values
        fs = 1.0 / (times[1] - times[0])
        sos = signal_bessel(critical_sampling_frequency, fs)
        low_passed = scipy.signal.sosfiltfilt(sos, values)
        return low_passed

    def _critical_sampling_frequency_LPoint(self, filters:np.ndarray, sse:np.ndarray) -> float | None:
        "optimize critical sampling frequency using an L-point method"
        try:
            critical_sampling_frequency, _ = LPointProblem(filters, np.log(sse)).LPoint()
        except gumps.common.exceptions.NoLPointFoundError:
            critical_sampling_frequency = None

        return critical_sampling_frequency

    def _find_critical_sampling_frequency(self, values: np.ndarray) -> float | None:
        "optimize critical sampling frequency using an L-point method"
        filters, sse = self._sample_critical_frequency(values)

        crit_fs_max = self._find_max_signal(values, filters)

        critical_sampling_frequency = self._critical_sampling_frequency_LPoint(filters, sse)

        if critical_sampling_frequency is not None:
            critical_sampling_frequency = self._refine_signal(values, x=filters, y=np.log(sse), fs=self.params.sampling_frequency)

        if critical_sampling_frequency  is not None and crit_fs_max < critical_sampling_frequency:
            critical_sampling_frequency = crit_fs_max

        return critical_sampling_frequency

    def _sample_critical_frequency(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        "sample critical sampling frequency"
        filters = []
        sse = []

        sampling_frequency = self.params.sampling_frequency

        max_sampling_frequency = sampling_frequency / 2.0

        for i in np.logspace(-8, np.log10(max_sampling_frequency), 50):
            try:
                sos = signal_bessel(i, sampling_frequency)
                low_passed = scipy.signal.sosfiltfilt(sos, values)

                filters.append(i)
                sse.append(np.sum((low_passed - values) ** 2))
            except (ValueError, np.linalg.LinAlgError):
                continue

        return np.array(filters), np.array(sse)

    def _refine_signal(self, values: np.ndarray, x: np.ndarray, y: np.ndarray, fs: float) -> float:
        lpp = LPointProblem(x,y)

        problem = LPointOptimization(lpp, fs, values)

        algorithm = PatternSearch(n_sample_points=50, eps=1e-13)

        res = minimize(problem,
                algorithm,
                verbose=False,
                seed=1)

        crit_fs = 10**res.X[0]

        return crit_fs

    def _find_max_signal(self, values: np.ndarray, filters: np.ndarray) -> float:
        "find the max sampling frequency"
        filters = np.log10(filters)
        problem = CriticalSamplingFrequencyOptimization(filters[0], filters[-1], self.smoothing_factor, values, self.params.sampling_frequency)

        algorithm = PatternSearch(n_sample_points=50, eps=1e-13)

        res = minimize(problem,
                algorithm,
                verbose=False,
                seed=1)

        critical_sampling_frequency = 10**res.X[0]

        return critical_sampling_frequency

    def signal(self, params:SmoothingInput|None = None) -> SmoothingOutput:
        "return the smoothed signal"
        if params is None:
            params = self.params
        spline, factor = self._create_spline(params)
        signal = spline(params.df[params.x_label]) / factor
        return SmoothingOutput(times=params.df[params.x_label], signal=signal, derivative=None)

    def signal_derivative(self, params:SmoothingInput|None = None) -> SmoothingOutput:
        "return the smoothed signal and derivative"
        if params is None:
            params = self.params
        spline, factor = self._create_spline(params)
        signal = spline(params.df[params.x_label]) / factor

        values_filter_der = spline.derivative()(params.resampled_signal.times) / factor

        factor_der = 1.0/np.max(values_filter_der)
        values_filter_der = values_filter_der * factor_der

        values_filter_der = self._smoothing_filter_signal(params.resampled_signal.times, values_filter_der, self.critical_sampling_frequency_derivative)

        values_filter_der = values_filter_der / factor_der
        spline_der = scipy.interpolate.InterpolatedUnivariateSpline(params.resampled_signal.times, values_filter_der, k=5, ext=3)
        values_filter_der = spline_der(params.df[params.x_label])

        derivative = spline(params.df[params.x_label]) / factor

        return SmoothingOutput(times=params.df[params.x_label], signal=signal, derivative=derivative)

    def save_as_hdf5(self, path:Path) -> None:
        if path.exists():
            raise FileExistsError(f"File {path} already exists")

        data = hdf5.H5(path.as_posix())
        data['critical_sampling_frequency'] = self.critical_sampling_frequency
        data['critical_sampling_frequency_derivative'] = self.critical_sampling_frequency_derivative
        data['smoothing_factor'] = self.smoothing_factor
        data['rmse_target'] = self.rmse_target
        data['x'] = self.params.df[self.params.x_label].values
        data['y'] = self.params.df[self.params.y_label].values
        data.save()

    @classmethod
    def load_from_hdf5(cls, path:Path) -> "Smoothing":
        if path.exists():
            data = hdf5.H5(path.as_posix())
            data.load()
            params = SmoothingInput(pd.DataFrame({'x':data['x'], 'y':data['y']}), 'x', 'y')
            smooth = cls(params, rmse_target = data['rmse_target'],
                         critical_sampling_frequency = data['critical_sampling_frequency'],
                         critical_sampling_frequency_derivative = data['critical_sampling_frequency_derivative'],
                         run_initialization=False)
            return smooth
        else:
            raise FileNotFoundError(path.as_posix())
