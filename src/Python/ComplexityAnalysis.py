"""
Determine the time and space performance of a function over a set of parameters.

This module acts as a wrapper for timeit and tracemalloc, to determine the time-cost and memory cost of running an algorithm.
The result of these timed
By passing a desired modeling function,
"""
import gc
import tracemalloc
from functools import partial
from pathlib import Path
from timeit import repeat, timeit
from typing import Any, Callable, Iterable, List

from common import *
from matplotlib.pyplot import subplots
from numpy import asarray, mean, std
from scipy.optimize import curve_fit


class ComplexityAnalysis:
    '''
    Docstring for ComplexityAnalysis

    The assumption of this class regarging the function being analyzed
    is that the arguments are all accessible as keyword arguments, that
    is none of the arguments are anonymous or are not strictly
    positional. All arguments must be accessible as keyword arguments.
    '''

    def __init__(
        self,
        func: Callable[..., Any],
        default_args: dict[str, Any],
        iteration_parameter: tuple[str, ScalarArray],
        number: int = 1_000,
        enable_gc: bool = True,
        repitions: int = 5
    ):
        self.func = func
        self.number = number
        self.args = default_args
        self.iteration_parameter = iteration_parameter

        # Add to the setup param for timeit
        self.enable_gc: str = "gc.enable()" if enable_gc else None
        self.setup: str = self.enable_gc

        self.repitions = repitions

    def _time(
            self,
            kwarg: dict[str, Any]
    ) -> list[float]:

        return repeat(
            partial(self.func, **(self.args | kwarg)),
            setup=self.setup,
            number=self.number,
            globals=globals(),
            repeat=self.repitions
        )

    def _space(self) -> float:
        ...

    def measure_time(
        self
    ) -> list[list[float]]:

        repeat_array: list[list[float]] = list()
        key, values = self.iteration_parameter

        for i, value in enumerate(values):
            # Repeat the function several times to get a set of data.
            repeat_array.append(self._time({key: value}))

            print(
                f"{i+1} of {len(values)}:",
                f"Completed time evaluation of {value} for parameter {key}"
            )

        return repeat_array

    def measure_space(self) -> float:
        ...


class ComplexityGraph:
    """
    Docstring for ComplexityGraph

    Generate matplotlib graphics showing the performance of a function
    with respect to variations in parameters.
    """

    def __init__(
            self,
            time_data: Optional[list[list[ScalarArray]]] = None,
            space_data: Optional[ScalarArray] = None,
    ) -> None:
        self.time_data = time_data
        self.space_data = space_data

    def time_graph(
            self,
            argument_values: ScalarArray,
            model_information: Optional[dict[str, Any]] = None,
            time_fitting_function: Optional[ArrayFunction] = None,
            save_file: Optional[str | Path] = None,
            matplotlib_kwargs: Optional[dict[str, str]] = None,
            curve_fit_kwargs: Optional[dict[str, Any]] = None,
            figure_dimension: Optional[tuple[Scalar, Scalar]] = (8, 8),
            model_name: Optional[str] = None
    ) -> None:
        '''
        Generate the time graph of a function representing the performance
        of the function over a domain of inputs. If the inputs are not
        strictly ordered, then the argument values must be mapped to an
        ordered set at call time.

        The argument_values = [1, 2, 3, 4, 5, ...] or [1e0, 1e1, 1e2, ...]
        are perfectly fine, but the
        argument_values = [(1,2),(3,4),(4,5), ...] or [1+2i, 3+4i, ...]
        must be mapped to an ordered set through some metric.

        :param self: Description
        :param argument_values: The numeric argument
        :type argument_values: Iterable[float | int]
        :param save_file: File location to save graphic.
        :type save_file: Optional[str | Path]
        '''

        assert argument_values is not None, \
            "A plot cannot be made without the arguments passed to the function."
        assert self.time_data is not None, \
            "A plot cannot be made without the time the function took to run."

        if matplotlib_kwargs is None:
            matplotlib_kwargs = dict()

        plot_title: Optional[str] = matplotlib_kwargs.get('plot_title', None)
        scatter_label: Optional[str] = matplotlib_kwargs.get(
            'scatter_label', None)
        plot_xlabel: Optional[str] = matplotlib_kwargs.get('plot_xlabel', None)
        plot_ylabel: Optional[str] = matplotlib_kwargs.get('plot_ylabel', None)
        plot_xscale: Optional[str] = matplotlib_kwargs.get('plot_xscale', None)
        plot_yscale: Optional[str] = matplotlib_kwargs.get('plot_yscale', None)

        (figure, plot) = subplots(1, 1, figsize=figure_dimension)

        # Take the timing data and calculate some statistics for each run.
        timings: list[float] = list()
        timing_arguments: ScalarArray = list()
        means: ScalarArray = list()
        deviations: ScalarArray = list()

        for i, data in enumerate(self.time_data):
            timings.extend(data)
            timing_arguments.extend([argument_values[i]] * len(data))
            means.append(mean(data))
            deviations.append(std(data))

        # Scatter the raw data
        plot.scatter(timing_arguments, timings,
                     label=scatter_label,
                     s=5,
                     c="black")
        plot.errorbar(argument_values, means, 2*asarray(deviations),
                      fmt="None")

        if model_information is not None:
            time_fitting_function = model_information.get("time_fitting_function",
                                                          None)
            assert time_fitting_function is not None, \
                "Please pass the function that scipy.curve_fit is supposed to use."

            # time fitting function must take
            fit_result: tuple[NumpyScalarArray, NumpyScalarArray] = curve_fit(
                f=time_fitting_function,
                xdata=timing_arguments, ydata=timings,
                **(curve_fit_kwargs if curve_fit_kwargs is not None else dict()))

            parameter_mean, parameter_covariance = fit_result

            # Calculate Standard Deviation
            standard_deviation = sqrt(
                diag(parameter_covariance))
            standard_error = standard_deviation / sqrt(len(argument_values))

            # Add the fitting function & standard deviation curves.
            model_name = model_information.get("model_name", None)
            plot.plot(argument_values,
                      time_fitting_function(argument_values,
                                            *parameter_mean),
                      label="Model fit" + (
                          ": " + model_name.format(*parameter_mean)
                          if model_name is not None else ""),
                      color="orange")
            plot.fill_between(argument_values,
                              time_fitting_function(
                                  argument_values,
                                  *(parameter_mean - standard_deviation)),
                              time_fitting_function(
                                  argument_values,
                                  *(parameter_mean + standard_deviation)),
                              label="Standard Deviation: ±σ",
                              alpha=.3,
                              color="red")

        plot.legend()
        if plot_xlabel is not None:
            plot.set_xlabel(plot_xlabel)
        if plot_ylabel is not None:
            plot.set_ylabel(plot_ylabel)
        if plot_title is not None:
            plot.set_title(plot_title)
        if plot_xscale is not None:
            plot.set_xscale(plot_xscale)
        if plot_yscale is not None:
            plot.set_yscale(plot_yscale)

        if save_file is not None:
            figure.savefig(save_file)

    def space_graph(self) -> None:
        ...
