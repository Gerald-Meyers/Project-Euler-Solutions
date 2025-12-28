"""
Determine the time and space performance of a function over a set of parameters.

This module acts as a wrapper for timeit and tracemalloc, to determine the time-cost and memory cost of running an algorithm.
The result of these timed
By passing a desired modeling function,
"""

import tracemalloc
from functools import partial
from pathlib import Path
from timeit import timeit
from typing import Any, Callable, Iterable, List

from common import *
from matplotlib.pyplot import subplots
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
        iteration_parameter: tuple[str, Sequence[Any]],
        number: int = 1_000,
        enable_gc: bool = True
    ):
        self.func = func
        self.number = number
        self.args = default_args
        self.iteration_parameter = iteration_parameter

        # Add to the setup param for timeit
        self.enable_gc: str = "gc.enable()" if enable_gc else ""
        self.setup: str = self.enable_gc

    def _time(
            self,
            kwarg: dict[str, Any]
    ) -> float:

        return timeit(
            partial(self.func, self.args | kwarg),
            setup=self.setup,
            number=self.number,
            globals=globals()
        )

    def _space(self) -> float:
        ...

    def measure_time(self) -> list[float]:
        key, values = self.iteration_parameter
        return [self._time({key: value}) for value in values]

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
            time_data: Optional[ScalarArray] = None,
            space_data: Optional[ScalarArray] = None,
    ) -> None:
        self.time_data = time_data
        self.space_data = space_data

    def time_graph(
            self,
            argument_values: ScalarArray,
            time_fitting_function: Optional[ArrayFunction] = None,
            save_file: Optional[str | Path] = None,
            plot_title: Optional[str] = None,
            plot_label: Optional[str] = None,
            plot_xlabel: Optional[str] = None,
            plot_ylabel: Optional[str] = None,
            plot_xscale: Optional[str] = None,
            plot_yscale: Optional[str] = None,
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

        (figure, plot) = subplots(1, 1, figsize=(8, 8))

        plot.scatter(argument_values, self.time_data,
                     label=plot_label)

        if time_fitting_function is not None:
            # time fitting function must take
            fit_result: tuple[NumpyScalarArray, NumpyScalarArray] = curve_fit(
                time_fitting_function,
                xdata=argument_values, ydata=self.time_data
            )
            parameter_mean, parameter_covariance = fit_result

            standard_deviation = sqrt(
                diag(parameter_covariance))
            standard_error = standard_deviation / sqrt(len(argument_values))

            plot.plot(time_fitting_function(
                argument_values, *parameter_mean),
                label="Model fit"
            )
            plot.fill_between(
                argument_values,
                time_fitting_function(
                    argument_values,
                    *(parameter_mean - standard_error)
                ),
                time_fitting_function(
                    argument_values,
                    *(parameter_mean + standard_error)
                ),
                label="Standard error: ±σ/√n"
            )

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
