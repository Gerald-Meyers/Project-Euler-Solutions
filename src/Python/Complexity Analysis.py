"""
Determine the time and space performance of a function over a set of parameters.

This module acts as a wrapper for timeit and tracemalloc, to determine the time-cost and memory cost of running an algorithm.
The result of these timed
By passing a desired modeling function,
"""

import tracemalloc
from functools import partial
from timeit import timeit
from typing import Any, Callable, Iterable, List

from common import *
from matplotlib import pyplot


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
        iteration_parameter: tuple[str, Iterable[Any]],
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
            partial(self.func, self.args | kwarg)
            setup=self.setup
            number=self.number,
            globals=globals()
        )

    def _space(self) -> float:
        ...

    def measure_time(self) -> list[float]:
        key, values = self.iteration_parameter
        return [
            self._time({key: value})
            for value in values
        ]

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
            time_data: Optional[Iterable[float]] = None,
            space_data: Optional[Iterable[float]] = None,
            time_fitting_function: Optional[Callable[[float], float]] = None,
            space_fitting_function: Optional[Callable[[float], float]] = None,
    ) -> None:
        self.time_data = time_data
        self.space_data = space_data
        self.time_fit = time_fitting_function
        self.space_fit = space_fitting_function

    def time_graph(
            self,
            argument_values: Iterable[float | int],
            save_file: Optional[str] = None
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
        :type save_file: Optional[str]
        '''
        ...

    def space_graph(self) -> None:
        ...
