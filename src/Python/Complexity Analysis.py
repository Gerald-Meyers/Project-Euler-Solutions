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
    def __init__(
        self,
        func: Callable[..., Any],
        iteration_input: Iterable[Any],
        number: int = 1_000,
        *args: Any,
        **kwargs: Any
    ):
        self.func = func
        self.number = number
        self.args = args
        self.kwargs = kwargs

    def _time(self) -> float:

        return timeit(
            partial(self.func, , )
            setup="gc.enable()",
            number=self.number,
            globals=globals()
        )

    def _space(self) -> float:
        ...

    def measure_time(self) -> float:
        ...

    def measure_space(self) -> float:
        ...

    def time_graph(self) -> None:
        ...

    def space_graph(self) -> None:
        ...
