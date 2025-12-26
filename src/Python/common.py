from typing import Any, Callable, Iterable, Optional, Union

from numpy import append, array
from numpy import concatenate as concat
from numpy import (diag, einsum, float16, float32, float64, float128, hsplit,
                   int_, ndarray, newaxis, row_stack, split, sqrt, stack, sum,
                   vstack, where)
from numpy.linalg import norm
from numpy.typing import NDArray

PythonScalars: tuple[type] = (int, float)
PythonScalar: type = Union[*PythonScalars]
PythonArray = Iterable[PythonScalar]

FloatScalars = (
    int_,
    float16,
    float32,
    float64,
    # float80,
    # float96,
    float128,
    # float256
)
NumpyScalar: type = Union[*FloatScalars]
NumpyArray: type = NDArray[NumpyScalar]
CallableScalar: type = Callable[[PythonArray | NumpyArray],
                                PythonScalars | NumpyScalar]
