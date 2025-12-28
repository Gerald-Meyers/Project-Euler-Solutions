from typing import Any, Callable, Optional, Sequence, TypeAlias

from numpy import floating, number
from numpy.typing import NDArray

NumpyScalar: TypeAlias = floating[Any]
PythonScalar: TypeAlias = int | float
Scalar: TypeAlias = NumpyScalar | PythonScalar

NumpyScalarArray: TypeAlias = NDArray[NumpyScalar]
PythonScalarArray: TypeAlias = Sequence[PythonScalar]
ScalarArray: TypeAlias = NumpyScalarArray | PythonScalarArray

ScalarFunction: TypeAlias = Callable[[Scalar | ScalarArray], Scalar]
ArrayFunction: TypeAlias = Callable[[Scalar | ScalarArray], ScalarArray]
