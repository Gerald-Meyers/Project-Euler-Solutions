from typing import Any, Callable, Optional, Sequence, TypeAlias

from numpy import bool_, floating, int_, number, signedinteger, str_
from numpy.typing import NDArray

# Single element types
Boolean: TypeAlias = bool | bool_
String: TypeAlias = str | str_
Integer: TypeAlias = int | signedinteger[Any]
NumpyInteger = signedinteger[Any]
NumpyScalar: TypeAlias = floating[Any]
PythonScalar: TypeAlias = int | float
Scalar: TypeAlias = NumpyScalar | PythonScalar

# Array types
StringArray: TypeAlias = Sequence[str] | NDArray[str_]
BooleanArray: TypeAlias = Sequence[bool] | NDArray[bool_]
IntegerArray: TypeAlias = Sequence[int] | NDArray[signedinteger[Any]]
NumpyScalarArray: TypeAlias = NDArray[NumpyScalar]
NumpyIntegerArray: TypeAlias = NDArray[NumpyInteger]
PythonScalarArray: TypeAlias = Sequence[PythonScalar]
ScalarArray: TypeAlias = NumpyScalarArray | PythonScalarArray


# Function types
StringFunction: TypeAlias = Callable[[String | StringArray],
                                     String | StringArray]
BooleanFunction: TypeAlias = Callable[[Boolean | BooleanArray],
                                      Boolean | BooleanArray]
ScalarFunction: TypeAlias = Callable[[Scalar | ScalarArray],
                                     Scalar | ScalarArray]
