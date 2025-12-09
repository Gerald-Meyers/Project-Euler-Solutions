"""
<p>If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9. The sum of these multiples is 23.</p>
<p>Find the sum of all the multiples of 3 or 5 below 1000.</p>
"""

from collections.abc import Iterable


def factor_set(integers: [list[int], int],
               max_value=1000,
               ):
    if not isinstance(integers, Iterable):
        integers = list(integers)

    factor_set = set()
    for n in integers:
        factor_set |= {i
                       for i in range(0, max_value, n)
                       }
    print(factor_set)
    return factor_set


print(sum(factor_set([3, 5],
                     1000
                     ))
      )
