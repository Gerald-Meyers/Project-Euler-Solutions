"""
If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9. 
The sum of these multiples is 23.
Find the sum of all the multiples of 3 or 5 below 1000.
"""
from collections.abc import Sequence

from ComplexityAnalysis import ComplexityAnalysis, ComplexityGraph
from numpy import asarray, inf, linspace, log, power


def MlnN_factor_sum(
        integers: Sequence[int],
        max_value: int = 1000,) -> int:
    """
    This is a simple function that takes advantage of the fact that sets
        only contain unique elements.

    Time Complexity:
    Let I be the set of provided integers.
    Let N = len(integers) and M = max_value.
    Then for each n∈I, there are at most M/n multiples of n.

    The total number of integers being summed is T, and there are at most 
        T   = T1 + T2 + ... TN = M/n1 + M/n2 + ... + M/nN 
    multiples to be summed.
    Simplifying, 
        T   = M(1/n1 + 1/n2 + ... + 1/nN) 
            = M sum_{n∈I}^M 1/n.

    In a worst case scenario, I is the set of integers contains every
        integer below N; that is I = [1, 2, ..., N].
    Here, the total is approximated by the harmonic series:
        T   = M sum_{n=1}^N 1/n 
            = M int_1^M 1/x dx 
            = M*ln(N)

    ∴ Therefore the time complexity is bounded from above by 
        O(M*ln(N)).

    There is another caveat, python has to check the uniqueness of each 
        element during a union operation, so in addition to the sum 
        operation taking O(M*ln(N)) time, the union operation also take 
        O(M*ln(N)) time.
    This just adds a factor of 2 to the time complexity, and can be 
        neglected.

    Space Complexity:
    Since this algorithm stores all mutiples of each integer below 
        max_value then updates the set of multiples, each individual
        set contains M/n elements.
    If each set were stored separately and then unioned, there would be 
        M/n1 + M/n2 + ... + M/nN elements.

    In the same worst case scenario, there would be O(M*ln(N)) space 
        complexity.


    :param integers: The list of integers to find all multiples of below max_value.
    :type integers: list[int]
    :param max_value: The bounding value to find the multiples of integers.
    :type max_value: int
    :return: The set of all multiples of integers below max_value.
    :rtype: int
    """

    assert isinstance(integers, Sequence), "Provide a list of integers."

    multiples_set: set[int] = set()
    for n in integers:
        multiples_set |= {
            # starting from n, form a set of all integers
            i for i in range(n, max_value, n)
        }

    # print(multiples_set)
    return sum(multiples_set)


def MxN_broken_mutiple_sum(
        integers: Sequence[int],
        max_value: int = 1000,) -> int:
    """
    This is a simple function that iterates through each integer in the 
    set of integers below max_value, and checking if it is divisible by 
    any of the integers in the set of integers.

    Algorithmic Complexity:
    Let N = len(integers) and M = max_value.
    For each integer below M

    :param integers: The list of integers to find all multiples below max_value.
    :type integers: list[int]
    :param max_value: The bounding value to find the multiples of integers.
    :type max_value: int
    :return: The set of all multiples of integers below max_value.
    :rtype: int
    """

    total_sum: int = 0
    for i in range(0, max_value):
        for n in integers:
            if i % n == 0:
                total_sum += i
                # this will multi-count mutiples for which n is
                # divisible by two or more elements
                break  # exit inner for loop

    return total_sum


if __name__ == "__main__":

    argument_values = asarray(power(linspace(1, 4, 20), 10),
                              dtype=int)

    timings = ComplexityAnalysis(
        func=MlnN_factor_sum,
        default_args={"integers": [3, 5],
                      "max_value": 1000},
        iteration_parameter=("max_value", argument_values)
    ).measure_time()

    ComplexityGraph(
        time_data=timings
    ).time_graph(argument_values=argument_values,
                 model_information={"time_fitting_function": (lambda m, n, c: m*log(n) + c),
                                    "model_name": "{:.2f}x + {:.2f}"},
                 matplotlib_kwargs={"plot_title": "Time Complexity Graph with linear fit.",
                                    # "plot_xscale": "log",
                                    #  "plot_yscale":"log",
                                    "plot_xlabel": "Value of max_value",
                                    "plot_ylabel": "Time taken (s)",
                                    "save_file": "PE1 Time Complexity Graph.png"},
                 curve_fit_kwargs={"bounds": ([-inf, 0], [inf, inf]),
                                   "p0": (1, 1)})
