from lib.types import *
from numpy import any, arange, array, concatenate, ndarray, ones_like


def is_prime(n: Integer) -> Boolean:
    """
    Check the prime status of an integer n by iterating through all
        possible factors less than or equal to the square root of n and
        checking for divisibility using the modulus operator.

    The modulus returns zero if and only if the integer n is a multiple
        of the common_factor.
    All non-integer reals are not prime, because there is no integer
        which returns a zero modulus with any irrational or non-integer
        rational number.

    This algorithm takes advantage of Numpy's ability to speed up
        calculations and drammatically simplifies the logic.
    There is no need for a for loop to check if any interger is
        divisible by a possible common_factor.

    :param n: The integer that the prime status of is unknown
    :type n: Integer
    :return: Returns the prime status of an integer.
    :rtype: Boolean
    """
    return any(n % arange(2, n ** (1 / 2) + 1, dtype=int) == 0)


def check_prime_in_sieve(n: Integer) -> Boolean:

    return any(n == Sieve_of_Eratosthenes(n))


def Sieve_of_Eratosthenes(
    quantity_of_primes: Integer | Scalar = 1e7,
) -> NumpyIntegerArray:
    """

    The Sieve of Eratosthenes is an algorithm that generates all primes
    by successive removal of composite numbers.

    :param quantity_of_primes: Description
    :type quantity_of_primes: Scalar
    :return: Description
    :rtype: ScalarArray
    """
    # Use the positive integers here since this makes the code match the
    # logic of the process without having to shift the indic
    integers: NumpyIntegerArray = arange(0, quantity_of_primes + 1, dtype=int)

    # A list indicating the current prime status of each with zero and one
    prime_check = ones_like(integers, dtype=bool)
    # Zero and one are not prime.
    # Zero, because all multiples of zero are zero.
    # One, because if it was, then all other integers would not be prime.
    prime_check[:2] = False
    # Mark all multiples of two as composite.
    prime_check[4::2] = False

    # Begin main logic by iterating through the rest of the set of integers
    for val in integers[3:]:
        # If the value is true, then it is not a multiple of anything
        # seen so far.
        if prime_check[val]:
            # All multiples of val are composite, except val.
            prime_check[val * 2 :: val] = False

    return integers[prime_check]


def prime_factors(
    n: Integer,
) -> NumpyIntegerArray:
    primes: NumpyIntegerArray = Sieve_of_Eratosthenes(n)
    prime_check: BooleanArray = n % primes == 0
    return primes[prime_check] if any(prime_check) else array([n])


def multiplicity(integer: Integer, common_factor: Integer) -> Integer:
    """
    Recursively determine the multiplicity of a common_factor in an integer by
    dividing the integer by the common_factor with integer division (floor
    division) and returning 1 if the integer is divisible, and zero if
    not.

    :param integer: The integer being divided by common_factor.
    :type integer: Integer
    :param common_factor: The common_factor with unknown multiplicity in integer.
    :type common_factor: Integer
    :return: The number of times common_factor divides integer.
    :rtype: Integer
    """
    assert (
        common_factor > 1
    ), f"i must be greater than 1. Instead common_factor={common_factor}."
    return (
        0
        if integer % common_factor != 0
        else 1 + multiplicity(integer // common_factor, common_factor)
    )


def prime_factor_multiplicity(
    integer: Integer,
) -> dict[Integer, Integer]:
    return {
        common_factor: multiplicity(integer, common_factor)
        for common_factor in prime_factors(integer)
    }


def disjoint_factors_multiplicity(
    set_of_factored_integers: Sequence[dict[Integer, Integer]],
) -> dict[Integer, Integer]:
    """
    Take a Sequence of dictionaries, with each dictionary representing
    the set of prime factors that compose the Integer; then, find the
    set of keys which are common to both, and take the multiplicity to
    be the lowest of the two.

    Completely disjoint
    e.g. 5={5:1}, 4={2:2}
    GCF({5:1},{2:2})={1:1}=1

    Partially disjoint
    e.g. 20={5:1,2:2}, 30={2:1,3:1,5:1}
    GCF([{5:1,2:2},{5:1}])={5:1}=5

    Not-disjoint
    e.g. 50={5:2,2:1}, 40={5:1,2:3}
    GCF([{5:2,2:2},{5:1,2:3}])={5:1,2:1}=10

    :param set_of_factored_integers: Description
    :type set_of_factored_integers: Sequence[dict[Integer, Integer]]
    :return: Description
    :rtype: dict[Integer, Integer]
    """
    # Find all common factors
    common_factor_set: set[Integer] = set([1]) | set(set_of_factored_integers[0].keys())
    for factor_dict in set_of_factored_integers[1:]:
        # Intersect common_factor_set with the set of all keys for each
        # factored integer set
        common_factor_set &= set(factor_dict.keys())

    # Extract the smallest multiplicity from all factor_dicts
    common_factors: dict[Integer, Integer] = {1: 1}
    for common_factor in common_factor_set:
        common_factors[common_factor] = min(
            [
                factor_dict.get(common_factor, 0)
                for factor_dict in set_of_factored_integers
            ]
        )

    return common_factors


def greatest_common_factor(integers: IntegerArray) -> Integer:
    return multiply_factors(
        disjoint_factors_multiplicity(
            [prime_factor_multiplicity(integer) for integer in integers]
        )
    )


def union_factors_multiplicity(
    set_of_factored_integers: Sequence[dict[Integer, Integer]],
) -> dict[Integer, Integer]:
    """
    Take a Sequence of dictionaries, with each dictionary representing
    the set of prime factors that compose the integer; then, union these
    dictionaries on the keys, but keep the values which is largest.

    Completely disjoint
    e.g. 5={5:1}, 4={2:2}
    LCM([{5:1},{2:2}])={5:1,2:2}=20

    Partially disjoint
    e.g. 20={5:1,2:2}, 30={2:1,3:1,5:1}
    LCM([{5:1,2:2},{5:1}])={2:2,3:1,5:1}=60

    Not-disjoint
    e.g. 50={5:2,2:1}, 40={5:1,2:3}
    LCM([{5:2,2:1},{5:1,2:3}])={5:2,2:3}=200

    :param set_of_factored_integers: Sequence of dictionaries
    :type set_of_factored_integers: Sequence[dict[Integer, Integer]]
    :return: Dictionary representing the union of the dictionaries
    :rtype: dict[Integer, Integer]
    """
    # Return dict, contains so factors. The key:value=1:1 is merely a
    # redundant placeholder and can be removed without issue.
    common_factors: dict[Integer, Integer] = {1: 1} | set_of_factored_integers[0]

    # Iterate through all provided Integers
    for factor_dict in set_of_factored_integers[1:]:
        # Iterate through each key in an Integer dict
        for common_factor in factor_dict.keys():
            # If common_factor has more multiplicity than in common_factors,
            # assign key to common_factors with the higher multiplicity.
            if factor_dict.get(common_factor, 0) > common_factors.get(common_factor, 0):
                common_factors[common_factor] = factor_dict.get(common_factor, 0)

    return common_factors


def multiply_factors(factor_set: dict[Integer, Integer]) -> Integer:
    product: Integer = 1
    for common_factor in factor_set.keys():
        product *= common_factor ** factor_set.get(common_factor, 0)

    return product


def least_common_multiple(integers: IntegerArray) -> Integer:

    return multiply_factors(
        union_factors_multiplicity(
            [prime_factor_multiplicity(integer) for integer in integers]
        )
    )
