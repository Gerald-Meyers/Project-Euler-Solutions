"""
A palindromic number reads the same both ways. 
The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 × 99.
Find the largest palindrome made from the product of two 3-digit numbers.
"""

# The largest product of three digit numbers is 999 * 999 = 998_001, and
# the smallest product of three digit numbers is 100 * 100 = 10_000. A
# brute force approach would be to iterate between all integers greater
# than 100 and find the largest palindrome. Beginning from the smallest
# integers is inefficient, in this already inefficient approach, so
# beggining from 999 and working down would produce an answer in
# significantly less time.

# A brute force approach could use a single array listing out all
# numbers greater than 900, two pointers progressively working through
# the list, taking the product of the pointers and checking for
# palindromicity. The question then becomes which pair of numbers
# produces the largest palindrome. A third variable stores the largest
# seen so far. This outline describes a relatively quick algorithm,
# but suffers from having to manage pointers iterating through an array
# and deciding which pointer to change and when.

# It was mentioned above that we could start at 900. This is because for
# all three digits numbers, numbers that start with 9 are all strictly
# greater than numbers which start with 8. This means that any
# palindrome which is a product of two numbers greater than 800 and less
# than 900 would be strictly less than a palindrome which is a product
# of two numbers greater than 900 and less than 1000. So any algorithm
# that solves this specific problem will have to run at most 1e4
# checks need to be performed for palindromicity. Moreover, since
# multiplication is commutative, then only half of the possible numbers
# need to be checked.

# The above comments on more efficient algorithm design are great, but
# introduce more logic and more potential sources of error. Instead,

########

# Unfortunately, there is no general method to solve the problem that
# does not involve a search. At least there is no method without the use
# of number theory which specifically locks the generated solution
# independent of "two three digit numbers". If you wanted to find the
# largest four digit number, either computational algebra systems would
# be required, or the work would have to be done manually.

# Here is a basic first attempt at solving the problem manually.
# (a * 1e0 + b  * 1e1 + c * 1e2) * (d * 1e0 + e * 1e1 + f * 1e2)
#   = ad * 1e0 ae * 1e1 + af * 1e2
#     + bd * 1e1 + be * 1e2 + bf * 1e3
#       + cd * 1e2 + ce * 1e3 + cf * 1e4
#   = ad * 1e0 + (ae + bd) * 1e1 + (af + be + cd) * 1e2
#     + (bf + ce) * 1e3 + cf * 1e4
# if the product is palindromic, then
#   ad * 1e0 + (ae + bd) * 1e1 + (af + be + cd) * 1e2
#     + (bf + ce) * 1e3 + cf * 1e4
#   =
#   cf * 1e0
#     + (bf + ce) * 1e1
#       + (af + be + cd) * 1e2
#         + (ae + bd) * 1e3
#           + ad * 1e4

# The above solution is incomplete since this is a tedious project.
# Instead the user Begoner provides a completed solution from 2005.
# https://projecteuler.net/post_id=1214


from CommonTypes import *
from numpy import arange, log10, max, ndarray, outer, vectorize, zeros
from Primes import prime_factor_multiplicity


def reverse_str_array(integer_array: StringArray | IntegerArray
                      ) -> StringArray:
    return vectorize(lambda x: str(x)[:: -1])(integer_array.astype(str))


def is_palindromic(integer_array: StringArray | IntegerArray
                   ) -> BooleanArray:
    return integer_array.astype(str) == reverse_str_array(integer_array)


number_array = arange(900, 1000)
product_array = outer(number_array, number_array)

print(max(product_array[is_palindromic(product_array)]
          ))
