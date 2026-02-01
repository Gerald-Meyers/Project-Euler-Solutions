"""
$2520$ is the smallest number that can be divided by each of the numbers
from $1$ to $10$ without any remainder.

What is the smallest positive number that is
evenly divisible by all of the numbers from $1$ to $20$?
"""

# To do this by hand, a method called the 'cake method' would allow us
# to determine the LCM of all of these numbers.

from time import perf_counter

from lib.math.Primes import Sieve_of_Eratosthenes
from lib.storage.manager import ShardManager
from numpy import all
