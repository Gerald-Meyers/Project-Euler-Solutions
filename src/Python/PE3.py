"""
<p>The prime factors of 13195 are 5, 7, 13 and 29.</p>
<p>What is the largest prime factor of the number 600851475143 ?</p>
"""

from numpy import max
from Primes import prime_factors

n = 600851475143

print( max( prime_factors( n ) ) )