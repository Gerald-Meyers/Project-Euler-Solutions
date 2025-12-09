"""
<p>A palindromic number reads the same both ways. The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 × 99.</p>
<p>Find the largest palindrome made from the product of two 3-digit numbers.</p>
"""

from numpy import sum, log10, ndarray, max, zeros, vectorize
from Primes import prime_factor_multiplicity


def reverse_str_array( numpy_array: ndarray[int, str] ):
    return vectorize(lambda x : str( x )[ : : -1 ])(numpy_array.astype(str))
    
    
def is_palindromic( numpy_array: ndarray
                    ) :
    return numpy_array.astype(str) == reverse_str_array( numpy_array )



dim = int( 1e3 - 1 - 8e2 )
number_array = zeros( (dim, dim),
                      dtype=int )
for i in range( dim ) :
    for j in range( dim ) :
        number_array[ i, j ] = (8e2 + i) * (8e2 + j)

# print( is_palindromic( number_array ) )
print( max( number_array[ is_palindromic( number_array ) ]
            ) )

# for i in range( 1e5, 1e6, -1 ) :
#     if is_palindromic( i ) :
#         factors = prime_factor_multiplicity( i )
#         for split in range( sum( factors[ :, 1 ] ) ) :
#
