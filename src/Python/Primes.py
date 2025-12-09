from numpy import (
    arange, ndarray, ones_like,
    any, array, concatenate,
    )

def is_prime( n
              ) :
    
    return any( n % arange( 2, n ** (1 / 2),
                            dtype=int ) == 0 )

def check_sieve( n: int
                 ) :
    
    return any( n == Sieve_of_Eratosthenes( n ) )

def Sieve_of_Eratosthenes( quantity_of_primes: [ int, float ] = 1e7
                           ) -> ndarray :
    """

    The Sieve of Eratosthenes is an algorithm that generates all primes by successive removal of composite numbers

    """
    
    integer_list = arange( 0, quantity_of_primes + 1,
                           dtype=int )
    
    # A list indicating the current prime status of each with zero and one
    prime_check = ones_like( integer_list,
                             dtype=bool )
    prime_check[ : 2 ] = False
    prime_check[ 4 : : 2 ] = False
    
    ################## MAIN LOOP ##################
    for val in integer_list[ 3 : ] :
        if prime_check[ val ] :
            prime_check[ val * 2 : : val ] = False
    
    return integer_list[ prime_check ]

def prime_factors( n: int,
                   ) -> ndarray[ int ] :
    primes = Sieve_of_Eratosthenes( n - 1 )
    prime_check = n % primes == 0
    return primes[ prime_check ] if any( prime_check ) else array( [ n ] )

def multiplicity( integer: int, factor: int
                  ) -> int :
    assert factor > 1, f'i must be greater than 1. Instead i={factor}.'
    return 0 if integer % factor != 0 else 1 + multiplicity( integer // factor, factor )

def prime_factor_multiplicity( integer: int,
                               ) -> ndarray[ int ] :
    return array( [ (factor, multiplicity( integer, factor ))
                    for factor in prime_factors( integer ) ] )

# print( prime_factor_multiplicity( 18 ) )
# print( prime_factor_multiplicity( 19999 ) )
