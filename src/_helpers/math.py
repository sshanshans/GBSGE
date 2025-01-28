import numpy as np
import itertools
import math
from thewalrus._hafnian import hafnian, hafnian_repeated
from pathlib import Path

CONST_PI = math.pi
CONST_DOUBLEPI = 2 * math.pi
CONST_EXP = np.exp(1/25 - 1/6)
CONST_SQRT_PI = 1/np.sqrt(math.pi)

def haf(B):
    return hafnian(B)
    
def hafsq(B):
    return hafnian(B)**2

def haf_I(B, I):
    return hafnian_repeated(B, I)

def q_minus(q):
    return min(q, 0)

def product_of_powers_double(values, indices):
    """
    Computes the product of each value in 'values' raised to the corresponding power in 'indices'.

    Args:
    values (list): A list of values [x1, x2, ..., xn, xn+1, ..., x2n].
    indices (list): A list of indices [i1, i2, ..., in].

    Returns:
    float: The result of x1^(i1) * x2^(i2) * ... * xn^(in) * xn+1^(i1) * ... * x2n^(in).
    """
    doubled_indices = indices + indices
    
    if len(values) != len(doubled_indices):
        raise ValueError("The length of values and doubled indices lists must be the same.")

    result = 1
    for value, index in zip(values, doubled_indices):
        result *= value ** index

    return result

def product_of_powers_single(values, indices):
    """
    Computes the product of each value in 'values' raised to the corresponding power in 'indices'.

    Args:
    values (list): A list of values [x1, x2, ..., xn].
    indices (list): A list of indices [i1, i2, ..., in].

    Returns:
    float: The result of x1^(i1) * x2^(i2) * ... * xn^(in).
    """
    if len(values) != len(indices):
        raise ValueError("The length of values and indices lists must be the same.")

    result = 1
    for value, index in zip(values, indices):
        result *= value ** index

    return result

def polylog(x, s, K):
    """
    Computes the polylogarithm function up to finite sum K and order s
    Args:
    x (float): funciton input
    s (float): order of the polylog
    K (int): truncation
    
    Returns: 
    float: function value
    """
    if K < 0:
        raise ValueError("K must be a non-negative integer")
    
    if K == 0:
        return 0
    
    k = np.arange(1, K + 1)
    terms = (x**k) / (k**s)
    return np.sum(terms)

def Hi(x, q, N):
    """
    Computes hyperbolic function of finite sum up to 2k = N or (N-1 if N is odd) without the constant 1. This corresbonds to Hi_{q, N/2} in the paper.
    Args:
    x (float): funciton input
    q (float): similar to polylog s
    N (int): truncation level as given in the paper
    
    Returns: 
    float: function value
    """
    k = np.arange(1, N//2 + 1)
    # Vectorize the math.factorial function to apply it to each element
    factorial_vectorized = np.vectorize(math.factorial)
    terms = x**(2*k) * pow(k, q) / factorial_vectorized(2*k)
    return np.sum(terms)

def Gqk(z, q, N, K):
    """
    The G_q, K(z, N) function defined as in the paper
    
    Args:
    z (float)
    q (float)
    N (int)
    K (int)
    
    Returns: function value (float)
    """
    sk = compute_Sk(N, K)
    return Hi(z, q, N) +  2**((N-1)/2 - q_minus(q)) * CONST_PI**((N-1)/2) * N**(q - 1/2) * np.exp(N/13) * polylog(2*z/N, 0, N) * polylog( (z/N)**N, 1/2-N/2-q, sk)

def Rqk(z, q, K):
    """
    The R_q,K(z, N) function defined as in the paper
    
    Args:
    z (float)
    q (float)
    K (int)
    
    Returns: function value (float)
    """
    c = CONST_SQRT_PI * 1/2
    if q >= 0:
        return c * 2**(-q) * polylog(z, 1/2 - q, K)
    else:
        return c * polylog(z, 1/2 - 2*q, K)
    
def compute_Sk(N, K):
    """
    Computes SK such that 2K = N * SK + RK where RK is between 1 and N
    
    Args:
    N (int)
    K (int)
    
    Returns: 
    float: SK
    """
    RK = 2*K % N
    SK = 2*K // N
    if RK == 0:
        if SK >=1:
            SK = SK-1
            RK = N
    return SK

def compute_mk(N, k):
    """
    Computes mk defined as in the paper
    mk = (sk!)^N * (sk + 1)^rk
    with 2k = N * sk + rk
    
    Args:
    N (int)
    k (int)
    
    Returns: 
    float: mk
    """
    RK = 2*k % N
    SK = 2*k // N
    if RK == 0:
        if SK >=1:
            SK = SK-1
            RK = N
    return (math.factorial(SK))**N * (SK + 1)**RK

def compute_c1(z, q, K):
    """
    Computes c1 defined as in the paper
    
    Args:
    z (float)
    q (float) as defined in the epxression; this is usually 0
    K (int)
    
    Returns: 
    c1 (float)
    """
    return 1 + CONST_SQRT_PI * CONST_EXP * polylog(z, 1/2 -q, K)

def compute_c2(z, q, K):
    """
    Computes c2 defined as in the paper
    
    Args:
    z (float)
    q (float) as defined in the expression
    K (int)
    
    Returns: 
    c2 (float)
    """
    return 1 + CONST_SQRT_PI * polylog(z, 1/2-q, K)

def ifac(I):
    """
    Computes the product of the factorials of each element in the tuple I
    """
    return math.prod(math.factorial(i) for i in I)