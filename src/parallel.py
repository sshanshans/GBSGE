from joblib import Parallel, delayed
import numpy as np
from piquasso._math import hafnian, permanent, linalg
from src.eval import *

def parallel_helper(I, J, aI, aJ, Bmat, d_inv, memo):
    """
    Helper function to parallelize the inner loop computation.
    """
    IJsum = tuple(a + b for a, b in zip(I, J))
    BIJ_key = tuple(sorted(IJsum))
    if BIJ_key not in memo:
        BIJ = linalg.reduce_(Bmat, IJsum)
        memo[BIJ_key] = np.real(hafnian.hafnian(BIJ))
    mc_q_contrib = aI * aJ * memo[BIJ_key]

    # Assuming BI and BJ calculations can similarly be memoized
    BI_key = tuple(sorted(I))
    BJ_key = tuple(sorted(J))
    if BI_key not in memo:
        BI = linalg.reduce_(Bmat, I)
        memo[BI_key] = np.real(hafnian.hafnian(BI))
    if BJ_key not in memo:
        BJ = linalg.reduce_(Bmat, J)
        memo[BJ_key] = np.real(hafnian.hafnian(BJ))

    gbs_q_contrib = ifac(J) * d_inv * aI * aJ * memo[BI_key] / memo[BJ_key]

    return mc_q_contrib, gbs_q_contrib

def check_q_term_optimized(Bmat, tcpair):
    covc = convert_bmat_to_covc(Bmat)
    covq = compute_covq(covc)
    d_inv = np.sqrt(np.linalg.det(covq))
    
    memo = {}  # Dictionary to memoize hafnian calculations
    results = Parallel(n_jobs=-1)(delayed(parallel_helper)(I, J, aI, aJ, Bmat, d_inv, memo)
                                  for I, aI in tcpair.items()
                                  for J, aJ in tcpair.items())

    mc_q, gbs_q = map(sum, zip(*results))
    print('mc_q', mc_q)
    print('gbs_q', gbs_q)
    return mc_q, gbs_q