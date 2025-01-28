import numpy as np
import itertools
import math
import random
from pathlib import Path
from collections import defaultdict
from src.utils.CoeffDict import CoeffDict
from src.utils.CovMat import CovMat
from src.utils.Bounds import Bounds
from src._helpers.math import *
from src._helpers.random import * 
import pprint
import pickle
import matplotlib.pyplot as plt

def construct_matrix(bmax, bmin, btilde, N, m):
    """
    Constructs an N x N matrix with the specified structure.

    Args:
        bmax (float): Value to be added to the diagonal for the first m entries.
        bmin (float): Base value for all entries in the matrix.
        btilde (float): Value to be added to the diagonal for the remaining entries.
        N (int): Size of the matrix (N x N).
        m (int): Number of entries on the diagonal set to bmax.

    Returns:
        numpy.ndarray: The constructed N x N matrix.
    """
    # Start with a matrix filled with bmin
    matrix = np.full((N, N), bmin)

    # Create the diagonal values
    diagonal_values = [bmax] * m + [btilde] * (N - m)

    # Add the diagonal values to the matrix
    np.fill_diagonal(matrix, diagonal_values)

    return matrix

def compute_btilde(N, p, m, bmin):
    """
    Computes the value of btilde based on the given parameters.

    Parameters:
    N (int): Dimension of the square matrix.
    p (float): Parameter to calculate q.
    m (int): Number of diagonal elements to add bmax.
    bmin (float): Base value for all elements in the matrix.

    Returns:
    float: Computed value of btilde.
    """
    q = p / 2
    tau = N**(q / (m - N))
    print('tau', tau)
    btilde = bmin + np.sqrt(1 - tau**2)
    print('btilde',btilde)
    return btilde

def generate_symmetric_positive_noise(N, bmin, bmax):
    np.random.seed(3)
    noise = np.random.randn(N, N)  # Normally distributed random values
    noise = (noise + noise.T) / 2  # Symmetrize the matrix
    noise += np.abs(np.min(noise))  # Ensure minimum is positive
    max_val = np.max(noise)  # Find the maximum value of the noise
    scale_factor = 0.1 * (bmax - bmin) / max_val  # Scaling factor
    noise *= scale_factor 
    return noise - np.diag(np.diag(noise))

def compute_m(N, p):
    m = int(p/2 * np.log(N))
    print('m', m)
    return m

def create_B(N, bmax, bmin, p):
    m = compute_m(N, p)
    btilde = min(compute_btilde(N, p, m, bmin), bmax)
    return construct_matrix(bmax, bmin, btilde, N, m)

def are_eigenvalues_between_0_and_1(B):
    """
    Check if all eigenvalues of matrix B are in the range (0, 1).
    """
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(B)
    # Check if all eigenvalues are > 0 and < 1
    return np.all((eigenvalues > 0) & (eigenvalues < 1))

def geometric_series_sum(z, K):
    """
    Computes the sum of the geometric series: sum_{k=1}^K z^k for |z| < 1.

    Parameters:
    z (float or complex): The common ratio of the series, |z| < 1.
    K (int): The number of terms in the series.

    Returns:
    float or complex: The sum of the series.
    """

    if K == float('inf'):
        # Infinite sum formula for |z| < 1
        if abs(z) >= 1:
            raise ValueError("The series only converges for |z| < 1.")
        return z / (1 - z)
    else:
        # Finite sum formula
        return z * (1 - z**K) / (1 - z)

def polylog(z, q, K):
    """
    Computes the sum: sum_k=1^K (z^k / k^q)
    
    Parameters:
        z (float): The base of the exponential term.
        q (float): The exponent for k.
        K (int): The upper limit of the sum.
    
    Returns:
        float: The computed sum.
    """
    return sum( z**k / (k**q) for k in range(1, K + 1))

def Hi(z, q, K):
    """
    Computes the sum: sum_k=1^K (k^q * z^(2k) / (2k)!)
    
    Parameters:
        z (float): The base of the exponential term.
        q (float): The exponent for k.
        K (int): The upper limit of the sum.
    
    Returns:
        float: The computed sum.
    """
    return sum(k**q * z**(2*k) / math.factorial(2 * k) for k in range(1, K + 1))

def generate_latex_table_noise():
    # Header for the LaTeX table
    table_header = """\\begin{table}[h!]
    \\centering
    \\begin{tabular}{|c|c|c|c|c|}
    \\hline
    $N$ & $\\mu_{\\text{haf}}$ & $\\mu_{\\text{haf}}$ & $\\text{GBS}$ & $\\text{MC}$\\\\ 
    \\hline"""

    # Footer for the LaTeX table
    table_footer = """\\hline
    \\end{tabular}
    \\caption{Comparison of $\\mu_{\\text{haf}}$, GBS, MC for different $K$.}
    \\label{tab:comparison}
    \\end{table}"""

    # Initialize LaTeX table content
    table_content = []
    mc = []
    gbs = []
    
    # Define a range for n
    # n_values = [n * 2 for n in range(1, 15)]
    n_values = [n for n in range(5, 26)]
    
    # Populate gbs and mc with some function values
    for n in n_values:
        N = n
        K = N**2
        bmin = 0.96/N
        bmax = 1.11/N
        gamma_a = 10/21 * N
        gamma_b = 5/11 * N
        q = -N/2
        B = create_B(N, bmax, bmin, 3)
        noise = generate_symmetric_positive_noise(N, bmin, bmax)
        noise[0, N-1] = 0
        noise[N-1, 0] = 0
        B = B + noise
        if not are_eigenvalues_between_0_and_1(B):
            raise ValueError("The matrix B has eigenvalues outside the range (0, 1).")

        print(B)

        B = CovMat(B)
        dinv = B.dinv
        
        z_mc = 4 * gamma_a * bmin
        q_mc = q
        v_mc = 1 + np.exp(1/25-1/6) * (1/np.sqrt(np.pi)) * polylog(z_mc, 1/2 - 2*q_mc, K)
            
        z_gbs = np.sqrt(2 * gamma_b /bmin)
        q_gbsh = q
        h = Hi(z_gbs, q_gbsh, int(N/2))
        q_gbs = -q_gbsh - N/2 + 1/2
        v_gbs = dinv * (1 + (1/np.sqrt(np.pi)) * (h + (2 * np.pi)**((N-1)/2) * N**(q_gbsh - 1/2) * np.exp(N/13) * geometric_series_sum(2*z_gbs/N, int(N)) *  polylog( (z_gbs/N)**N, q_gbs, int(2*N))))

        z_lb = 2* gamma_a * bmin
        v_lb = 1 + (1/np.sqrt(np.pi)) * np.exp(1/25-1/6) * polylog(z_lb, 1/2 - q_mc, K)

        z_ub = 2 * gamma_b * bmax
        v_ub = 1 + (1/np.sqrt(np.pi)) * polylog(z_ub, 1/2 - q_mc, K)
        
        # Format values
        mu1_formatted = f"{v_lb:.6f}"
        mu2_formatted = f"{v_ub:.6f}"
        gbs_formatted = f"{v_gbs:.4e}"
        mc_formatted = f"{v_mc/v_gbs:.4e}"

        mc.append(v_mc/v_gbs)
        gbs.append(v_gbs)

        # Append formatted row to table content
        table_content.append(rf"{N}&{mu1_formatted} &{mu2_formatted} &  {gbs_formatted} & {mc_formatted}\\")

    # Combine header, content, and footer
    latex_table = "\n".join([table_header] + table_content + [table_footer])

    # Save LaTeX table to file
    with open("../rslt/example4noise.tex", "w") as f:
        f.write(latex_table)

    print("LaTeX table has been saved to table.tex.")

    N = np.array(n_values)
    mc = np.array(mc)
    plt.figure(figsize=(8, 6))
    plt.scatter(N, np.log(mc), color='blue', label=r'$L_{\text{Haf}}^\text{MC}$')
    plt.plot(N, N**2*np.log(1.25), color='red', label=r'${1.25}^{N^2}$')
    plt.xlabel('N')
    plt.ylabel('log(Values)')
    plt.legend()
    plt.grid(True)
    plt.savefig("../rslt/gc-haf-mc-log-n.png", bbox_inches="tight")

    N = np.array(n_values)
    gbs = np.array(gbs)
    plt.figure(figsize=(8, 6))
    plt.scatter(np.log(N), np.log(gbs), color='blue', label=r'$U_{\text{Haf}}^\text{GBS-P}$')
    plt.plot(np.log(N), np.log(N**3), color='red', label=r'$N^3$')
    plt.xlabel('log(N)')
    plt.ylabel('log(Values)')
    plt.legend()
    plt.grid(True)
    plt.savefig("../rslt/gc-haf-gbs-log-N.png", bbox_inches="tight")

generate_latex_table_noise()
