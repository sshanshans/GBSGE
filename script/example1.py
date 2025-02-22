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

def polylog_minus_2(z):
    if abs(z) >= 1:
            raise ValueError("The series only converges for |z| < 1.")
    return z * (1 + z) / (1 - z)**3  - 1

def polylog_minus2_finite(x, K):
    return sum(k**2 * x**k for k in range(1, K + 1))

def compute_c1_geom(z, K):
    return 1 + 1/np.sqrt(np.pi) * np.exp(1/25 - 1/6) * geometric_series_sum(z, K)

def compute_c2_geom(z, K):
    return 1 + 1/np.sqrt(np.pi) * geometric_series_sum(z, K)

def compute_c1_geom_inf(z):
    return 1 + 1/np.sqrt(np.pi) * np.exp(1/25 - 1/6) * geometric_series_sum(z, float('inf'))

def compute_mc_geom(z, K):
    """
    Computes 1 + exp(1/25 - 1/6) * R_{1/2, K}(z)
    """
    return 1 + np.exp(1/25 - 1/6) * 1/(2 * np.sqrt(2) * np.sqrt(np.pi)) * geometric_series_sum(z, K)

def compute_gbs_finite(z, K):
    """
    Computes for N = 3, 2q_beta = 1
    """
    sk = int(np.ceil(2*K / 3))
    H = 1/2 * z**2
    p1 = 2 * np.pi * np.sqrt(3) * np.exp(3/13) * geometric_series_sum(2*z/3, 3)
    p2 = polylog_minus2_finite((z/3)**3, sk)
    return 1 + (H + p1 * p2)/np.sqrt(np.pi)
    
def compute_gbs_infinite(z):
    """
    Computes for N = 3, 2q_beta = 1
    """
    H = 1/2 * z**2
    p1 = 2 * np.pi * np.sqrt(3) * np.exp(3/13) * geometric_series_sum(2*z/3, 3)
    p2 = polylog_minus_2((z/3)**3)
    return 1 + (H + p1 * p2)/np.sqrt(np.pi)
    
def generate_latex_table():
    # Header for the LaTeX table
    table_header = """\\begin{table}[h!]
    \\centering
    \\begin{tabular}{|c|c|c|c|}
    \\hline
    $K$ & $\\mu_{\\text{hafsq}}$ & $\\text{GBS}$ & $\\text{MC}$ \\\\ 
    \\hline"""

    # Footer for the LaTeX table
    table_footer = """\\hline
    \\end{tabular}
    \\caption{Comparison of $\\mu_{\\text{hafsq}}$, MC, and GBS for different $K$.}
    \\label{tab:comparison}
    \\end{table}"""

    # Initialize LaTeX table content
    table_content = []

    N = 3
    K = 10
    exp_id = f'hafsq-N_{N}-K_{K}'
    path_to_folder = f'/work/GBSGE/exp/{exp_id}'
    with open(f"{path_to_folder}/{exp_id}.pkl", "rb") as f:
        T = pickle.load(f)

    q_alpha = 1/2
    q_beta = 1/2

    bmax = T.B.bmax
    bmin = T.B.bmin

    gamma_beta = 0.96 * N / bmax
    gamma_alpha = gamma_beta

    print('asym MC factor', 4 * gamma_alpha * bmin * bmin / (gamma_beta * bmax * bmax)**2)
    print('upper bound of UGBS', compute_gbs_infinite(gamma_beta * bmax)/ T.B.d /compute_c1_geom_inf(gamma_alpha * bmin * bmin))

    for K in range(1, 11):
        K = K * 5

        # Compute values
        c1 = compute_c1_geom(gamma_alpha * bmin * bmin, K)
        c2 = compute_c2_geom(gamma_beta * bmax * bmax, K)
        r = compute_mc_geom(4 * gamma_alpha * bmin * bmin, K)
        g = compute_gbs_finite(gamma_beta * bmax, K)
        exp_id = f'hafsq-N_{N}-K_{K}'
        path_to_folder = f'/work/GBSGE/exp/{exp_id}'

        # Load mu_haf
        with open(f"{path_to_folder}/{exp_id}.pkl", "rb") as f:
            T = pickle.load(f)
        mu_haf = T.gt

        # Format values
        mu_haf_formatted = f"{mu_haf:.6f}"
        mc_formatted = f"{r / c2**2:.4e}"
        gbs_formatted = f"{g / c1**2 / T.B.d:.4e}"

        # Append formatted row to table content
        table_content.append(rf"{K}&{mu_haf_formatted} & {gbs_formatted} & {mc_formatted} \\")

    # Combine header, content, and footer
    latex_table = "\n".join([table_header] + table_content + [table_footer])

    # Save LaTeX table to file
    with open("../rslt/example1.tex", "w") as f:
        f.write(latex_table)

    print("LaTeX table has been saved to table.tex.")


generate_latex_table()
