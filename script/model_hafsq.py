import numpy as np
import itertools
import math
import random
from pathlib import Path
from collections import defaultdict
from src.utils.CoeffDict import CoeffDict
from src.utils.CovMat import CovMat
from src._helpers.math import *
from src._helpers.random import * 
from src._helpers.check import * 
import pprint
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run experiment with specified parameters.")
    parser.add_argument('-N', type=int, required=True, help='Number of dimension (N)')
    parser.add_argument('-K', type=int, required=True, help='Number of taylor series coeff (K)')
    args = parser.parse_args()
    
    phi = 'hafsq'
    rseed = 370203
    a0 = 1
    
    N = args.N
    K = args.K
    
    exp_id = f'hafsq-N_{N}-K_{K}'
    path_to_folder = f'/work/GBSGE/exp/{exp_id}'
    check_and_create_folder(path_to_folder)
    path_to_save = f'{path_to_folder}/{exp_id}.pkl'
    path_to_log = f'{path_to_folder}/log.txt'
    
    Bmat = generate_random_Bmat_w_bounded_positive_entries(N, randseed=rseed)
    B = CovMat(Bmat)
    bmax = B.bmax
    bmin = B.bmin
    print('bmin', bmin)
    print('bmax', bmax)
    
    q_alpha = 1/2
    q_beta = 1/2
    
    gamma_beta = 0.96 * N / bmax
    gamma_alpha = gamma_beta
    
    scale_const = 1
    
    print('Start initializing coefficient dictionary')
    
    T = CoeffDict(N, K, B, phi, a0)
    T.populate_special_haf()
    T.update_all('phival')
    T.update_all('prob')
    T.update_coeffs(method='hafsqwithr', param = (gamma_beta, q_beta, scale_const))
    T.update_all('aIphi')
    
    # set gt
    print('gt', T.gt)
    
    # check bounds
    T.save(path_to_save)
    T.print_log(path_to_log)

if __name__ == "__main__":
    main()
