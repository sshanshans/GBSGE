from src.utils.CoeffDict import CoeffDict
from src.utils.Estimator import Estimator
from src.utils.pEstimator import pEstimator
from src.utils.Plotter import Plotter
from src.methods.MonteCarlo import MonteCarlo
import pickle
import os
from os.path import join
import numpy as np
from src._helpers.check import check_and_create_folder
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run experiment with specified parameters.")
    parser.add_argument('-N', type=int, required=True, help='Number of dimension (N)')
    parser.add_argument('-K', type=int, required=True, help='Number of taylor series coeff (K)')
    args = parser.parse_args()
    
    N = args.N
    K = args.K
    
    exp_id = f'haf-N_{N}-K_{K}'
    path_to_folder = f'/work/GBSGE/exp/{exp_id}'
    with open(f"{path_to_folder}/{exp_id}.pkl", "rb") as f:
        T = pickle.load(f)
    print('gt', T.gt)
    
    total_sample_used = int(4 * 1e8)
    step_size = 1000
    
    # Monte carlo
    path_to_samples=f"{path_to_folder}/samples/MC2"
    estimator = Estimator(T, path_to_samples)
    method = MonteCarlo()
    estimator.set_processing_method(method)
    estimator.run_sampling(num_samples=total_sample_used, num_tasks=400)

if __name__ == "__main__":
    main()