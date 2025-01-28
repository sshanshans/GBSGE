import pickle
import os
import numpy as np
import uuid
from os.path import join
from multiprocessing import Pool
from src._helpers.check import check_and_create_folder
from src._helpers.gensam import *
from src._helpers.math import *

class sEstimator():
    def __init__(self, T, path_to_samples):
        self.T = T
        self.path_to_samples = path_to_samples
        self.samples = None
        self.estimates = []
        self.processing_method = None

    def set_processing_method(self, method):
        self.processing_method = method
        
    def compute_estimates(self, T, num_samples, num_tasks, step_size):
        es = [] # reset estimates, just to keep single thread
        # initialize the count table
        count_dict = create_default_nested_dict(T.data, 0)
        count_a0 = 0
        d_inv = T.B.dinv
        for i in range(num_tasks):
            single_task_sample_size = int(num_samples / num_tasks)
            if self.processing_method in ['GBSP', None]:
                xs = generate_sample(T, single_task_sample_size)
            elif self.processing_method == 'GBSLP':
                xs = generate_sample_at_level_k(T, single_task_sample_size)
            else:
                raise ValueError("processing method must be set to be GBSP or GBSLP")
            for n, x in enumerate(xs):
                current_size = i * single_task_sample_size + n
                k = sum(x)
                if ~np.isnan(k):
                    if k == 0:
                        count_a0 +=1
                    else:
                        count_dict[k][x] += 1
                # Compute the integral by summing coeff and the probability estimation (remember to divide by num_samples)
                if (current_size + 1) % step_size == 0:
                    current_estimate = np.sum([aI * sI* np.sqrt(d_inv * ifac(I) * query_single_I_from_count_dict(count_dict, I)/(current_size + 1)) for I, aI, sI in T.enumerate_all_aI_w_sign()]) + T.a0 * np.sqrt(d_inv * count_a0/(current_size + 1))
                    es.append(current_estimate)
        self.estimates.append(es)
                
    def compute_multiplicative_errors(self):
        if self.estimates is None:
            raise ValueError("Estimates have not been computed yet")
        gt = self.T.gt
        return np.abs(np.array(self.estimates) - gt)/ np.abs(gt)

    def compute_additive_errors(self):
        if self.estimates is None:
            raise ValueError("Estimates have not been computed yet")
        gt = self.T.gt
        return np.abs(np.array(self.estimates) - gt)

    def save_estimates(self, path_to_estimates):
        try:
            check_and_create_folder(path_to_estimates)
            # save estimate vals
            path_to_file = join(path_to_estimates, 'val_est.pkl')
            with open(path_to_file, 'wb') as f:
                pickle.dump(self.estimates, f)
            # save multiplicative error
            path_to_file = join(path_to_estimates, 'mul_err.pkl')
            with open(path_to_file, 'wb') as f:
                pickle.dump(self.compute_multiplicative_errors(), f)
            # save multiplicative error
            path_to_file = join(path_to_estimates, 'add_err.pkl')
            with open(path_to_file, 'wb') as f:
                pickle.dump(self.compute_additive_errors(), f)
        except Exception as e:
            print(f"Error occurred while saving samples: {e}")

def query_single_I_from_count_dict(count_dict, I):
    # note that the enumerate_all_aI does not contain a0
    k = sum(I)
    return count_dict[k][I]

def create_default_nested_dict(d, default_value):
    if isinstance(d, dict):
        return {key: create_default_nested_dict(value, default_value) for key, value in d.items()}
    else:
        return default_value