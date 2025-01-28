import pickle
import os
import numpy as np
from src.methods.ProcessingMethod import ProcessingMethod
from src._helpers.math import *

class MonteCarlo(ProcessingMethod):
    def __init__(self):
        self.name = 'MC'
        
    def process(self, T, num_samples):
        es = []
        xs = self.generate_sample(T, num_samples)
        for x in xs:
            es.append(self.evaluate_sample(T, x))
        return es

    def generate_sample(self, T, num_samples):
        seed = os.getpid() + int.from_bytes(os.urandom(4), 'big')
        np.random.seed(seed)
        if T.phi == 'haf':
            mean = np.zeros(T.N)
            cov = T.B.bmat
        else:
            mean = np.zeros(2*T.N)
            cov = T.B.convert_bmat_to_cov_normal()
        return np.random.multivariate_normal(mean, cov, num_samples)

    def evaluate_sample(self, T, x):
        if T.phi == 'haf':
            f = product_of_powers_single
        else:
            f = product_of_powers_double
        return T.a0 + np.sum([aI * f(x, I) for I, aI in T.enumerate_all_aI()])
