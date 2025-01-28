import math
import pickle
import numpy as np
from collections import defaultdict
from src._helpers.math import *
from src._helpers.random import *
from src._helpers.check import *
from src.utils.SingleTuple import SingleTuple
from src.utils.CovMat import CovMat
import pprint
import itertools

class CoeffDict(defaultdict):
    """
    Class of a tuple list
    """
    def __init__(self, N, K, B, phi, a0):
        # Check if N and K are positive integers
        if not (isinstance(N, int) and N > 0):
            raise ValueError("N must be a positive integer")
        if not (isinstance(K, int) and K > 0):
            raise ValueError("K must be a positive integer")
        
        # Check if B is an instance of CovMat
        if not isinstance(B, CovMat):
            raise ValueError("B must be an instance of CovMat")
        
        # Check if phi is either 'haf' or 'hafsq' or noise
        if phi not in ['haf', 'hafsq', 'noise']:
            raise ValueError("phi must be either 'haf' or 'hafsq' or 'noise'")
        
        self.N = N     # dimension of the integral
        self.K = K     # order of multivariate polynomial
        self.B = B     # single block covariance matrix in GE
        self.phi = phi # choice of 'haf' or 'hafsq' or 'noise'
        self.a0 = a0
        self.data = self._initialize_dict()
        self._check_phi()

    def __reduce__(self):
        # Return a tuple with the class, the arguments to pass to __init__, and the instance's state
        return (self.__class__, (self.N, self.K, self.B, self.phi, self.a0), self.__dict__)
    
    def __setstate__(self, state):
        # Update the instance's __dict__ with the unpickled state
        self.__dict__.update(state)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
          return pickle.load(f)

    @property
    def coeffnum(self):
        num = 0
        for key in list(self.data.keys()):
            num_to_add = int(len(self.data[key]))
            num = num_to_add + num
        return num

    @property
    def gt(self):
        return self.querysum('aIphi') + self.a0

    def _check_phi(self):
        if self.phi not in {'haf', 'hafsq'}:
            raise ValueError("Invalid phi value. Must be 'haf', or 'hafsq'")


    def _initialize_dict(self):
        """
        Initialize the data dictionary using the
        following hierarchical structure
        # Level 1: Initialize cells 0, 2, ..., 2K
        # Level 2: tuples of sum 2*i 
        # Level 3: SingleTuple class
        """
        data_dict = {2 * i: {} for i in range(1, self.K + 1)}
        return data_dict

    def populate(self):
        completed_tasks = 0
        total_tasks = (2 * self.K + 1) ** self.N
        print('total', total_tasks)
        for I in itertools.product(range(2 * self.K + 1), repeat=self.N):
            self._process_single_I_populate(I)
            completed_tasks += 1
            print(f"{completed_tasks} tasks completed", end="\r")

    def populate_special_haf(self):
        N = self.N
        completed_tasks = 0
        for k in self.data.keys(): # note: keys are from 2, 4, ..., 2K
            s = k // N
            r = k % N
            base_tuple = [s] * N # create the base tuple of s's
            for positions in itertools.combinations(range(N), r):
                binary_tuple = [0] * N # Create a tuple of 0's
                # Set the 1s at the chosen positions
                for pos in positions:
                    binary_tuple[pos] = 1
                I = tuple(map(lambda x, y: x + y, binary_tuple, base_tuple))
                self._process_single_I_populate(I)
                completed_tasks += 1
                print(f"{completed_tasks} tasks completed", end="\r")

    def _process_single_I_populate(self, I):
        k = sum(I)
        if k % 2 == 0 and k <= 2 * self.K and k > 0:
            if I not in self.data[k]:
                self.data[k][I] = SingleTuple(I, self.B, self.phi)

    def coeffnumlist(self):
        rslt = []
        for key in list(self.data.keys()):
            num_to_add = int(len(self.data[key]))
            rslt.append(num_to_add)
        return rslt

    def querysum(self, propertyname):
        """
        Sum the phival, prob, and aIphi values over all k
        """
        total_sum = 0
        for key in self.data.keys():
            result = self._process_single_querysum(key, propertyname)
            total_sum = total_sum + result
        return total_sum

    def querysumlist(self, propertyname):
        """
        Sum the phival, prob, and aIphi values over all k and display the for each k in a list
        """
        if self.phi == 'noise':
            data_dict = {i: {} for i in range(1, self.K + 1)}
        else:
            data_dict = {2 * i: {} for i in range(1, self.K + 1)}
        for key in self.data.keys():
            data_dict[key] = self._process_single_querysum(key, propertyname)
        return data_dict

    def _process_single_querysum(self, key, propertyname):
        if propertyname == 'prob':
            return sum(tuple_data.prob for tuple_data in self.data[key].values() if tuple_data.prob is not None)
        elif propertyname == 'phival':
            return sum(tuple_data.phival for tuple_data in self.data[key].values() if tuple_data.phival is not None)
        elif propertyname == 'aI':
            return sum(tuple_data.aI for tuple_data in self.data[key].values() if tuple_data.aI is not None)
        elif propertyname == 'aIIfac':
            return sum(tuple_data.aI * tuple_data.ifac for tuple_data in self.data[key].values() if tuple_data.aI is not None)
        elif propertyname == 'aIsqIfac':
            return sum(tuple_data.aI**2 * tuple_data.ifac for tuple_data in self.data[key].values() if tuple_data.aI is not None)
        elif propertyname == 'aIphi':
            return sum(tuple_data.aIphi for tuple_data in self.data[key].values() if tuple_data.aIphi is not None)
        elif propertyname == 'qhaf':
            return sum(tuple_data.aI * tuple_data.ifac / tuple_data.phival for tuple_data in self.data[key].values() if tuple_data.phival is not None)
        else:
            raise ValueError("Unknown propertyname: {}".format(propertyname))

    def querysinglekey(self, key, propertyname):
        if propertyname == 'prob':
            return  zip(*[(tuple_data.I, tuple_data.prob) for tuple_data in self.data[key].values()])
        elif propertyname == 'phival':
            return  zip(*[(tuple_data.I, tuple_data.phival) for tuple_data in self.data[key].values()])
        elif propertyname == 'aI':
            return  zip(*[(tuple_data.I, tuple_data.aI) for tuple_data in self.data[key].values()])
        elif propertyname == 'aIphi':
            return  zip(*[(tuple_data.I, tuple_data.aIphi) for tuple_data in self.data[key].values()])
        else:
            raise ValueError("Unknown propertyname: {}".format(propertyname))

    def enumerate_all_aI(self):
        for key in self.data.keys():
            for I in self.data[key].keys():
                yield (I, self.data[key][I].aI)

    def enumerate_all_aI_w_sign(self):
        for key in self.data.keys():
            for I in self.data[key].keys():
                yield (I, self.data[key][I].aI, np.sign(self.data[key][I].phival))

    def enumerate_all_pI(self):
        keys, weights = [], []
        for key in self.data.keys():
            for I, tuple_data in self.data[key].items():
                keys.append(I)
                weights.append(tuple_data.prob)
        return keys, weights

    def update_all(self, propertyname):
        completed_tasks = 0
        for key in self.data.keys():
            for I in self.data[key].keys():
                self._update_single_I(I, propertyname)
                completed_tasks += 1
                print(f"{completed_tasks} tasks completed", end="\r")

    def _update_single_I(self, I, propertyname):
        k = sum(I)
        if propertyname == 'prob':
            if self.data[k][I].prob is None:
                self.data[k][I].compute_prob()
        elif propertyname == 'phival':
            if self.data[k][I].phival is None:
                self.data[k][I].compute_phival()
        elif propertyname == 'aIphi':
            if self.data[k][I].aIphi is None:
                self.data[k][I].compute_aIphi()
        else:
            raise ValueError("Invalid propertyname: {}".format(propertyname))
            
    def update_coeffs(self, method='random', param=0):
        if method == 'random':
            for key in self.data.keys():
                for I in self.data[key].keys():
                    self.data[key][I].aI = np.random.uniform(-1, 1)
        else:
            for key in self.data.keys():
                for I in self.data[key].keys():
                    if method == 'hafwithr':
                        self._process_single_coeff_haf_with_r(I, param)
                    elif method == 'hafsqwithr':
                        self._process_single_coeff_hafsq_with_r(I, param)
                    else:
                        self._process_single_coeff_special(I, noise_level=param)
        self.coeff_read = True

    def update_scaled_bmat(self, new_B, t):
        '''
        Right now, this only works for GBS-LI
        '''
        old_B = self.B
        old_d = old_B.d
        self.B = new_B # assign in the system a new B
        for key in self.data.keys():
            for I in self.data[key].keys():
                aI = self.data[key][I].aI
                phival = self.data[key][I].phival
                prob = self.data[key][I].prob
                if self.phi == 'hafsq':
                    self.data[key][I].aI = aI * t**(-2*self.K)
                    self.data[key][I].phival = phival * t**(2*self.K)
                    self.data[key][I].prob = prob * t**(2*self.K) * new_B.d / old_d
                else:
                    self.data[key][I].aI = aI * t**(-1*self.K)
                    self.data[key][I].phival = phival * t**(self.K)
                    self.data[key][I].prob = prob * t**(2*self.K) * new_B.d / old_d
                

    def  _process_single_coeff_haf_with_r(self, I, r):
        '''
        For each I, assign the weights to be 1/ | I : I! = mk| * the corresponding upper bound of the theorem 
        '''
        gamma_beta, q_beta, scale_const = r
        key = sum(I)
        re = key % self.N
        v = math.comb(self.N, re)
        l = int(key/2)
        tupledata = self.data[key][I]
        self.data[key][I].aI = scale_const * (l**q_beta) * (gamma_beta**l) / (v * math.factorial(l))

    def  _process_single_coeff_hafsq_with_r(self, I, r):
        '''
        For each I, assign the weights to be 1/ | I : I! = mk| * the corresponding upper bound of the theorem 
        '''
        gamma_beta, q_beta, scale_const = r
        key = sum(I)
        re = key % self.N
        v = math.comb(self.N, re)
        l = int(key/2)
        tupledata = self.data[key][I]
        self.data[key][I].aI = scale_const * (l**q_beta) * (gamma_beta**l) / (v * math.factorial(key)) 
    
    def print_log(self, path_to_log):
        with open(path_to_log, 'w') as f:
            # Redirect print statements to the text file
            log_message('Logging started...', f)
            log_message('========================', f)
            log_message('Basic information...', f)
            log_message(f'N: {self.N}', f)
            log_message('Bmat: ', f)
            log_message(f'{self.B.bmat}', f)
            log_message(f'd: {self.B.d}', f)
            log_message(f'dinv: {self.B.dinv}', f)
            log_message(f'K: {self.K}', f)
            log_message(f'phi: {self.phi}', f)
            log_message(f'gt: {self.gt}', f)
        print('Logging complete.')

########################################################
# utility function
########################################################
def generate_combinations(N, K, max_value):
    """
    Generate all combinations of length N where the elements sum to 2*K
    and each element is in the range [0, max_value].
    """
    def backtrack(current_tuple, remaining_sum, remaining_elements):
        # Base case: If we've filled N elements and sum is exactly 2*K
        if remaining_elements == 0:
            if remaining_sum == 0:
                yield tuple(current_tuple)
            return
        
        # Try adding numbers to the current tuple
        for i in range(min(remaining_sum, max_value) + 1):
            current_tuple.append(i)
            yield from backtrack(current_tuple, remaining_sum - i, remaining_elements - 1)
            current_tuple.pop()

    return backtrack([], 2 * K, N)
