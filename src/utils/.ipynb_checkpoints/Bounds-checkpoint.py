from src._helpers.math import *
from src._helpers.check import *
from src.utils.CovMat import CovMat
import pickle

class Bounds():
    """
    Class of bounds
    """
    def __init__(self, N, K, B, phi, gamma_alpha, gamma_beta, q_alpha, q_beta):
        # Check if N and K are positive integers
        if not (isinstance(N, int) and N > 0):
            raise ValueError("N must be a positive integer")
        if not (isinstance(K, int) and K > 0):
            raise ValueError("K must be a positive integer")
        
        # Check if B is an instance of CovMat
        if not isinstance(B, CovMat):
            raise ValueError("B must be an instance of CovMat")
        
        # Check if phi is either 'haf' or 'hafsq'
        if phi not in ['haf', 'hafsq']:
            raise ValueError("phi must be either 'haf' or 'hafsq'")
            
        self.N = N     # dimension of the integral
        self.K = K     # order of multivariate polynomial
        self.B = B     # single block covariance matrix in GE
        self.phi = phi # choice of 'haf' or 'hafsq'
        self._a0 = None
        self._gt = None
        self.gamma_alpha = gamma_alpha
        self.gamma_beta = gamma_beta
        self.q_alpha = q_alpha
        self.q_beta = q_beta
        self.c1 = self._initialize_c1()
        self.c2 = self._initialize_c2()

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def _initialize_c1(self):
        if self.phi == 'haf':
            z = 2 * self.gamma_alpha * self.B.bmin
            return compute_c1(z, self.q_alpha, self.K)
        else:
            #hafsq
            z = self.gamma_alpha * (self.B.bmin)**2
            return compute_c1(z, self.q_alpha, self.K)
    
    def _initialize_c2(self):
        if self.phi == 'haf':
            z = 2 * self.gamma_beta * self.B.bmax
            return compute_c2(z, self.q_beta, self.K)
        else: 
            #hafsq
            z = self.gamma_beta * (self.B.bmax)**2
            return compute_c2(z, self.q_beta, self.K)

    @property
    def a0(self):
        return self._a0

    @a0.setter
    def a0(self, value):
        """
        Update coeff using the user-defined value a
        """
        self._a0 = value

    @property
    def gt(self):
        return self._gt

    @gt.setter
    def gt(self, value):
        """
        Update coeff using the user-defined value a
        """
        self._gt = value

    @property
    def GBS_bound_in_expcond(self):
        if self.phi == 'haf':
            # haf
            val = self.a0 + Gqk(np.sqrt(2 * self.gamma_beta/ self.B.bmin), self.q_beta, self.N, self.K)
            return 1/self.B.d * val
        else: 
            #hafsq
            val =  self.a0 + CONST_SQRT_PI * Gqk(self.gamma_beta * self.B.bmax, 2*self.q_beta - 1/2, self.N, self.K)
            return 1/self.B.d * val

    @property
    def GBS_bound_in_expcond_new(self):
        if self.phi == 'haf':
            # haf
            val = self.a0
            for k in range(self.K):
                k = k + 1
                val = val + k**self.q_beta * (2 * self.gamma_beta/ self.B.bmin)**k / math.factorial(2*k)
            return 1/self.B.d * val
        else: 
            raise ValueError("Invalid phi value. Must be 'haf'.")

    @property
    def MC_bound_in_expcond(self):
        if self.phi == 'haf':
            # haf
            val = self.a0 + 2* CONST_EXP *  Rqk(4 * self.gamma_alpha * self.B.bmin, self.q_alpha, self.K)
        else: 
            #hafsq
            val = self.a0 + CONST_EXP**2 * Rqk(4 * self.gamma_alpha * self.B.bmin**2, self.q_alpha, self.K)
        return val

    @property
    def GBS_upper_bound(self):
        """
        Upper bound of GBS estimator (GBS-I or GBS-P) given in the paper
        for a0 to be between 1/c2 mu and 1/c1 mu
        """
        if self.a0 is None:
            raise ValueError('a0 is None. Must set a0')

        if not self._check_a0_bound():
            raise ValueError('a0 is not within the bounds. Must set gt')
            
        if self.phi == 'haf':
            # need to be updated!!
            val = self.a0 + (1/self.c1) * Gqk(np.sqrt(2 * self.gamma_beta/ self.B.bmax), self.q_beta, self.N, self.K)
        else: 
            #hafsq
            val =  1 + CONST_SQRT_PI * Gqk(self.gamma_beta * self.B.bmax, 2*self.q_beta - 1/2, self.N, self.K)
        return 1/self.B.d * (1/self.c1)**2 * val - 1
    
    @property
    def MC_lower_bound(self):
        """
        The lower bound of MC estimator given in the paper
        for a0 to be between 1/c2 mu and 1/c1 mu
        """
        if self.a0 is None:
            raise ValueError('a0 is None. Must set a0')

        if not self._check_a0_bound():
            raise ValueError('a0 is not within the bounds. Must set gt')
        
        if self.phi == 'haf':
            # need to be updated!!
            val = self.a0 + (1/self.c2) * CONST_SQRT_PI * CONST_EXP *  polylog(4 * self.gamma_alpha * self.B.bmin, 1/2 - self.q_alpha, self.K)
        else: 
            #hafsq
            val = 1 + CONST_EXP**2 * Rqk(4 * self.gamma_alpha * self.B.bmin**2, self.q_alpha, self.K)
        return val * (1/self.c2)**2 - 1

    def _debug_radius_convergence(self, file_id=None):
        if self.phi == 'haf':
            log_message('to be done', file_id)
        else: # hafsq
            log_message('==============', file_id)
            log_message('radius of convergence check', file_id)
            log_message('GBS', file_id)
            g1 = self.gamma_beta * self.B.bmax
            g2 = g1 / self.N
            g3 = (g1 / self.N) ** self.N
            log_message(f'bmax: {self.B.bmax}', file_id)
            log_message(f'bmin: {self.B.bmin}', file_id)
            log_message(f'z = gamma_beta * bmax: {g1}', file_id)
            log_message(f'z / N: {g2}', file_id)
            log_message(f'z / N pow N: {g3}', file_id)
            log_message('==============', file_id)
            log_message('MC', file_id)
            m1 = 4 * self.gamma_alpha * self.B.bmin**2
            log_message(f'4 gamma_alpha * sq(bmin): {m1}', file_id)
            log_message('mu', file_id)
            m2 = 4 * self.gamma_beta * self.B.bmax**2
            log_message(f'4 gamma_beta * sq(bmax): {m2}', file_id)

    def _debug_uniform_speedup_condition(self, file_id=None):
        # can then use the logged number for experimenting exponential speedup
        log_message(f'GBS upper bound: {self.GBS_upper_bound}', file_id)
        log_message(f'MC lower bound: {self.MC_lower_bound}', file_id)


    def check_uniform_speedup_condition(self):
        if self.GBS_upper_bound < self.MC_lower_bound:
            return True
        else:
            return False

    def compute_coeff_bounds_single_k(self, k):
        """
        Computes coefficient bounds given as in Theorem 1.3 or Theorem 1.4 for a single k.
        
        Args:
            k (int): The parameter k.
            
        Returns:
            dict: A dictionary containing the following
            BD1: lower bound for sum a_I over all I whose entry sum is 2k
            BD2: upper bound for sum a_I over all I whose entry sum is 2k
            BD3: upper bound for sum a_I^2 I! for haf (or sum a_I I! for haf) over all I whose entry sum is 2k
        """
        q_alpha = self.q_alpha
        q_beta = self.q_beta
        gamma_alpha = self.gamma_alpha
        gamma_beta = self.gamma_beta

        if self.phi == 'haf':
            BD1 = 1/self.c2 * k**q_alpha * gamma_alpha**k / math.factorial(k)
            BD2 = 1/self.c1 * k**q_beta * gamma_beta**k / math.factorial(k)
            BD3 = BD2 * compute_mk(self.N, k)   
        else: 
            #hafsq
            BD1 = 1/self.c2 * k**q_alpha * gamma_alpha**k / math.factorial(2 * k) * self.gt
            BD2 = 1/self.c1 * k**q_beta * gamma_beta**k / math.factorial(2 * k) * self.gt
            BD3 = BD2**2 * compute_mk(self.N, k)   
        return {'BD1': BD1, 'BD2': BD2, 'BD3': BD3}

    def _check_a0_bound(self):
        '''
        check if a0 is between 1/c1 mu and 1/c2 mu
        '''
        ub = 1/self.c1 * self.gt
        lb = 1/self.c2 * self.gt
        return lb <= self.a0 <= ub
        




    
