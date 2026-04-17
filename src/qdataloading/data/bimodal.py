import random

import numpy as np

from .base import DataBaseClass

class Bimodal(DataBaseClass):

    def __init__(self, n_bit: int, mu1: float, sigma1: float, mu2: float, sigma2: float, aux_bit=0):
        self._n_bit = n_bit
        self._dist_property = 'dense'
        self.aux_bit = aux_bit
        self.range = 2 ** n_bit
        self.mu1, self.sigma1, self.mu2, self.sigma2 = mu1, sigma1, mu2, sigma2
        if aux_bit:
            self.name = f'bimodal {n_bit} - {aux_bit}'
        else:
            self.name = f'bimodal {n_bit}'

    def get_data(self) -> np.array:
        
        k = np.linspace(0, self.range, self.range * 2**self.aux_bit, endpoint=False)
        normal_pmf = lambda k, mu, sigma: 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(k-mu)**2 / (2 * sigma**2))
        
        p = normal_pmf(k, self.mu1, self.sigma1) + normal_pmf(k, self.mu2, self.sigma2)
        return p / np.sum(p)