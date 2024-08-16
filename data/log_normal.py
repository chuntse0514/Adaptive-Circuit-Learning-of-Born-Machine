from .base import DataBaseClass

import numpy as np


class LogNormal(DataBaseClass):

    def __init__(self, n_bit: int, mu: float, sigma: float, aux_bit=0):
        self._n_bit = n_bit
        self._dist_property = 'dense'
        self.aux_bit = aux_bit
        self.range = 2 ** n_bit
        self.mu, self.sigma = mu, sigma
        self.name = f'log normal {n_bit}'

    def get_data(self) -> np.array:
        
        k = np.linspace(0, self.range, self.range * 2**self.aux_bit, endpoint=False)
        pmf = lambda k: np.exp(-(np.log(k)-self.mu) ** 2 / (2 * self.sigma**2)) / k 
        p = np.zeros(self.range * 2**self.aux_bit)
        p[k>0] = pmf(k[k>0])
        return p / np.sum(p)
