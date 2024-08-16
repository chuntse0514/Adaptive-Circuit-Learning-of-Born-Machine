from .base import DataBaseClass

import numpy as np


class Triangular(DataBaseClass):
    
    def __init__(self, n_bit: int, left: int, mode: int, right: int, aux_bit=0):
        self._n_bit = n_bit
        self._dist_property = 'dense'
        self.aux_bit = aux_bit
        self.range = 2 ** n_bit
        self.left, self.mode, self.right = left, mode, right
        self.name = f'triangular {n_bit}'
        
    def tri_pmf(self, k, a, b, c):
        p = np.zeros_like(k)
        p[k<=c] = 2 * (k[k<=c]-a) / ((b-a) * (c-a))
        p[k>c] = 2 * (b-k[k>c]) / ((b-a) * (b-c))
        
        return p

    def get_data(self) -> np.array:
        
        k = np.linspace(0, self.range, self.range * 2**self.aux_bit, endpoint=False)
        p = self.tri_pmf(k, self.left, self.right, self.mode)
        return p / np.sum(p)