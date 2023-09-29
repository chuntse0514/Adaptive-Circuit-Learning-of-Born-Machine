import random

import numpy as np

from .base import DataBaseClass


class Bimodal(DataBaseClass):

    mu1, sigma1 = .5, 1.
    mu2, sigma2 = 3.5, .5

    def __init__(self, n_bit: int, mu1: float, sigma1: float, mu2: float, sigma2: float):
        self._n_bit = n_bit
        self.range = 2 ** n_bit
        self.mu1, self.sigma1, self.mu2, self.sigma2 = mu1, sigma1, mu2, sigma2
        self.name = f'bimodal {n_bit}'

    def get_point(self):
        return [
            np.random.normal(self.mu1, self.sigma1, 1)[0],
            np.random.normal(self.mu2, self.sigma2, 1)[0],
        ][random.randint(0, 1)]

    def get_data(self, num: int) -> np.array:
        ds = []
        while len(ds) < num:
            p = self.get_point()
            if 0 <= p < self.range - 1:
                ds.append(p)
        ds = np.round(ds).astype(int)
        ds, _ = np.histogram(ds, bins=list(range(self.range+1)), density=True)
        return ds
