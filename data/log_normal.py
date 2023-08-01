from .base import DataBaseClass

import numpy as np


class LogNormal(DataBaseClass):

    def __init__(self, n_bit: int, mu: float, sigma: float):
        self._n_bit = n_bit
        self.range = 2 ** n_bit
        self.mu, self.sigma = mu, sigma

    def get_data(self, num: int) -> np.array:
        ds = []
        while len(ds) < num:
            ds.extend([
                d
                for d in np.random.lognormal(self.mu, self.sigma, num)
                if 0 <= d <= self.range - 1
            ])
        ds = np.round(ds[:num]).astype(int)
        ds, _ = np.histogram(ds, bins=list(range(self.range+1)), density=True)
        return ds
