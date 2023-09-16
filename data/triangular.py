from .base import DataBaseClass

import numpy as np


class Triangular(DataBaseClass):

    left, mode, right = 0, 2, 7

    def __init__(self, n_bit: int, left: int, mode: int, right: int):
        self._n_bit = n_bit
        self.range = 2 ** n_bit
        self.left, self.mode, self.right = left, mode, right
        self.name = f'triangular {n_bit}'

    def get_data(self, num: int) -> np.array:
        ds = []
        while len(ds) < num:
            ds.extend([
                d
                for d in np.random.triangular(self.left, self.mode, self.right, num)
                if 0 <= d <= self.range - 1
            ])
        ds = np.round(ds[:num]).astype(int)
        ds, _ = np.histogram(ds, bins=list(range(self.range+1)))
        return ds
