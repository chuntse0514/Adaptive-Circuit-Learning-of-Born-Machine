from .base import DataBaseClass

import numpy as np


class BarAndStripes(DataBaseClass):

    def __init__(self, width: int, height: int):
        self._n_bit = width * height
        self.COL, self.ROW = width, height
        self.name = f'bas {width}x{height}'

    def get_data(self, num: int) -> np.array:
        indices = self.get_indices()
        data = np.random.choice(indices, size=num)
        data, _ = np.histogram(data, bins=list(range(2 ** self._n_bit + 1)), density=True)
        return data

    def get_indices(self):
        RIGHT_COL = sum(1 << (r * self.COL) for r in range(self.ROW))
        bars = [i * RIGHT_COL  for i in range(2 ** self.COL)]
        ONE_ROW = 2 ** self.COL - 1
        strips = [ONE_ROW * self.row_base(i) for i in range(2 ** self.ROW)]
        return bars + strips[1:-1]

    def row_base(self, i: int):
        s = 0
        for _ in range(self.ROW):
            s = s << self.COL
            s |= i % 2
            i = i >> 1    
        return s
