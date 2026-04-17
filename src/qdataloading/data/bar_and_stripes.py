from .base import DataBaseClass

import numpy as np


class BarAndStripes(DataBaseClass):

    def __init__(self, width: int, height: int):
        self._n_bit = width * height
        self._dist_property = 'sparse'
        self.COL, self.ROW = width, height
        self.name = f'bas {width}x{height}'

    def get_data(self) -> np.array:
        indices = self.get_indices()
        p = np.zeros(2 ** self._n_bit)
        p[indices] = 1 / len(indices)
        return p
    
    def binary_row_to_int(self, binary_row):
        return int(''.join(binary_row.astype(str)), 2)

    def get_indices(self):
        
        bitstring_stripes = np.array([list(np.binary_repr(i, self.COL)) for i in range(2**self.COL)], dtype=int) # (2 ** COL, COL) 
        stripes_pattern = np.repeat(bitstring_stripes, repeats=self.ROW, axis=0) # ((2 ** COL) * ROW, COL)
        stripes_index_bin = stripes_pattern.reshape(2 ** self.COL, self.ROW * self.COL) # (2 ** COL, ROW * COL)
        stripes_index = np.apply_along_axis(self.binary_row_to_int, 1, stripes_index_bin)

        bitstring_bars = np.array([list(np.binary_repr(i, self.ROW)) for i in range(2**self.ROW)], dtype=int).reshape(self.ROW * 2 ** self.ROW, 1)
        bars_pattern = np.repeat(bitstring_bars, repeats=self.COL, axis=1)
        bars_index_bin = bars_pattern.reshape(2 ** self.ROW, self.ROW * self.COL)
        bars_index = np.apply_along_axis(self.binary_row_to_int, 1, bars_index_bin)
        
        return stripes_index.tolist() + bars_index[1:-1].tolist() # exclude the first and last bar indices since they are repeated.