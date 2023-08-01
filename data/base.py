from abc import ABC, abstractmethod

import numpy as np


class DataBaseClass(ABC):

    @abstractmethod
    def get_data(self, num: int) -> np.array:
        pass

    @property
    def n_bit(self):
        return self._n_bit
