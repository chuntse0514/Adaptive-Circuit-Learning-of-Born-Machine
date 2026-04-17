from abc import ABC, abstractmethod

import numpy as np


class DataBaseClass(ABC):

    @abstractmethod
    def get_data(self) -> np.array:
        pass

    @property
    def n_bit(self):
        if hasattr(self, 'aux_bit'):
            return self._n_bit + self.aux_bit
        return self._n_bit
    
    @property
    def dist_property(self):
        return self._dist_property
