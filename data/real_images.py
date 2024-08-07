from .base import DataBaseClass

import numpy as np
from PIL import Image

class RealImage(DataBaseClass):

    def __init__(self, n_bit, filename, remapped=False):
        self._n_bit = n_bit
        self.filename = filename
        if not remapped:
            self.name = f'real image {filename[-5]}'
        else:
            self.name = f'real image {filename[-5]} (R)'
        self.remapped = remapped

    def get_data(self) -> np.array:
        image = Image.open(self.filename)
        image = np.array(image.convert('L'))
        image = image.flatten()
        if self.remapped:
            sorted_indices = np.argsort(image)
            self.inverse_indices = np.argsort(sorted_indices)
            image = image[sorted_indices]
            
        return image / np.sum(image)
    
    def get_normalize_const(self):
        image = np.squeeze(np.array(Image.open(self.filename)).reshape(1, -1))
        return np.sum(image)
