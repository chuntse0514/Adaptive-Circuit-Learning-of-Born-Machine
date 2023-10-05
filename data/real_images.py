from .base import DataBaseClass

import numpy as np
from PIL import Image

class RealImage(DataBaseClass):

    def __init__(self, n_bit, filename):
        self._n_bit = n_bit
        self.filename = filename
        self.name = f'real image {filename[-5]}'

    def get_data(self, **kwargs) -> np.array:
        image = Image.open(self.filename)
        image = np.array(image.convert('L'))
        image = np.squeeze(image.reshape(1, -1))
        return image / np.sum(image)
    
    def get_normalize_const(self):
        image = np.squeeze(np.array(Image.open(self.filename)).reshape(1, -1))
        return np.sum(image)
