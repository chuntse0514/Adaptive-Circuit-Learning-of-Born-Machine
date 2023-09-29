from .base import DataBaseClass

import numpy as np
from PIL import Image

class RealImage(DataBaseClass):

<<<<<<< HEAD
    def __init__(self, filename):
        self.filename = filename

    def get_data(self) -> np.array:
=======
    def __init__(self, n_bit, filename):
        self._n_bit = n_bit
        self.filename = filename
        self.name = f'real image {filename[-5]}'

    def get_data(self, **kwargs) -> np.array:
>>>>>>> ab16b69e607c79e6d61606b6e4ba0ad8f839a41e
        image = np.array(Image.open(self.filename))
        image = np.squeeze(image.reshape(1, -1))
        return image / np.sum(image)
    
    def get_normalize_const(self):
        image = np.squeeze(np.array(Image.open(self.filename)).reshape(1, -1))
<<<<<<< HEAD
        return np.sum(image)
        
=======
        return np.sum(image)
>>>>>>> ab16b69e607c79e6d61606b6e4ba0ad8f839a41e
