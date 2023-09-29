from .base import DataBaseClass

import numpy as np
from PIL import Image

class RealImage(DataBaseClass):

    def __init__(self, filename):
        self.filename = filename

    def get_data(self) -> np.array:
        image = np.array(Image.open(self.filename))
        image = np.squeeze(image.reshape(1, -1))
        return image / np.sum(image)
    
    def get_normalize_const(self):
        image = np.squeeze(np.array(Image.open(self.filename)).reshape(1, -1))
        return np.sum(image)
        