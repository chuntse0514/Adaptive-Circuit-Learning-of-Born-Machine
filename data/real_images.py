from .base import DataBaseClass

import numpy as np
from PIL import Image

class RealImage(DataBaseClass):

    def __init__(self, n_bit, filename, remapped=False):
        self._n_bit = n_bit
        self._dist_property = 'dense'
        self.filename = filename
        filename = filename.replace(".", "_")
        filename = filename.replace("/", "_")
        split_filename = filename.split("_")[4:-1]
        if not remapped:
            self.name = " ".join(split_filename)
        else:
            self.name = " ".join(split_filename) + " (R)"
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
