from .base import DataBaseClass

import numpy as np
from PIL import Image
import os

class RealImage(DataBaseClass):

    def __init__(self, n_bit, filename, remapped=False):
        self._n_bit = n_bit
        self._dist_property = 'dense'
        self.filename = filename
        self.remapped = remapped
        
        # Determine name from filename if not explicitly provided
        base_name = os.path.basename(filename).split('.')[0]
        if not remapped:
            self.name = base_name.replace('_', ' ')
        else:
            self.name = base_name.replace('_', ' ') + " (R)"

    def get_data(self) -> np.array:
        image = Image.open(self.filename)
        image = np.array(image.convert('L'))
        image = image.flatten()
        if self.remapped:
            sorted_indices = np.argsort(image)
            self.inverse_indices = np.argsort(sorted_indices)
            image = image[sorted_indices]
            
        return image / np.sum(image)
