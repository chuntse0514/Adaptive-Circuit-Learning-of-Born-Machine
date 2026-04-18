from .log_normal import LogNormal
from .triangular import Triangular
from .bimodal import Bimodal
from .bar_and_stripes import BarAndStripes
from .real_images import RealImage
import os
import glob

# Synthetic datasets remain hardcoded as they have specific parameters
_DATA_CONFIGS = {
    'log normal 3': (LogNormal, {'n_bit': 3, 'mu': 1.0, 'sigma': 0.5}),
    'triangular 3': (Triangular, {'n_bit': 3, 'left': 0, 'mode': 2, 'right': 7}),
    'bimodal 3': (Bimodal, {'n_bit': 3, 'mu1': 1.25, 'sigma1': 1, 'mu2': 5.25, 'sigma2': 1}),

    'log normal 10': (LogNormal, {'n_bit': 10, 'mu': 5.5, 'sigma': 0.9}),
    'triangular 10': (Triangular, {'n_bit': 10, 'left': 0, 'mode': 256, 'right': 1023}),
    'bimodal 10': (Bimodal, {'n_bit': 10, 'mu1': 2 ** 10 * 2 / 7, 'sigma1': 2 ** 10 / 8, 'mu2': 2 ** 10 * 5 / 7, 'sigma2': 2 ** 10 / 8}),
    
    'log normal 10 - 1': (LogNormal, {'n_bit': 10, 'mu': 5.5, 'sigma': 0.9, 'aux_bit': 1}),
    'log normal 10 - 2': (LogNormal, {'n_bit': 10, 'mu': 5.5, 'sigma': 0.9, 'aux_bit': 2}),
    'log normal 10 - 3': (LogNormal, {'n_bit': 10, 'mu': 5.5, 'sigma': 0.9, 'aux_bit': 3}),

    'bas 2x2': (BarAndStripes, {'width': 2, 'height': 2}),
    'bas 3x3': (BarAndStripes, {'width': 3, 'height': 3}),
    'bas 4x4': (BarAndStripes, {'width': 4, 'height': 4}),
}

def _register_real_images():
    """Automatically discover and register images from the images directory."""
    data_dir = os.path.dirname(__file__)
    image_dir = os.path.join(data_dir, 'images')
    
    # Support common image formats
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    for path in image_paths:
        filename = os.path.basename(path)
        # Create a clean name (e.g., "real_image_1.jpg" -> "real image 1")
        base_name = os.path.splitext(filename)[0].replace('_', ' ')
        
        # Relative path for RealImage class to use
        rel_path = os.path.join('src/qdataloading/data/images', filename)
        
        # Register standard version
        _DATA_CONFIGS[base_name] = (RealImage, {'n_bit': 16, 'filename': rel_path})
        
        # Register remapped version
        _DATA_CONFIGS[f"{base_name} (R)"] = (RealImage, {'n_bit': 16, 'filename': rel_path, 'remapped': True})

# Initialize dynamic registration
_register_real_images()

def get_dataset(name):
    if name not in _DATA_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(_DATA_CONFIGS.keys())}")
    
    cls, kwargs = _DATA_CONFIGS[name]
    dataset = cls(**kwargs)
    dataset.name = name
    return dataset

def list_datasets():
    return sorted(list(_DATA_CONFIGS.keys()))
