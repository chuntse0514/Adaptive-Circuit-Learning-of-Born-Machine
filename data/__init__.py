from .log_normal import LogNormal
from .triangular import Triangular
from .bimodal import Bimodal
from .bar_and_stripes import BarAndStripes
from .real_images import RealImage


DATA_HUB = {
    'log normal 3': LogNormal(n_bit=3, mu=1., sigma=0.5),
    'triangular 3': Triangular(n_bit=3, left=0, mode=2, right=7),
    'bimodal 3': Bimodal(n_bit=3, mu1=1.25, sigma1=1, mu2=5.25, sigma2=1),

    'log normal 10': LogNormal(n_bit=10, mu=5.5, sigma=0.9),
    'triangular 10': Triangular(n_bit=10, left=0, mode=256, right=1023),
    'bimodal 10': Bimodal(n_bit=10, mu1=2 ** 10 * 2 / 7, sigma1=2 ** 10 / 8, mu2=2 ** 10 * 5 / 7, sigma2=2 ** 10 / 8),

    'bas 2x2': BarAndStripes(width=2, height=2),
    'bas 3x3': BarAndStripes(width=3, height=3),
    'bas 4x4': BarAndStripes(width=4, height=4),

    'real image 1': RealImage(n_bit=16, filename='./data/images/real_image_1.jpg'),
    'real image 2': RealImage(n_bit=16, filename='./data/images/real_image_2.jpg'),
    'real image 3': RealImage(n_bit=16, filename='./data/images/real_image_3.jpg'),
    
    'real image 1 (R)': RealImage(n_bit=16, filename='./data/images/real_image_1.jpg', remapped=True),
    'real image 2 (R)': RealImage(n_bit=16, filename='./data/images/real_image_2.jpg', remapped=True),
    'real image 3 (R)': RealImage(n_bit=16, filename='./data/images/real_image_3.jpg', remapped=True),
}
