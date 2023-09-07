from functools import partial

from .log_normal import LogNormal
from .triangular import Triangular
from .bimodal import Bimodal
from .bar_and_stripes import BarAndStripes


DATA_HUB = {
    'log normal': partial(LogNormal, n_bit=3, mu=1., sigma=1.),
    'triangular': partial(Triangular, n_bit=3, left=0, mode=2, right=7),
    'bimodal': partial(Bimodal, n_bit=3, mu1=0.5, sigma1=1., mu2=3.5, sigma2=0.5),

    'log normal 10': partial(LogNormal, n_bit=10, mu=6., sigma=1.5),
    'triangular 10': partial(Triangular, n_bit=10, left=0, mode=2, right=1023),
    'bimodal 10': partial(Bimodal, n_bit=10, mu1=2 ** 10 * 2 / 7, sigma1=2 ** 10 / 8, mu2=2 ** 10 * 5 / 7, sigma2=2 ** 10 / 8),

    'bas 2x2': partial(BarAndStripes, width=2, height=2),
    'bas 3x3': partial(BarAndStripes, width=3, height=3),
    'bas 4x4': partial(BarAndStripes, width=4, height=4),
}
