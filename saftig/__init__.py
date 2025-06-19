"""Static & Adaptive Filtering In Gravitational-wave-research
Implementations of prediction techniques with a unified interface.
"""
from .common import RMS
from .evaluation import TestDataGenerator

from .wf import WienerFilter
from .uwf import UpdatingWienerFilter
from .lms import LMSFilter
from .polylms import PolynomialLMSFilter
