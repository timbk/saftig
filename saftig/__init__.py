"""Static & Adaptive Filtering In Gravitational-wave-research
Implementations of prediction techniques with a unified interface.
"""
from .common import RMS, total_power
from .evaluation import (
    TestDataGenerator,
    residual_power_ratio,
    residual_amplitude_ratio,
)

from .wf import WienerFilter
from .uwf import UpdatingWienerFilter
from .lms import LMSFilter
from .polylms import PolynomialLMSFilter
