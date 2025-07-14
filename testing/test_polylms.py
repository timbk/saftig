import unittest

import saftig as sg

#from .test_filters import TestFilter
from .test_lms import TestLMSFilter

class TestPolynomialLMSFilter(TestLMSFilter):
    """ tests for the polynomial vaiant of a LeastMeanSquares filter implementation """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        test_configurations = [
            {'order': 1},
            {'order': 1, 'coefficient_clipping': 2},
            {'order': 2},
            {'order': 1, 'normalized': False, 'step_scale': 0.001},
        ]
        self.set_target(sg.PolynomialLMSFilter, test_configurations)
