import unittest

import saftig as sg

from .test_filters import TestFilter

class TestLMSFilterC(unittest.TestCase, TestFilter):
    """ tests for the LeastMeanSquares filter implementation """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        test_configurations = [
            {'normalized': True},
            {'normalized': False, 'step_scale': 0.001},
        ]
        self.set_target(sg.LMSFilterC, test_configurations)
