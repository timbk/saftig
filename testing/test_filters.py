import unittest
from scipy.signal import welch
import numpy as np
from icecream import ic

import saftig as sg

class TestFilter:
    """ wrapper class for """
    def set_target(self, target_filter, default_filter_parameters={}):
        self.target_filter = target_filter
        self.default_filter_parameters = default_filter_parameters

    def instantiate_filter(self, n_filter=128, idx_target=0, n_channel=1):
        return self.target_filter(n_filter, idx_target, n_channel, **self.default_filter_parameters)

    def test_output_shapes(self):
        n_filter = 128
        witness, target = sg.TestDataGenerator(0.1).generate(int(1e4))

        filt = self.instantiate_filter(n_filter)
        filt.condition(witness, target)

        # with padding
        prediction = filt.apply(witness, target)
        self.assertEqual(prediction.shape, target.shape)

        # without padding
        prediction = filt.apply(witness, target, pad=False)
        self.assertEqual(len(prediction), len(target) - n_filter + 1)

    def test_performance(self):
        n_filter = 128
        witness, target = sg.TestDataGenerator([0.1]*2).generate(int(1e4))

        filt = self.instantiate_filter(n_filter, n_channel=2)
        filt.condition(witness, target)

        prediction = filt.apply(witness, target)
        residual = sg.RMS((target - prediction)[3000:])
        
        self.assertGreater(residual, 0.05)
        self.assertLess(residual, 0.15)

