import unittest
from scipy.signal import welch
import numpy as np
from icecream import ic
import io, contextlib
import warnings

import saftig as sg



class TestFilter:
    """ wrapper class for """
    def set_target(self, target_filter, default_filter_parameters=[{}]):
        assert type(default_filter_parameters) == list
        self.target_filter = target_filter
        self.default_filter_parameters = default_filter_parameters

    # # a super wild wrapper to check stdout output; not needed anymore as I switched to using warnings
    # def get_stdout(self, cmd, parameter=None):
    #     """ execute the given function call string with exec(), return the return value and the stdout string """
    #     f = io.StringIO()
    #     returnvalue = None
    #     with contextlib.redirect_stdout(f):
    #         exec('returnvalue = '+cmd)
    #     return returnvalue, f.getvalue()

    def instantiate_filters(self, n_filter=128, idx_target=0, n_channel=1):
        for parameters in self.default_filter_parameters:
            yield self.target_filter(n_filter, idx_target, n_channel, **parameters)

    def test_exception_on_missshaped_input(self):
        n_filter = 128
        witness, target = sg.TestDataGenerator(0.1).generate(int(1e4))

        for filt in self.instantiate_filters(n_filter):
            with warnings.catch_warnings(): # warnings are expected here
                warnings.simplefilter("ignore")
                filt.condition(witness, target)
            self.assertRaises(ValueError, filt.apply, 1, 1)
            self.assertRaises(ValueError, filt.apply, [[[1], [1]]], [1])

    def test_output_shapes(self):
        n_filter = 128
        witness, target = sg.TestDataGenerator(0.1).generate(int(1e4))

        for filt in self.instantiate_filters(n_filter):
            with warnings.catch_warnings(): # warnings are expected here
                warnings.simplefilter("ignore")
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

        for filt in self.instantiate_filters(n_filter, n_channel=2):
            with warnings.catch_warnings(): # warnings are expected here
                warnings.simplefilter("ignore")
                filt.condition(witness, target)

            prediction = filt.apply(witness, target)
            residual = sg.RMS((target - prediction)[3000:])

            self.assertGreater(residual, 0.05)
            self.assertLess(residual, 0.15)
