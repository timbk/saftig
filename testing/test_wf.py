import unittest

import saftig as sg

from .test_filters import TestFilter

class TestWienerFilter(unittest.TestCase, TestFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_target(sg.WienerFilter)

    def test_conditioning_warning(self):
        n_filter = 128
        witness, target = sg.TestDataGenerator([0.1]).generate(int(1e4))

        # using two identical input datasets produces non-full-rank autocorrelation matrices
        witness = [witness[0], witness[0]]

        for filt in self.instantiate_filters(n_filter, n_channel=2):
            self.assertWarns(RuntimeWarning, filt.condition, witness, target)
