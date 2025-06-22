import unittest

import saftig as sg

from .test_filters import TestFilter

class TestUpdatingWienerFilter(unittest.TestCase, TestFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_target(sg.UpdatingWienerFilter, [{'context_pre': 128*40, 'context_post': 128*40}])

    def test_conditioning_warning(self):
        n_filter = 128
        witness, target = sg.TestDataGenerator([0.1]).generate(int(1e4))

        for filt in self.instantiate_filters(n_filter, n_channel=2):
            self.assertWarns(RuntimeWarning, filt.condition, witness, target)

    def test_conditioning_warning(self):
        n_filter = 128
        n_channel = 2
        witness, target = sg.TestDataGenerator([0.1]).generate(int(1e4))

        # using two identical input datasets produces non-full-rank autocorrelation matrices
        witness = [witness[0]]*n_channel

        for filt in self.instantiate_filters(n_filter, n_channel=n_channel):
            self.assertWarns(RuntimeWarning, filt.apply, witness, target)

    def test_acceptance_of_minimum_input_length_different_context_length(self):
        n_filter = 128
        witness, target = sg.TestDataGenerator([0.1]).generate(n_filter*2)

        for context_len in [0, 10000]:
            filt = sg.UpdatingWienerFilter(n_filter, 0, 1, context_pre=context_len)
            pred = filt.apply(witness, target)
            self.assertEqual(len(pred), len(target))

            filt = sg.UpdatingWienerFilter(n_filter, 0, 1, context_post=context_len)
            pred = filt.apply(witness, target)
            self.assertEqual(len(pred), len(target))
