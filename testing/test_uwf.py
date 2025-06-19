import unittest

import saftig as sg

from .test_filters import TestFilter

class TestUpdatingWienerFilter(unittest.TestCase, TestFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_target(sg.UpdatingWienerFilter, {'context_pre': 128*40})
