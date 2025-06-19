import sys
import unittest, doctest
import saftig as sg

import test_evaluation

'''
TODO:
- check output dimension correctness
- check that no exceptions are thrown for different input lengths
- check that FIR filter coefficient peak is at correct index (correct targeting)
- check that cancellation efficiencies match the expectation
- effect of all parameters
'''

module_list = [
    sg.common,
    sg.evaluation,
    sg.wf,
    sg.uwf,
    sg.lms,
    sg.polylms,
]

if __name__ == "__main__":
    suite = unittest.TestLoader().discover('.')

    # load unittests
    for module in module_list:
        suite.addTest(doctest.DocTestSuite(module))

    runner = unittest.TextTestRunner()
    status = runner.run(suite)

    if not status.wasSuccessful():
        exit(8) # indicate failure for CI

