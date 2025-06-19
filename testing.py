import unittest, doctest
import saftig

'''
TODO:
- check output dimension correctness
- check that no exceptions are thrown for different input lengths
- check that FIR filter coefficient peak is at correct index (correct targeting)
- check that cancellation efficiencies match the expectation
- effect of all parameters
'''


module_list = [
    saftig.common,
    saftig.evaluation,
    saftig.wf,
    saftig.uwf,
    saftig.lms,
    saftig.polylms,
]

if __name__ == "__main__":
    suite = unittest.TestSuite()

    for module in module_list:
        suite.addTest(doctest.DocTestSuite(module))

    runner = unittest.TextTestRunner()
    runner.run(suite)

