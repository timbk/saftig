import unittest, doctest
import saftig

module_list = [
    saftig.common,
    saftig.evaluation,
    saftig.wf,
    saftig.uwf,
    saftig.lms,
]

if __name__ == "__main__":
    suite = unittest.TestSuite()

    for module in module_list:
        suite.addTest(doctest.DocTestSuite(module))

    runner = unittest.TextTestRunner()
    runner.run(suite)

