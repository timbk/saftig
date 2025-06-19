import sys
import unittest, doctest
import saftig as sg

module_list = [
    sg.common,
    sg.evaluation,
    sg.wf,
    sg.uwf,
    sg.lms,
    sg.polylms,
]

def load_tests(loader, tests, ignore):
    for module in module_list:
        tests.addTests(doctest.DocTestSuite(module))
    return tests
