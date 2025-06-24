from setuptools import setup, Extension
import setuptools
import numpy as np

c_modules = [
    Extension('saftig._lms_c', sources=['saftig/_lms_c.cpp'])
]

setup(
        packages=['saftig'],

        ext_modules = c_modules,
        include_dirs=[np.get_include()],  # Include NumPy headers
)
