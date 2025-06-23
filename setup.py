from setuptools import setup, Extension
import setuptools

c_modules = [
    Extension('saftig._lms_c', sources=['saftig/_lms_c.c'])
]

setup(
        name="SAFTIG",
        version="0.1",
        description="Collection of implementations for noise prediction techniques like the Wiener filter",
        url="https://github.com/timbk/saftig",
        author="Tim Kuhlbusch et al.",
        author_email="kuhlbusch@physik.rwth-aachen.de",
        install_requires=[
                          "numpy",
                          "matplotlib",
                          "scipy",
                          "icecream",
                          ],
        ext_modules = c_modules,
        packages=['saftig'],
)
