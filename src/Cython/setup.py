# Run from commandline using
# python setup.py build_ext --inplace

from setuptools import setup

from Cython.Build import cythonize

file_name = "HelloWorld.pyx"

setup(
    ext_modules=cythonize(file_name)
)
