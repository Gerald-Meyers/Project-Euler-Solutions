# Run from commandline using
# python setup.py build_ext --inplace

from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "HelloWorld",
        ["HelloWorld.pyx"],
        include_dirs=None,
        define_macros=None,
        undef_macros=None,
        library_dirs=None,
        libraries=None,
        runtime_library_dirs=None,
        extra_objects=None,
        extra_compile_args=None,
        extra_objects=None,
        extra_link_args=None
    )
]

setup(
    name="HelloWorld",
    author="Gerald",
    author_email="GeraldSMeyers@gmail.com",
    version="0.0.1",

    ext_modules=cythonize(extensions),
    compiler_directives={
        "language_level": "3"
    }
)
