from distutils.core import setup
from Cython.Build import cythonize
# import Cython.Compiler.Options
# Cython.Compiler.Options.annotate = True

setup(
  name='Word class',
  ext_modules=cythonize("word.pyx"),
)
# python setup.py build_ext --inplace