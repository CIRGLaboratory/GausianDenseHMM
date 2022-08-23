from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
	ext_modules = cythonize("hmmc/_hmmc.pyx", include_path=[np.get_include()])
)

