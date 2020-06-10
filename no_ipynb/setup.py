from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np


# OPT="-DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes" python3 setup.py build_ext
# export CFLAGS='-I/usr/lib/python3.7/site-packages/numpy/core/include/'



setup(name='cy',
      ext_modules=cythonize(
            [Extension(
                "cy", 
                ["cy.pyx"], 
                include_dirs=[np.get_include()])])
)