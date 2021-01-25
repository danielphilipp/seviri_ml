# seviri_ml
External SEVIRI cloud masking and cloud phase determination module for ORAC (https://github.com/ORAC_CC/orac). This module applies neural networks on SEVIRI and auxiliary data using SEVIRI's full spectral capabilities.

REQUIREMENTS
-------------------------------------------
- C compiler (recommended gfortran or Cray C compiler)
- Fortran 90 compiler (recommended gfortran or Cray Fortran compiler)
- Python 3.6.x (other Python versions not tested)
- Theano 1.0.4

COMPILE
-------------------------------------------
1. Edit make.config:
   - Select compilers (CC / F90)
   - Set full path to your numpy includes (NUMPYINCLUDE)
     Typically located at: /path/to/your/python/lib/python3.6/site-packages/numpy/core/include/numpy
   - Set full path to your Python includes (PYINCLUDE)
     Typically located at: /path/to/your/python/include/python3.6m
2. Type "make"

USE WITH ORAC
-------------------------------------------
