# seviri_ml
External SEVIRI cloud masking and cloud phase determination module for ORAC (https://github.com/ORAC_CC/orac). This module applies neural networks on SEVIRI and auxiliary data using SEVIRI's full spectral capabilities.

REQUIREMENTS
-------------------------------------------
- C compiler (recommended gfortran or Cray C compiler)
- Fortran 90 compiler (recommended gfortran or Cray Fortran compiler)
- Run seviri_ml with Theano:
   - Python 3.6 (other Python versions not tested)
   - Theano 1.0.4 recommended
- Run seviri_ml with Tensorflow 2:
   - Python 3.5 - 3.8
   - Tensorflow 2.4.1 recommended
   
THEANO OR TENSORFLOW
-------------------------------------------
In this repo models trained with two different backends are available:
   - Theano 1.0.4 (outdated but runs at ECMWF HPC)
   - Tensorflow 2.4.1 (consequently updated)
Both are provided in './data'. To select a certain backend set environment variable 'SEVIRI_ML_BACKEND' to either 'TENSORFLOW' or 'THEANO'. Default behaviour is 'TENSORFLOW'.

COMPILE
-------------------------------------------
1. Edit make.config:
   - Select compilers (CC / F90)
   - Set full path to your numpy includes (NUMPYINCLUDE) typically located at: '/path/to/your/python/lib/python3.x/site-packages/numpy/core/include/numpy'.
     '/path/to/your/python/lib/python3.x' can be determined with the shell command 'python3-config --ldflags'.
   - Set full path to your Python includes (PYINCLUDE). You can find your PYINCLUDES with the following shell command: 'python3-config --includes'
2. Run 'make'.
3. static library 'libsevann.a' will be created.
4. Fortran module file 'SEVIRI_NEURAL_NET_M.mod' for use 
   inside ORAC will be created.

DRIVER FILE
-------------------------------------------
The driver file 'nn_driver.txt' defines the basic behaviour of seviri_ml. If you use the provided model files with the given filename within the standard directory (./data) you won't have to modify anything for options 1 - 5. Options 6 and 7 are Theano specific and are not read if backend is Tensorflow. Options 8 - 19 are specific for the provided networks and should not be changed unless you might use your own model with different characteristics. Using non-standard (your own) models at your own risks.

USE WITH ORAC
-------------------------------------------
1. Compile this library as described above.
2. Edit nn_driver.txt according to your needs.
3. Add the seviri_ml library to your ORAC LIB file e.g. '-L$(SEVML_DIR) -lsevann'.
4. Add NUMPYINCLUDE and PYINCLUDE from make.config to your main ORAC LIB file
5. Add Python libs to your main ORAC LIB FILE. You can find your Python libraries with the following shell command: 'python3-config --ldflags'
6. Compile the pre_processor with 
   "-DINCLUDE_SEVIRI_NEURALNET" macro.
6. Run ORAC with preproc driver option 
   "USE_SEVIRI_ANN=True" (default is to False) and 
   "USE_GSICS=True" (default is to True).
   
USE WITH PURE PYTHON
-------------------------------------------
1. Import predictCPHCOT.py into your main script.
2. Edit nn_driver.txt according to your needs.
3. Call predictCPHCOT.predict_CPH_COT(vis006, vis008, nir016, ir039, ir062, ir073, ir087, ir108, ir120, ir134, lsm, skt)
4. The function returns a list with following structure: CMA_regression, CMA_binary, CMA_uncertainty, CPH_regression, CPH_binary, CPH_uncertainty 
