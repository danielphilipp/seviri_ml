# SEVIRI_ML
Machine learning based module to retrieve a set of cloud variables from Spinning Enhanced Visible and InfraRed Imager (SEVIRI) measurements using its full spectral capabilities. The available variables to be retrieved are:

- Cloud Mask (CMA) - predictCPHCOT.py

- Cloud Phase (CPH) - predictCPHCOT.py

- Cloud Top Pressure (CTP) - predictCTP.py

- Multilayer Flag (MLAY) - predictMLAY.py

The repository contains pre-trained networks which can easily be used. Networks trained with Theano and Tensorflow2 backends are available. It is written in Python and makes use of the Keras machine learining functionalities (https://keras.io/). A Fortran and C interface is excluded as well. Also to be used as an external module with the ORAC retrieval software (https://github.com/ORAC_CC/orac). The CMA and CPH networks are available in three version (1-3). CTP and MLAY is only available for version 3. Version 3 is the default for all networks and highly recommended. 

REQUIREMENTS
-------------------------------------------
General:
- Run seviri_ml with Theano:
   - Python 3.6 (other Python versions not tested)
   - Theano 1.0.4 recommended
- Run seviri_ml with Tensorflow 2:
   - Python 3.5 - 3.8
   - Tensorflow 2.4.1 recommended

Additional for use with ORAC:
   - C compiler (recommended gcc or Cray C compiler)
   - Fortran 90 compiler (recommended gfortran or Cray Fortran compiler)
   
THEANO OR TENSORFLOW
-------------------------------------------
In this repo models trained with two different backends are available:
   - Theano 1.0.4 (outdated but runs at ECMWF HPC at Reading)
   - Tensorflow 2.4.1 (consequently updated)

Both are provided in './data/v{1,2,3}'. To select a certain backend set environment variable 'SEVIRI_ML_BACKEND' to either 'TENSORFLOW' or 'THEANO'. Default behaviour is 'TENSORFLOW'. Using Tensorflow 2 is recommended but can be sketchy to install and run on some machines.

USE WITH PURE PYTHON
-------------------------------------------
1. Import predictCPHCOT.py / predictCTP.py / predictMLAY.py into your main script.
2. Call predict{CPHCOT, CTP, MLAY}.predict_{CPH_COT, CTP, MLAY}(vis006, vis008, nir016, ir039, ir062, ir073, ir087, ir108, ir120, ir134, lsm, skt, solzen=None, satzen=None, undo_true_refl=False, correct_vis_cal_nasa_to_impf=0)
3. The function returns a list with following structure: 
   - predict_CPH_COT(args, kwargs): [COT_regression, CMA, CMA_uncertainty, CPH_regression, CPH, CPH_uncertainty]
   - predict_CTP(args, kwargs): [CTP, CTP_uncertainty]
   - predict_MLAY(args, kwargs): [MLAY_probability, MLAY_flag, MLAY_uncertainty]

undo_true_refl is a logical variable specifying if visible channels (vis006, vis008, ir016) should be multiplied by the cosine of the solar zenith angle to remove this normalization (default False). correct_vis_cal_nasa_to_impf is a integer variable specifying if the visible channels should be linearily corrected from the NASA calibration to the IMPF calibration with which the networks were trained. 0 = No correction if your visible channels were calibrated with IMPF coefficients. 1 = Your visible channels are calibrated with the NASA calibration and your satellite is Meteosat MSG1. 2 = Your visible channels are calibrated with the NASA calibration and your satellite is Meteosat MSG2. 3 = Your visible channels are calibrated with the NASA calibration and your satellite is Meteosat MSG3. 4 = Your visible channels are calibrated with the NASA calibration and your satellite is Meteosat MSG4.

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
The driver file 'nn_driver.txt' defines the basic behaviour of seviri_ml. 'nn_driver.txt' has to be located in the base directory. For non-standard driver filenames set environment variable SEVIRI_ML_DRIVER_FILENAME=newfilename.txt. Also non-standard driver filenames have to be located in the base directory.

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
   
INFORMATION ABOUT MODELS
-------------------------------------------
- Version 1 and 2:

   Models are trained with four months out of 2018 CALIOP v3 data collocated with SEVIRI measurements, a land-sea mask and ERA5 skin temperature. SEVIRI data were read with satpy (https://github.com/pytroll/satpy) applying the GSICS calibration. Be aware that for older satpy versions GSICS calibration had a bug (https://github.com/pytroll/satpy/pull/1323). Since version 2 models satpy's new functionality 'apply_earthsun_distance_correction' is applied. 

   Data for model input were normalized using z-score scaling (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). Scaling coefficients for transformation and re-transformation are saved in the SCALER files.

   Changes from version 1 to version 2 models:
      1.  ORAC (seviri_util) calculates the visible channel reflectances including a sun-earth-distance correction as a function of the day of the year to account for the earth's orbital eccentricity. The satpy version used for reading data for training did not apply this correction. With https://github.com/pytroll/satpy/pull/1341 satpy included this functionality and thus version 2 models were trained consistently with the prediction data in ORAC.

- Version 3:

   Models were re-trained with all 2018 months of CALIOP v4 data collocated with SEVIRI measurements, a land-sea mask and ERA5 skin temperature. Solar and satellite zenith angles have been added to the network input features. Theresholds and uncertainty characterization has been updated. CTP and MLAY has been added to the available variables. 
