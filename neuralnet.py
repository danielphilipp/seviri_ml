"""
    This module contains classes for configuration, preparation	and prediction
    of COT (CMA) and CPH using artificial neural networks. This module is
    used in predictCPHCOT.py.

    - class NetworkBase: Basic parent class for NetworkCPH and NetworkCOT
    - class NetworkCPH(NetworkBase): Class for CPH prediction
    - class NetworkCOT(NetworkBase): Class for COT prediction
    - class NetworkCTP(NetworkBase): Class for CTP prediction

    First author: Daniel Philipp (DWD)
    ---------------------------------------------------------------------------
    24/04/2020, DP: Initial version
    23/07/2020, DP: Import of load_model depending on used backend. Includes
                    filename checking.
"""

import os
import joblib
import warnings
import helperfuncs as hf
import logging

fmt = '%(levelname)s : %(filename)s : %(message)s'
logging.basicConfig(level=logging.DEBUG,
                    format=fmt)


def _throw_backend_not_found_error(s):
    msg = 'Backend {} not recognized.'
    raise Exception(RuntimeError, msg.format(s))


class NetworkBase:
    def __init__(self, modelpath, scalerpath, backend):
        """
        Initialize NetworkBase class.

        Args:
        - modelpath (str):  Path to trained ANN model file
        - scalerpath (str): Path to file containing scaling values
        - backend (str):    Used Keras neural network backend
        """

        self.modelpath = modelpath
        self.scalerpath = scalerpath
        self.backend = backend

    def get_model(self):
        """ Load Tensorflow or Theano trained model (.h5 file) from disk. """

        if self.backend.lower() == 'tensorflow2':
            logging.info('Setting KERAS_BACKEND env. variable  to tensorflow')
            os.environ['KERAS_BACKEND'] = 'tensorflow'
            from tensorflow.keras.models import load_model

        elif self.backend.lower() == 'theano':
            logging.info('Setting KERAS_BACKEND env. variable  to theano')
            os.environ['KERAS_BACKEND'] = 'theano'

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                from keras.models import load_model

        else:
            _throw_backend_not_found_error(self.backend)

        # for CTP and CTT we need 3 models to get uncertainty
        if isinstance(self.modelpath, list):
            models = dict()
            names = ['lower', 'median', 'upper']
            for m, n in zip(self.modelpath, names):
                models[n] = load_model(m, compile=False)
            return models
        # for CMA and CPH we need ony one model
        elif isinstance(self.modelpath, str):
            return load_model(self.modelpath, compile=False)
        else:
            raise Exception('modelpath must be list or string.')

    def _get_scaler(self):
        """ Load sklearn.preprocessing scaler from disk (.pkl file). """
        return joblib.load(self.scalerpath)

    def scale_input(self, arr):
        """ Scale input with the correct sklearn.preprocessing scaler. """
        scaler = self._get_scaler()
        return scaler.transform(arr)


class NetworkCPH(NetworkBase):
    def __init__(self, opts):

        self.opts = opts

        modelpath = opts['CPH_MODEL_FILEPATH']
        scalerpath = opts['CPH_SCALER_FILEPATH']
        backend = opts['BACKEND']

        self.version = opts['CPH_MODEL_VERSION']

        if backend == 'THEANO':
            hf.check_theano_version(modelpath)
        else:
            hf.check_tensorflow_version(modelpath)

        super().__init__(modelpath, scalerpath, backend)


class NetworkCMA(NetworkBase):
    def __init__(self, opts):
        self.opts = opts
        modelpath = opts['CMA_MODEL_FILEPATH']
        scalerpath = opts['CMA_SCALER_FILEPATH']
        backend = opts['BACKEND']

        self.version = opts['CMA_MODEL_VERSION']

        if backend == 'THEANO':
            hf.check_theano_version(modelpath)
        else:
            hf.check_tensorflow_version(modelpath)

        super().__init__(modelpath, scalerpath, backend)


class NetworkMLAY(NetworkBase):
    def __init__(self, opts):
        self.opts = opts
        modelpath = opts['MLAY_MODEL_FILEPATH']
        scalerpath = opts['MLAY_SCALER_FILEPATH']
        backend = opts['BACKEND']

        self.version = opts['MLAY_MODEL_VERSION']

        if backend == 'THEANO':
            hf.check_theano_version(modelpath)
        else:
            hf.check_tensorflow_version(modelpath)

        super().__init__(modelpath, scalerpath, backend)


class NetworkCTP(NetworkBase):
    def __init__(self, opts):

        self.opts = opts

        modelpaths = [opts['CTP_LOWER_MODEL_FILEPATH'],
                      opts['CTP_MEDIAN_MODEL_FILEPATH'],
                      opts['CTP_UPPER_MODEL_FILEPATH']]
        scalerpath = opts['CTP_SCALER_FILEPATH']
        backend = opts['BACKEND']

        self.version = opts['CTP_MODEL_VERSION']

        if backend == 'THEANO':
            hf.check_theano_version(modelpaths[1])
        else:
            hf.check_tensorflow_version(modelpaths[1])

        super().__init__(modelpaths, scalerpath, backend)


class NetworkCTT(NetworkBase):
    def __init__(self, opts):

        self.opts = opts

        modelpaths = [opts['CTT_LOWER_MODEL_FILEPATH'],
                      opts['CTT_MEDIAN_MODEL_FILEPATH'],
                      opts['CTT_UPPER_MODEL_FILEPATH']]
        scalerpath = opts['CTT_SCALER_FILEPATH']
        backend = opts['BACKEND']

        self.version = opts['CTT_MODEL_VERSION']

        if backend == 'THEANO':
            hf.check_theano_version(modelpaths[1])
        else:
            hf.check_tensorflow_version(modelpaths[1])

        super().__init__(modelpaths, scalerpath, backend)