""" Static definitions for use in various modules """

import numpy as np
from os.path import join as ojoin

# data types
SREAL = np.float32
BYTE = np.byte

# fill values
SREAL_FILL_VALUE = -999.0
DREAL_FILL_VALUE = -999.0
BYTE_FILL_VALUE = -127
LINT_FILL_VALUE = -32767
SINT_FILL_VALUE = -32767

# flags
IS_CLEAR = 0
IS_CLOUD = 1
IS_WATER = 1
IS_ICE = 2
IS_SLAY = 0
IS_MLAY = 1


# miscellaneous constants
DATA_NAMES = ['ir039', 'ir087', 'ir087_108', 'ir108', 'ir108_120',
              'ir120', 'ir134', 'lsm', 'nir016', 'satzen', 'skt',
              'solzen', 'vis006', 'vis008', 'ir062', 'ir073']

TRUE_OPTS = ['true', 'True', 'T', 't', '1', '.True.', '.true.']
FALSE_OPTS = ['false', 'False', 'F', 'f', '0', '.False.', '.false.']

MANDATORY_OPTS = {'DATA_PATH': 'PATH',
                  'CMA_MODEL_VERSION': 'INT',
                  'CPH_MODEL_VERSION': 'INT',
                  'CTP_MODEL_VERSION': 'INT',
                  'MLAY_MODEL_VERSION': 'INT',
                  'CORRECT_IR039_OUT_OF_RANGE': 'BOOL',
                  'USE_THEANO_COMPILEDIR_LOCK': 'BOOL',
                  'USE_PID_COMPILEDIR': 'BOOL',
                  'CTP_UNCERTAINTY_METHOD': 'STR'
                  }


class CMACPHVersion1Constants:
    def __init__(self):
        # uncertainty parameters
        self.UNC_INTERCEPT_LIQ = 50.18
        self.UNC_INTERCEPT_ICE = 50.77
        self.UNC_INTERCEPT_CLR = 50.34
        self.UNC_INTERCEPT_CLD = 44.96

        self.UNC_SLOPE_LIQ = 48.35
        self.UNC_SLOPE_ICE = -48.58
        self.UNC_SLOPE_CLR = 37.70
        self.UNC_SLOPE_CLD = -47.26

        # COT threshold
        self.NN_COT_THRESHOLD = 0.12
        # CPH threshold
        self.NN_CPH_THRESHOLD = 0.50

        # [0,1] regression value valid range
        self.VALID_NOR_REGRESSION_MAX = 1.0
        self.VALID_NOR_REGRESSION_MIN = 0.0


class CMACPHVersion2Constants:
    def __init__(self):
        # uncertainty parameters
        self.UNC_INTERCEPT_LIQ = 50.18 
        self.UNC_INTERCEPT_ICE = 50.77
        self.UNC_INTERCEPT_CLR = 50.34
        self.UNC_INTERCEPT_CLD = 44.96

        self.UNC_SLOPE_LIQ = 48.35
        self.UNC_SLOPE_ICE = -48.58
        self.UNC_SLOPE_CLR = 37.70
        self.UNC_SLOPE_CLD = -47.26

        # COT threshold
        self.NN_COT_THRESHOLD = 0.12
        # CPH threshold
        self.NN_CPH_THRESHOLD = 0.50

        # [0,1] regression value valid range
        self.VALID_NOR_REGRESSION_MAX = 1.0
        self.VALID_NOR_REGRESSION_MIN = 0.0


class CMACPHVersion3Constants:
    def __init__(self):
        # uncertainty parameters
        self.UNC_INTERCEPT_LIQ = 50.71
        self.UNC_INTERCEPT_ICE = 44.54
        self.UNC_INTERCEPT_CLR = 42.86
        self.UNC_INTERCEPT_CLD = 48.04

        self.UNC_SLOPE_LIQ = 47.16
        self.UNC_SLOPE_ICE = -47.23
        self.UNC_SLOPE_CLR = 25.31
        self.UNC_SLOPE_CLD = -51.51

        # COT threshold
        self.NN_COT_THRESHOLD = 0.13
        # CPH threshold
        self.NN_CPH_THRESHOLD = 0.50

        # [0,1] regression value valid range
        self.VALID_NOR_REGRESSION_MAX = 1.0
        self.VALID_NOR_REGRESSION_MIN = 0.0


class CTPVersion3Constants:
    def __init__(self):
        # CTP regression value valid range
        self.VALID_CTP_REGRESSION_MIN = 0.
        self.VALID_CTP_REGRESSION_MAX = 1050.


class MLAYVersion3Constants:
    def __init__(self):
        self.NN_MLAY_THRESHOLD = 0.35

        # [0,1] regression value valid range
        self.VALID_NOR_REGRESSION_MAX = 1.0
        self.VALID_NOR_REGRESSION_MIN = 0.0


class ModelSetupCMA:
    def __init__(self, version, backend, data_path):
        self.version = version
        self.backend = backend
        self.data_path = data_path

        self.model_filepath = None
        self.scaler_filepath = None

    def set_models_scalers(self):

        if self.version == 1:
            if self.backend == 'THEANO':
                model = 'MODEL_CMA_14_150_150_1_LOSS-MSE_OPT-ADAM' \
                        '_LR-0001_NE-300_BS-200_GSICS_THEANO__1.0.4__v1.h5'
            else:
                model = 'MODEL_CMA_14_150_150_1_LOSS-MSE_OPT-ADAM' \
                        '_LR-0001_NE-300_BS-200_GSICS_TF2__2.4.1__v1.h5'

            scaler = 'SCALER_CMA_GSICS_v1.pkl'

        elif self.version == 2:
            if self.backend == 'THEANO':
                model = 'MODEL_CMA_14_150_150_1_LOSS-MSE_OPT-ADAM' \
                        '_LR-0001_NE-300_BS-200_GSICS_THEANO__1.0.4__v2.h5'
            else:
                model = 'MODEL_CMA_14_150_150_1_LOSS-MSE_OPT-ADAM' \
                        '_LR-0001_NE-300_BS-200_GSICS_TF2__2.4.1__v2.h5'

            scaler = 'SCALER_CMA_GSICS_v2.pkl'

        elif self.version == 3:
            if self.backend == 'THEANO':
                model = 'MODEL_CMA_16_125_125_125_1_LOSS-MSE_OPT-ADAM' \
                        '_LR-0001_NE-300_BS-200_GSICS_THEANO__1.0.4__v3.h5'
            else:
                model = 'MODEL_CMA_16_125_125_125_1_LOSS-MSE_OPT-ADAM' \
                        '_LR-0001_NE-300_BS-200_GSICS_TF2__2.4.1__v3.h5'

            scaler = 'SCALER_CMA_GSICS_v3.pkl'

        else:
            raise Exception('Version {:d} not supported '
                            'for COT models.'.format(self.version))

        self.model_filepath = ojoin(self.data_path,
                                    'v{:d}'.format(self.version), model)
        self.scaler_filepath = ojoin(self.data_path,
                                     'v{:d}'.format(self.version), scaler)


class ModelSetupCPH:
    def __init__(self, version, backend, data_path):
        self.version = version
        self.backend = backend
        self.data_path = data_path

        self.model_filepath = None
        self.scaler_filepath = None

    def set_models_scalers(self):
        if self.version == 1:
            if self.backend == 'THEANO':
                model = 'MODEL_CPH_14_40_40_1_LOSS-MSE_OPT-ADAM' \
                        '_LR-0001_NE-300_BS-200_GSICS_THEANO__1.0.4__v1.h5'
            else:
                model = 'MODEL_CPH_14_40_40_1_LOSS-MSE_OPT-ADAM' \
                        '_LR-0001_NE-300_BS-200_GSICS_TF2__2.4.1__v1.h5'

            scaler = 'SCALER_CPH_GSICS_v1.pkl'

        elif self.version == 2:
            if self.backend == 'THEANO':
                model = 'MODEL_CPH_14_40_40_1_LOSS-MSE_OPT-ADAM' \
                        '_LR-0001_NE-300_BS-200_GSICS_THEANO__1.0.4__v2.h5'
            else:
                model = 'MODEL_CPH_14_40_40_1_LOSS-MSE_OPT-ADAM' \
                        '_LR-0001_NE-300_BS-200_GSICS_TF2__2.4.1__v2.h5'

            scaler = 'SCALER_CPH_GSICS_v2.pkl'

        elif self.version == 3:
            if self.backend == 'THEANO':
                model = 'MODEL_CPH_16_150_150_1_LOSS-MSE_OPT-ADAM' \
                        '_LR-0001_NE-300_BS-200_GSICS_THEANO__1.0.4__v3.h5'
            else:
                model = 'MODEL_CPH_16_150_150_1_LOSS-MSE_OPT-ADAM' \
                        '_LR-0001_NE-300_BS-200_GSICS_TF2__2.4.1__v3.h5'

            scaler = 'SCALER_CPH_GSICS_v3.pkl'

        else:
            raise Exception('Version {:d} not supported '
                            'for CPH models.'.format(self.version))

        self.model_filepath = ojoin(self.data_path,
                                    'v{:d}'.format(self.version), model)
        self.scaler_filepath = ojoin(self.data_path,
                                     'v{:d}'.format(self.version), scaler)


class ModelSetupCTP:
    def __init__(self, version, backend, data_path):
        self.version = version
        self.backend = backend
        self.data_path = data_path

        self.model_lower_filepath = None
        self.model_upper_filepath = None
        self.model_median_filepath = None
        self.scaler_filepath = None

    def set_models_scalers(self):
        if self.version == 3:
            if self.backend == 'THEANO':
                low = 'MODEL_CTP_LOWER-0.1587_16_120_120_1_LOSS-QRM_OPT-ADAM'\
                      '_LR-0001_NE-300_BS-200_GSICS_THEANO__1.0.4__v3.h5'
                med = 'MODEL_CTP_MEDIAN-0.5000_16_120_120_1_LOSS-QRM_OPT-ADAM'\
                      '_LR-0001_NE-300_BS-200_GSICS_THEANO__1.0.4__v3.h5'
                upp = 'MODEL_CTP_UPPER-0.8413_16_120_120_1_LOSS-QRM_OPT-ADAM'\
                      '_LR-0001_NE-300_BS-200_GSICS_THEANO__1.0.4__v3.h5'
            else:
                low = 'MODEL_CTP_LOWER-0.1587_16_120_120_1_LOSS-QRM_OPT-ADAM'\
                      '_LR-0001_NE-300_BS-200_GSICS_TF2__2.4.1__v3.h5'
                med = 'MODEL_CTP_MEDIAN-0.5000_16_120_120_1_LOSS-QRM_OPT-ADAM'\
                      '_LR-0001_NE-300_BS-200_GSICS_TF2__2.4.1__v3.h5'
                upp = 'MODEL_CTP_UPPER-0.8413_16_120_120_1_LOSS-QRM_OPT-ADAM'\
                      '_LR-0001_NE-300_BS-200_GSICS_TF2__2.4.1__v3.h5'

            scaler = 'SCALER_CTP_GSICS_v3.pkl'

        else:
            raise Exception('Version {:d} not supported '
                            'for CTP models.'.format(self.version))

        self.model_lower_filepath = ojoin(self.data_path,
                                          'v{:d}'.format(self.version),
                                          low)
        self.model_median_filepath = ojoin(self.data_path,
                                           'v{:d}'.format(self.version),
                                           med)
        self.model_upper_filepath = ojoin(self.data_path,
                                          'v{:d}'.format(self.version),
                                          upp)
        self.scaler_filepath = ojoin(self.data_path,
                                     'v{:d}'.format(self.version),
                                     scaler)


class ModelSetupMLAY:
    def __init__(self, version, backend, data_path):
        self.version = version
        self.backend = backend
        self.data_path = data_path

        self.model_filepath = None
        self.scaler_filepath = None

    def set_models_scalers(self):

        if self.version == 3:
            if self.backend == 'THEANO':
                model = 'MODEL_MLAY_16_100_100_1_LOSS-BCE_OPT-ADAM' \
                        '_LR-0001_NE-50_BS-200_GSICS_THEANO__1.0.4__v3.h5'
            else:
                model = 'MODEL_MLAY_16_100_100_1_LOSS-BCE_OPT-ADAM' \
                        '_LR-0001_NE-50_BS-200_GSICS_TF__2.4.1__v3.h5'

            scaler = 'SCALER_MLAY_GSICS_v3.pkl'

        else:
            raise Exception('Version {:d} not supported '
                            'for CPH models.'.format(self.version))

        self.model_filepath = ojoin(self.data_path,
                                    'v{:d}'.format(self.version), model)
        self.scaler_filepath = ojoin(self.data_path,
                                     'v{:d}'.format(self.version), scaler)
