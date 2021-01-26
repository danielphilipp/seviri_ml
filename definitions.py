""" Static definitions for use in various modules """

import numpy as np

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

# miscellaneous constants
DATA_NAMES = ['ir039', 'ir087', 'ir087_108', 'ir108', 'ir108_120',
              'ir120', 'ir134', 'lsm', 'nir016', 'skt', 'vis006',
              'vis008', 'ir062', 'ir073']

TRUE_OPTS = ['true', 'True', 'T', 't', '1', '.True.', '.true.']
FALSE_OPTS = ['false', 'False', 'F', 'f', '0', '.False.', '.false.']

MANDATORY_OPTS = {'DATA_PATH': 'PATH',
                  'COT_MODEL_FILENAME': 'PATH',
                  'COT_SCALER_FILENAME': 'PATH',
                  'CPH_MODEL_FILENAME': 'PATH',
                  'CPH_SCALER_FILENAME': 'PATH',
                  'USE_THEANO_COMPILEDIR_LOCK': 'BOOL',
                  'USE_PID_COMPILEDIR': 'BOOL',
                  'NN_COT_THRESHOLD': 'FLOAT',
                  'NN_CPH_THRESHOLD': 'FLOAT',
                  'VALID_REGRESSION_MIN': 'FLOAT',
                  'VALID_REGRESSION_MAX': 'FLOAT',
                  'UNC_CLD_MIN': 'FLOAT',
                  'UNC_CLD_MAX': 'FLOAT',
                  'UNC_CLR_MIN': 'FLOAT',
                  'UNC_CLR_MAX': 'FLOAT',
                  'UNC_WAT_MIN': 'FLOAT',
                  'UNC_WAT_MAX': 'FLOAT',
                  'UNC_ICE_MIN': 'FLOAT',
                  'UNC_ICE_MAX': 'FLOAT'
                  }

DEFAULT_THEANO_MODEL_COT = 'MODEL_CMA_14_150_150_1_LOSS-MSE_OPT-ADAM_LR-0001_NE-300_BS-200_GSICS_THEANO__1.0.4__v1.h5'
DEFAULT_THEANO_MODEL_CPH = 'MODEL_CPH_14_40_40_1_LOSS-MSE_OPT-ADAM_LR-0001_NE-300_BS-200_GSICS_THEANO__1.0.4__v1.h5' 

DEFAULT_TF2_MODEL_COT = 'MODEL_CMA_14_150_150_1_LOSS-MSE_OPT-ADAM_LR-0001_NE-300_BS-200_GSICS_TF2_v1.h5'
DEFAULT_TF2_MODEL_CPH = 'MODEL_CPH_14_40_40_1_LOSS-MSE_OPT-ADAM_LR-0001_NE-300_BS-200_GSICS_TF2_v1.h5'

DEFAULT_SCALER_CPH = 'SCALER_CPH_GSICS_v1.pkl'
DEFAULT_SCALER_COT = 'SCALER_CMA_GSICS_v1.pkl'

