""" Read driver file for SEVIRI neural network library """

import os
import pathlib
from definitions import MANDATORY_OPTS
from definitions import TRUE_OPTS, FALSE_OPTS
from definitions import (ModelSetupCMA, ModelSetupCPH, ModelSetupCTP,
                         ModelSetupMLAY, ModelSetupCTT)


def _set_default_filepath():
    ptf = pathlib.Path(__file__).parent.absolute()
    return os.path.join(ptf, 'data')


def _check_parsed_opts(opts):
    """ Check if parsed options are as expected. """
    for opt, filetype in MANDATORY_OPTS.items():
        try:
            opts[opt]
        except KeyError:
            msg = 'Mandatory option {} missing in driver file.'
            raise Exception(RuntimeError, msg.format(opt))

        if opt == 'DATA_PATH':
            if opts[opt].upper() == 'NONE':
                opts[opt] = _set_default_filepath()

            if not os.path.isdir(opts[opt]):
                raise Exception('DATA_PATH {} does '
                                'not exist'.format(opts[opt]))

        if filetype == 'FLOAT':
            opts[opt] = float(opts[opt])

        if filetype == 'INT':
            opts[opt] = int(opts[opt])

        if filetype == 'BOOL':
            if opts[opt] in TRUE_OPTS:
                opts[opt] = 1
            elif opts[opt] in FALSE_OPTS:
                opts[opt] = 0
            else:
                msg = '{} invalid argument for {}'
                msg = msg.format(opts[opt], opt)
                raise Exception(msg)

    return opts


def parse_nn_driver(driver_path, backend):
    """ Parse through driver file and put options to a dictionary. """
    opts = {}

    if os.path.isfile(driver_path):
        with open(driver_path, 'r') as dri:
            for line in dri:
                args = []
                tmp = line.split('=')
                if len(tmp) == 1:
                    # skip empty lines or comments
                    if tmp[0] == '' or tmp[0] == '\n' or \
                            tmp[0].startswith('#'):
                        continue
                for arg in tmp:
                    arg = arg.strip()
                    if arg[-2:] == '\n':
                        arg = arg[:-2]

                    args.append(arg)

                if len(args) != 2:
                    msg = 'Error in parsing NN driver: Extracted list ' + \
                          'contains {} arguments (2 allowed)'
                    raise Exception(RuntimeError, msg.format(len(args)))

                opts[args[0]] = args[1]

    opts['BACKEND'] = backend

    opts = _check_parsed_opts(opts)

    # select model and scaler depending on version and backend
    cma_setup = ModelSetupCMA(opts['CMA_MODEL_VERSION'],
                              opts['BACKEND'],
                              opts['DATA_PATH'])
    cma_setup.set_models_scalers()
    cph_setup = ModelSetupCPH(opts['CPH_MODEL_VERSION'],
                              opts['BACKEND'],
                              opts['DATA_PATH'])
    cph_setup.set_models_scalers()
    ctp_setup = ModelSetupCTP(opts['CTP_MODEL_VERSION'],
                              opts['BACKEND'],
                              opts['DATA_PATH'])
    ctp_setup.set_models_scalers()

    ctt_setup = ModelSetupCTT(opts['CTT_MODEL_VERSION'],
                              opts['BACKEND'],
                              opts['DATA_PATH'])
    ctt_setup.set_models_scalers()

    mlay_setup = ModelSetupMLAY(opts['MLAY_MODEL_VERSION'],
                                opts['BACKEND'],
                                opts['DATA_PATH'])
    mlay_setup.set_models_scalers()

    # set models to be used in options dictionary
    opts['CMA_MODEL_FILEPATH'] = cma_setup.model_filepath
    opts['CMA_SCALER_FILEPATH'] = cma_setup.scaler_filepath

    opts['CPH_MODEL_FILEPATH'] = cph_setup.model_filepath
    opts['CPH_SCALER_FILEPATH'] = cph_setup.scaler_filepath

    opts['CTP_LOWER_MODEL_FILEPATH'] = ctp_setup.model_lower_filepath
    opts['CTP_UPPER_MODEL_FILEPATH'] = ctp_setup.model_upper_filepath
    opts['CTP_MEDIAN_MODEL_FILEPATH'] = ctp_setup.model_median_filepath
    opts['CTP_SCALER_FILEPATH'] = ctp_setup.scaler_filepath

    opts['CTT_LOWER_MODEL_FILEPATH'] = ctt_setup.model_lower_filepath
    opts['CTT_UPPER_MODEL_FILEPATH'] = ctt_setup.model_upper_filepath
    opts['CTT_MEDIAN_MODEL_FILEPATH'] = ctt_setup.model_median_filepath
    opts['CTT_SCALER_FILEPATH'] = ctt_setup.scaler_filepath

    opts['MLAY_MODEL_FILEPATH'] = mlay_setup.model_filepath
    opts['MLAY_SCALER_FILEPATH'] = mlay_setup.scaler_filepath

    return opts
