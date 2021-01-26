""" Read driver file for SEVIRI neural network library """

import os
import pathlib
from definitions import MANDATORY_OPTS
from definitions import TRUE_OPTS, FALSE_OPTS
from definitions import DEFAULT_THEANO_MODEL_COT, DEFAULT_THEANO_MODEL_CPH
from definitions import DEFAULT_TF2_MODEL_COT, DEFAULT_TF2_MODEL_CPH
from definitions import DEFAULT_SCALER_CPH, DEFAULT_SCALER_COT

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
                raise Exception('DATA_PATH {} does not exist'.format(opts[opt]))

        if opt == 'COT_MODEL_FILENAME':
            fp_opt = 'COT_MODEL_FILEPATH'
            if opts[opt].upper() == 'NONE':
                if opts['BACKEND'] == 'THEANO':
                    opts[fp_opt] = os.path.join(opts['DATA_PATH'], 
                                                DEFAULT_THEANO_MODEL_COT)
                else:
                    opts[fp_opt] = os.path.join(opts['DATA_PATH'],
                                                DEFAULT_TF2_MODEL_COT)
            else:
                opts[fp_opt] = os.path.join(opts['DATA_PATH'],
                                            opts[opt] )

            if not os.path.isfile(opts[fp_opt]):
                raise Exception('{} {} does not exist'.format(fp_opt, opts[fp_opt]))

        if opt == 'CPH_MODEL_FILENAME':
            fp_opt = 'CPH_MODEL_FILEPATH'
            if opts[opt].upper() == 'NONE':
                if opts['BACKEND'] == 'THEANO':
                    opts[fp_opt] = os.path.join(opts['DATA_PATH'],
                                                DEFAULT_THEANO_MODEL_CPH)
                else:
                    opts[fp_opt] = os.path.join(opts['DATA_PATH'],
                                                DEFAULT_TF2_MODEL_CPH)
            else:
                opts[fp_opt] = os.path.join(opts['DATA_PATH'],
                                            opts[opt])
            if not os.path.isfile(opts[fp_opt]):
                raise Exception('{} {} does not exist'.format(fp_opt, opts[fp_opt]))

        if opt == 'CPH_SCALER_FILENAME':
            fp_opt = 'CPH_SCALER_FILEPATH'
            if opts[opt].upper() == 'NONE':
                opts[fp_opt] = os.path.join(opts['DATA_PATH'],
                                            DEFAULT_SCALER_CPH)
            else:
                opts[fp_opt] = os.path.join(opts['DATA_PATH'],
                                            opts[opt])
            if not os.path.isfile(opts[fp_opt]):
                raise Exception('{} {} does not exist'.format(fp_opt, opts[fp_opt]))

        if opt == 'COT_SCALER_FILENAME':
            fp_opt = 'COT_SCALER_FILEPATH'
            if opts[opt].upper() == 'NONE':
                opts[fp_opt] = os.path.join(opts['DATA_PATH'],
                                            DEFAULT_SCALER_COT)
            else:
                opts[fp_opt] = os.path.join(opts['DATA_PATH'],
                                            opts[opt])
            if not os.path.isfile(opts[fp_opt]):
                raise Exception('{} {} does not exist'.format(fp_opt, opts[fp_opt]))

        if filetype == 'FLOAT':
            opts[opt] = float(opts[opt])

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
    return _check_parsed_opts(opts)
