""" Helper functions for seviri neural networks"""

import logging
import shutil
import os
import readdriver
from definitions import CMACPHVersion1Constants
from definitions import CMACPHVersion2Constants
from definitions import CMACPHVersion3Constants
from definitions import CTPVersion3Constants
from definitions import MLAYVersion3Constants

fmt = '%(levelname)s : %(filename)s : %(message)s'
logging.basicConfig(level=logging.DEBUG,
                    format=fmt
                    )


def get_parameters(version, variable):
    if variable == 'CPHCOT':
        if version == 1:
            logging.info('Loading version 1 constants')
            return CMACPHVersion1Constants()
        if version == 2:
            logging.info('Loading version 2 constants')
            return CMACPHVersion2Constants()
        elif version == 3:
            logging.info('Loading version 3 constants')
            return CMACPHVersion3Constants()
        else:
            raise Exception('No constants defined for version '
                            '{}'.format(version))
    elif variable == 'CTP':
        if version == 3:
            logging.info('Loading version 3 constants')
            return CTPVersion3Constants()
        else:
            raise Exception('No constants defined for version '
                            '{}'.format(version))
    elif variable == 'MLAY':
        if version == 3:
            logging.info('Loading version 3 constants')
            return MLAYVersion3Constants()
        else:
            raise Exception('No constants defined for version '
                            '{}'.format(version))


def all_same(items):
    """
        Check whether all elements in a list are equal.
        Returns True or False.
    """
    return all(x == items[0] for x in items)


def _get_driver_opts(backend):
    """ Set path to driver file and read driver file. """
    # read driver file for SEVIRI neural network
    # assume driver file is in same directory as this file

    # if environment variable is set use non-standard driver
    # file name
    if "SEVIRI_ML_DRIVER_FILENAME" in os.environ:
        drifile = os.environ.get("SEVIRI_ML_DRIVER_FILENAME")
    else:
        drifile = 'nn_driver.txt'

    basepath = os.path.dirname(os.path.realpath(__file__))
    ptf = os.path.join(basepath, drifile)
    if not os.path.isfile(ptf):
        raise Exception('Driver file {} does not exist'.format(ptf))
    return readdriver.parse_nn_driver(ptf, backend)


def check_theano_version(modelpath):
    """
    Check if installed Theano version matches the
    Theano version used for training.
    """
    import theano
    cot_version = modelpath.split('__')[1]
    curr_version = theano.__version__
    if curr_version != cot_version:
        msg = 'WARNING: Mismatch between Theano version {} for training ' + \
              'and your currently used version {}. Version mismatch may' + \
              'lead to errors or unexpected behaviour.'
        msg = msg.format(cot_version, curr_version)
        logging.warning(msg)


def check_tensorflow_version(modelpath):
    """
    Check if installed Tensorflow version matches the
    Tensorflow version used for training.
    """
    import tensorflow
    cot_version = modelpath.split('__')[1]
    curr_version = tensorflow.__version__
    if curr_version != cot_version:
        msg = 'WARNING: Mismatch between TF version {} for training ' + \
              'and your currently used version {}. Version mismatch may' + \
              'lead to errors or unexpected behaviour.'
        msg = msg.format(cot_version, curr_version)
        logging.warning(msg)


class ConfigTheano:
    def __init__(self, opts):
        self.use_pid_compiledir = opts['USE_PID_COMPILEDIR']
        self.use_compiledir_lock = opts['USE_THEANO_COMPILEDIR_LOCK']
        self.cdir_pid = None
        self.configure_theano_compile_locking()

    def configure_theano_compile_locking(self):
        """ Configure how to deal with compile dir locking. """
        # enable usage of PID dependent compile directory
        # creates new compile_directory
        if self.use_pid_compiledir:
            self._set_pid_compiledir()

        # enable or disable compile directory locking
        if not self.use_compiledir_lock:
            import theano
            theano.gof.compilelock.set_lock_status(False)

    def _set_pid_compiledir(self):
        """
        Set Theano compile diretory so that the directory.
        is process id depending. In case you are running
        ORAC with MPI support you are not suffering
        from compile lock, as otherwise Theano uses the
        same compile directory for each process.
        """
        pid = os.getpid()
        tflags = os.getenv('THEANO_FLAGS').split(',')
        for f in tflags:
            if f.startswith('base_compiledir'):
                cdir = f.split('=')[1]

        cdir_pid = os.path.join(cdir, 'pid' + str(pid))
        self.cdir_pid = cdir_pid
        os.environ['THEANO_FLAGS'] = 'base_compiledir={}'.format(cdir_pid)

    def remove_pid_compiledir(self):
        """ Remove PID dependent compile directory. """
        if self.use_pid_compiledir:
            if os.path.isdir(self.cdir_pid):
                shutil.rmtree(self.cdir_pid)
            else:
                msg = 'Cannot delete {} because not existing'
                msg = msg.format(self.cdir_pid)
                logging.warning(msg)
