"""
    This module contains functions to do neural network CMA/CPH/CTP/CTT/MLAY
    predictions based on SEVIRI measurements.

    Author: Daniel Philipp (DWD, 2022)
"""

import time
import os
import logging
import helperfuncs as hf
import seviri_ml_core

fmt = '%(levelname)s : %(filename)s : %(message)s'
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format=fmt)

# set backend name from environment variable
backend = hf.get_backend_name(os.environ.get('SEVIRI_ML_BACKEND'))

# read nn_driver.txt
opts = hf.get_driver_opts(backend)

# configure theano compilation
if backend == 'THEANO':
    Tconfig = hf.ConfigTheano(opts)


def predict_cma(vis006, vis008, nir016, ir039, ir062, ir073, ir087,
                ir108, ir120, ir134, lsm, skt, solzen=None, satzen=None,
                undo_true_refl=False, correct_vis_cal_nasa_to_impf=0,
                make_binary=True, make_uncertainty=True):
    """ Run cloud mask (CMA) prediction. """

    logger.info('---------- RUNNING CMA ANN ----------')

    v = 'CMA'

    # put data into structure
    data = seviri_ml_core.InputData(
            vis006, vis008, nir016, ir039, ir062, ir073,
            ir087, ir108, ir120, ir134, lsm, skt, solzen, satzen
            )

    cldmask = None

    results = []

    # create a processor instance
    proc = seviri_ml_core.ProcessorCMA(
            data, undo_true_refl, correct_vis_cal_nasa_to_impf,
            cldmask, v, opts
            )

    # run prediction
    start = time.time()
    prediction = proc.get_prediction()
    logger.info("Time for prediction CMA: {:.3f}".format(time.time() - start))
    results.append(prediction)

    if make_binary:
        # apply threshold
        binary = proc.get_binary()
        results.append(binary)
    if make_uncertainty:
        # run uncertainty calculation
        uncertainty = proc.get_uncertainty()
        results.append(uncertainty)

    return results


def predict_cph(vis006, vis008, nir016, ir039, ir062, ir073, ir087,
                ir108, ir120, ir134, lsm, skt, solzen=None, satzen=None,
                undo_true_refl=False, correct_vis_cal_nasa_to_impf=0,
                cldmask=None, make_binary=True, make_uncertainty=True):
    """ Run cloud phase (CPH) prediction. """

    logger.info('---------- RUNNING CPH ANN ----------')

    v = 'CPH'

    # put data into structure
    data = seviri_ml_core.InputData(
            vis006, vis008, nir016, ir039, ir062, ir073,
            ir087, ir108, ir120, ir134, lsm, skt, solzen, satzen
            )

    # create a processor instance
    proc = seviri_ml_core.ProcessorCPH(
            data, undo_true_refl, correct_vis_cal_nasa_to_impf,
            cldmask, v, opts
            )

    results = []

    # run prediction
    start = time.time()
    prediction = proc.get_prediction()
    logger.info("Time for prediction CPH: {:.3f}".format(time.time() - start))
    results.append(prediction)

    if make_binary:
        # apply threshold
        binary = proc.get_binary()
        results.append(binary)
    if make_uncertainty:
        # run uncertainty calculation
        uncertainty = proc.get_uncertainty()
        results.append(uncertainty)

    return results


def predict_ctp(vis006, vis008, nir016, ir039, ir062, ir073, ir087,
                ir108, ir120, ir134, lsm, skt, solzen=None, satzen=None,
                undo_true_refl=False, correct_vis_cal_nasa_to_impf=0,
                cldmask=None, make_uncertainty=True):
    """ Run cloud top pressure (CTP) prediction. """

    logger.info('---------- RUNNING CTP ANN ----------')

    v = 'CTP'

    # put data into structure
    data = seviri_ml_core.InputData(
            vis006, vis008, nir016, ir039, ir062, ir073,
            ir087, ir108, ir120, ir134, lsm, skt, solzen, satzen
            )

    # create a processor instance
    proc = seviri_ml_core.ProcessorCTP(
            data, undo_true_refl, correct_vis_cal_nasa_to_impf,
            cldmask, v, opts
            )

    results = []

    # run prediction
    start = time.time()
    prediction = proc.get_prediction()
    logger.info("Time for prediction CTP: {:.3f}".format(time.time() - start))
    results.append(prediction)

    if make_uncertainty:
        # run uncertainty calculation
        start = time.time()
        uncertainty = proc.get_uncertainty()
        logger.info('Time for calculating uncertainty: '
                    '{:.3f}'.format(time.time() - start))
        results.append(uncertainty)

    return results


def predict_ctt(vis006, vis008, nir016, ir039, ir062, ir073, ir087,
                ir108, ir120, ir134, lsm, skt, solzen=None, satzen=None,
                undo_true_refl=False, correct_vis_cal_nasa_to_impf=0,
                cldmask=None, make_uncertainty=True):
    """ Run cloud top temperature (CTT) prediction. """

    logger.info('---------- RUNNING CTT ANN ----------')

    v = 'CTT'

    # put data into structure
    data = seviri_ml_core.InputData(
            vis006, vis008, nir016, ir039, ir062, ir073,
            ir087, ir108, ir120, ir134, lsm, skt, solzen, satzen
            )

    # create a processor instance
    proc = seviri_ml_core.ProcessorCTT(
            data, undo_true_refl, correct_vis_cal_nasa_to_impf,
            cldmask, v, opts
            )

    results = []

    # run prediction
    start = time.time()
    prediction = proc.get_prediction()
    logger.info("Time for prediction CTT: {:.3f}".format(time.time() - start))
    results.append(prediction)

    if make_uncertainty:
        # run uncertainty calculation
        start = time.time()
        uncertainty = proc.get_uncertainty()
        logger.info('Time for calculating uncertainty: '
                    '{:.3f}'.format(time.time() - start))
        results.append(uncertainty)

    return results


def predict_cbh(ir108, ir120, ir134, solzen=None, satzen=None,
                cldmask=None, make_uncertainty=True):
    """
        Main function that calls the neural network for CTT prediction.

        Input:
        - ir108 (2d numpy array):    SEVIRI IR 10.8 um (Ch 9)
        - ir120 (2d numpy array):    SEVIRI IR 12.0 um (Ch 10)
        - ir134 (2d numpy array):    SEVIRI IR 13.4 um  (Ch 11)
        - satzen (2d numpy array):   Satellite zenith angle
        - solzen (2d numpy array):   Solar zenith angle
        - cldmask (2d numpy array or None): External cloud mask.
        - make_uncertainty (bool): Generate uncertainties using DQR

        Return:
        - prediction (list): NN output list
                             [CBH, CBH 1-sigma uncertainty]
    """

    logger.info('---------- RUNNING CBH ANN ----------')

    v = 'CBH'

    # put data into structure
    data = seviri_ml_core.InputData(
        ir108=ir108, ir120=ir120, ir134=ir134,
        solzen=solzen, satzen=satzen)

    undo_true_refl = False
    correct_vis_cal_nasa_to_impf = False

    # create a processor instance
    proc = seviri_ml_core.ProcessorCBH(
                        data, undo_true_refl, correct_vis_cal_nasa_to_impf,
                        cldmask, v, opts
                        )

    results = []

    # run prediction
    start = time.time()
    prediction = proc.get_prediction()
    logger.info("Time for prediction CBH: {:.3f}".format(time.time() - start))
    results.append(prediction)

    if make_uncertainty:
        # run uncertainty calculation
        start = time.time()
        uncertainty = proc.get_uncertainty()
        logger.info('Time for calculating uncertainty: '
                    '{:.3f}'.format(time.time() - start))
        results.append(uncertainty)

    return results


def predict_mlay(vis006, vis008, nir016, ir039, ir062, ir073, ir087,
                 ir108, ir120, ir134, lsm, skt, solzen=None, satzen=None,
                 undo_true_refl=False, correct_vis_cal_nasa_to_impf=0,
                 cldmask=None, make_binary=True, make_uncertainty=True):
    """ Run multilayer flag (MLAY) prediction. """

    logger.info('---------- RUNNING MLAY ANN ----------')

    v = 'MLAY'

    # put data into structure
    data = seviri_ml_core.InputData(
            vis006, vis008, nir016, ir039, ir062, ir073,
            ir087, ir108, ir120, ir134, lsm, skt, solzen, satzen
            )

    # create a processor instance
    proc = seviri_ml_core.ProcessorMLAY(
            data, undo_true_refl, correct_vis_cal_nasa_to_impf,
            cldmask, v, opts
            )

    results = []

    # run prediction
    start = time.time()
    prediction = proc.get_prediction()
    logger.info("Time for prediction MLAY: {:.3f}".format(time.time() - start))
    results.append(prediction)

    if make_binary:
        # apply threshold
        binary = proc.get_binary()
        results.append(binary)
    if make_uncertainty:
        # run uncertainty calculation
        uncertainty = proc.get_uncertainty()
        results.append(uncertainty)

    return results
