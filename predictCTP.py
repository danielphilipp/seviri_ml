"""
    This module contains functions to do neural network CTP predictions
    based on SEVIRI measurements.

    - func _all_same(): Helper function. Checks whether all elements in a
                      list are equal.
    - func prepare_input_array(): Bring arrays in the right format for the
                                  Neural Network.
    - func getdata_dummy(): For DWD Servers. Load collocated dummy test data.
    - func predict_ANN(): To be called to execute the prediction.

    Author: Daniel Philipp (DWD)
"""

import neuralnet
import numpy as np
import time
import os
import logging
import helperfuncs as hf
from definitions import SREAL_FILL_VALUE, SREAL, IS_CLEAR
from nasa_impf_correction import correct_nasa_impf


fmt = '%(levelname)s : %(filename)s : %(message)s'
logging.basicConfig(level=logging.DEBUG,
                    format=fmt)

# set Keras backend (Theano or Tensorflow 2)
backend = os.environ.get('SEVIRI_ML_BACKEND')
if backend is not None:
    backend = backend.upper()
else:
    # default behaviour
    backend = 'TENSORFLOW2'
    logging.info('SEVIRI_ML_BACKEND env variable not '
                 'defined. Setting backend to default {}'.format(backend))

if backend in ['TENSORFLOW', 'TF', 'TF2', 'TENSORFLOW2']:
    backend = 'TENSORFLOW2'
elif backend == 'THEANO':
    backend = 'THEANO'
else:
    raise Exception('Backend {} is invalid'.format(backend))

# read nn_driver.txt
opts = hf._get_driver_opts(backend)

# configure theano compilation
if backend == 'THEANO':
    Tconfig = hf.ConfigTheano(opts)


def _prepare_input_arrays(vis006, vis008, nir016, ir039, ir062, ir073, ir087,
                          ir108, ir120, ir134, lsm, skt, solzen, satzen,
                          networks, undo_true_refl,
                          correct_vis_cal_nasa_to_impf,
                          model_version):
    """
        Prepare input array for the neural network. Takes required feature
        arrays, flattens them using row-major ordering and combines all flat
        arrays into a single array with shape (nsamples, nfeatures) to be
        used for prediction. VIS channel reflectances are expected to be in
        the 0-1 range. Conversion to 0-100 (as training data) range
        performed internally.

        Input:
        - vis006 (2d numpy array): SEVIRI VIS 0.6 um (Ch 1)
        - vis008 (2d numpy array): SEVIRI VIS 0.8 um (Ch 2)
        - nir016 (2d numpy array): SEVIRI NIR 1.6 um (Ch 3)
        - ir039 (2d numpy array):  SEVIRI IR 3.9 um  (Ch 4)
        - ir062 (2d numpy array):  SEVIRI WV 6.2 um  (Ch 5)
        - ir073 (2d numpy array):  SEVIRI WV 7.3 um  (Ch 6)
        - ir087 (2d numpy array):  SEVIRI IR 8.7 um  (Ch 7)
        - ir108 (2d numpy array):  SEVIRI IR 10.8 um (Ch 9)
        - ir120 (2d numpy array):  SEVIRI IR 12.0 um (Ch 10)
        - ir134 (2d numpy array):  SEVIRI IR 13.4 um  (Ch 11)
        - lsm (2d numpy array):    Land-sea mask
        - skt (2d numpy array):    (ERA5) Skin Temperature
        - solzen (2d numpy array or None): Solar Zenith Angle
        - satzen (2d numpy array or None): Satellite Zenith Angle
        - undo_true_refl (bool):     Remove true reflectances
                                     from VIS channels (* solzen)
        - correct_vis_cal_nasa_to_impf (bool/str):
                                     Whether to apply linear correction
                                     to convert NASA calibrated VIS
                                     channel data to IMPF calibration.
                                     0 (not applying) or
                                     [1, 2, 3, 4].
        - model_version (int): Version of models used.

        Return:
        - idata (2d numpy array): Scaled input array for ANN
    """
    # set reflectances below 0 to 0
    vis006p = vis006.copy()
    vis008p = vis008.copy()
    nir016p = nir016.copy()

    if correct_vis_cal_nasa_to_impf in [1, 2, 3, 4]:
        logging.info('Correcting VIS calibration from NASA to '
                     'IMPF for MSG{:d}'.format(correct_vis_cal_nasa_to_impf))
        c = correct_nasa_impf(vis006, vis008p, nir016p,
                              correct_vis_cal_nasa_to_impf)
        vis006p, vis008p, nir016p = c

    elif correct_vis_cal_nasa_to_impf == 0:
        logging.info('Not correcting VIS calibration from NASA to IMPF.')
    else:
        logging.info('correct_vis_cal_nasa_to_impf value {} '
                     'not known. However, not correcting VIS channel '
                     'calibration from NASA to '
                     'IMPF.'.format(correct_vis_cal_nasa_to_impf))

    vis006p[vis006p < 0] = 0
    vis008p[vis008p < 0] = 0
    nir016p[nir016p < 0] = 0

    # multiply reflectances by 100 to convert from 0-1
    # to 0-100 range as training data. Satpy outputs
    # 0-100 whereas SEVIRI util outputs 0-1.
    vis006p = vis006p * 100.
    vis008p = vis008p * 100.
    nir016p = nir016p * 100.

    # remove true reflectances
    if undo_true_refl:
        logging.info('Removing true reflectances')
        cond = np.logical_and(solzen >= 0., solzen < 90.)
        cos_sza = np.cos(np.deg2rad(solzen))
        vis006p = np.where(cond, vis006p * cos_sza, vis006p)
        vis008p = np.where(cond, vis008p * cos_sza, vis008p)
        nir016p = np.where(cond, nir016p * cos_sza, nir016p)
    else:
        logging.info('Not removing true reflectances')

    # calculate channel differences
    ir087_108 = ir087 - ir108
    ir108_120 = ir108 - ir120

    if model_version == 3:
        # list of arrays must be kept in this order!
        data_lst = [
                    ir039,      # 1
                    ir087,      # 2
                    ir087_108,  # 3
                    ir108,      # 4
                    ir108_120,  # 5
                    ir120,      # 6
                    ir134,      # 7
                    lsm,        # 8
                    nir016p,    # 9
                    satzen,     # 10
                    skt,        # 11
                    solzen,     # 12
                    vis006p,    # 13
                    vis008p,    # 14
                    ir062,      # 15
                    ir073       # 16
                    ]
    else:
        raise Exception(RuntimeError,
                        'Model version {} invalid.'
                        'Allowed is 3.'.format(model_version))

    # check if array dimensions are equal throughout all arrays
    # if all dimensions are equal: set dimension constants for reshaping
    xdims = []
    ydims = []
    for tmp in data_lst:
        xdims.append(tmp.shape[0])
        if len(tmp.shape) == 2:
            ydims.append(tmp.shape[1])
        else:
            ydims.append(1)

    if hf.all_same(xdims) and hf.all_same(ydims):
        xdim = data_lst[0].shape[0]
        if len(data_lst[0].shape) == 2:
            ydim = data_lst[0].shape[1]
            input_is_2d = True
        else:
            ydim = 1
            input_is_2d = False
    else:
        msg = 'xdim or ydim differ between input arrays for neural network.'
        raise Exception(RuntimeError, msg)

    # fill neural network input array with flattened data fields
    idata = np.empty((xdim*ydim, len(data_lst)))
    for cnt, d in enumerate(data_lst):
        tmp = d.ravel()
        idata[:, cnt] = tmp

    # check for each pixel if any channels is invalid (1), else 0
    has_invalid_item = np.any(np.where(idata < 0, 1, 0), axis=1)
    if input_is_2d:
        has_invalid_item = has_invalid_item.reshape((xdim, ydim))

    all_chs = np.array([vis006p, vis008p, nir016p, ir039, ir087,
                        ir108, ir120, ir134, ir062, ir073])

    # pixels with all IR channels invalid = 1, else 0 (as VIS can be
    # at night
    all_channels_invalid = np.all(np.where(all_chs[3:] < 0, 1, 0), axis=0)
    # indices of pixels with all channels valid
    all_channels_valid_indxs = np.nonzero(~all_channels_invalid.ravel())
    # dictionary of invalid pixel masks
    masks = {'hii': has_invalid_item,
             'aci': all_channels_invalid,
             'acvi': all_channels_valid_indxs[0]}

    scaled_data = {'CTP': networks['CTP'].scale_input(idata)}

    # apply scaling to input array
    return scaled_data, (xdim, ydim), input_is_2d, masks


def _select_network(opts):
    """ Setup configured networks """
    return {'CTP': neuralnet.NetworkCTP(opts)}


def _check_prediction(prediction, parameters, masks):
    """ Check neural net regression for invalid values. """
    # mask prediction values outside valid regression limits
    condition = np.logical_or(prediction > parameters.VALID_CTP_REGRESSION_MAX,
                              prediction < parameters.VALID_CTP_REGRESSION_MIN
                              )
    prediction = np.where(condition, SREAL_FILL_VALUE, prediction)

    # mask pixels where all channels are invalid (i.e. space pixels)
    prediction = np.where(masks['aci'] == 1, SREAL_FILL_VALUE, prediction)
    return prediction


def _uncertainty(models, input, median, variable, dims,
                 masks, parameters, input_is_2d, method):
    """ Get CTP uncertainty. Until yet, only Quantile
        Regression is implemented.
    """
    # quantile regression.
    if method.lower() in ['percentile', 'quantile', 'qrm']:
        # select scaled data for correct variable
        idata = input[variable]
        # predict only pixels indices where all channels are valid
        idata = idata[masks['acvi'], :]

        # run lower and upper percentile prediction on valid pixels
        prediction_lower = np.squeeze(models['lower'].predict(idata))
        prediction_upper = np.squeeze(models['upper'].predict(idata))
        prediction_lower = prediction_lower.astype(SREAL)
        prediction_upper = prediction_upper.astype(SREAL)

        # empty results array
        p_lower = np.ones((dims[0] * dims[1]), dtype=SREAL) * SREAL_FILL_VALUE
        p_upper = np.ones((dims[0] * dims[1]), dtype=SREAL) * SREAL_FILL_VALUE
        # fill indices of predicted pixels with predicted values
        p_lower[masks['acvi']] = prediction_lower
        p_upper[masks['acvi']] = prediction_upper

        if input_is_2d:
            p_lower = p_lower.reshape((dims[0], dims[1]))
            p_upper = p_upper.reshape((dims[0], dims[1]))

        # mask invalid pixels and set correct fill values
        p_lower = _postproc_prediction(p_lower, parameters, masks)
        p_upper = _postproc_prediction(p_upper, parameters, masks)

        # as the 1 sigma lower/upper interval is not symmetric
        # we take the mean of upper and lower
        lower_sigma = np.abs(p_lower - median)
        upper_sigma = np.abs(p_upper - median)
        mean_sigma = 0.5 * (lower_sigma + upper_sigma)
        return mean_sigma
    else:
        raise Exception('No uncertainty method except prcentile '
                        'regression implemented yet. '
                        'Set CTP_UNCERTAINTY_METHOD to '
                        'Percentile in the nn_driver.txt')


def _postproc_prediction(prediction, parameters, masks):
    """ Check invalid predictions and get uncertainty. """
    # regression
    reg = _check_prediction(prediction, parameters, masks)
    reg = np.where(~np.isfinite(reg), SREAL_FILL_VALUE, reg)
    return reg


def _run_prediction(variable, models, scaled_data, masks, dims):
    """ Run prediction with neural network. """

    # select scaled data for correct variable
    idata = scaled_data[variable]
    # predict only pixels indices where all channels are valid
    idata = idata[masks['acvi'], :]
    # run prediction on valid pixels
    prediction = np.squeeze(models['median'].predict(idata)).astype(SREAL)
    # empty results array
    pred = np.ones((dims[0]*dims[1]), dtype=SREAL) * SREAL_FILL_VALUE
    # fill indices of predicted pixels with predicted value
    pred[masks['acvi']] = prediction

    return pred


def predict_CTP(vis006, vis008, nir016, ir039, ir062, ir073, ir087,
                ir108, ir120, ir134, lsm, skt, solzen=None, satzen=None,
                undo_true_refl=False, correct_vis_cal_nasa_to_impf=0,
                cldmask=None):
    """
        Main function that calls the neural network for CTP prediction.

        Input:
        - vis006 (2d numpy array):   SEVIRI VIS 0.6 um (Ch 1)
        - vis008 (2d numpy array):   SEVIRI VIS 0.8 um (Ch 2)
        - nir016 (2d numpy array):   SEVIRI NIR 1.6 um (Ch 3)
        - ir039 (2d numpy array):    SEVIRI IR 3.9 um  (Ch 4)
        - ir062 (2d numpy array):    SEVIRI WV 6.2 um  (Ch 5)
        - ir073 (2d numpy array):    SEVIRI WV 7.3 um  (Ch 6)
        - ir087 (2d numpy array):    SEVIRI IR 8.7 um  (Ch 7)
        - ir108 (2d numpy array):    SEVIRI IR 10.8 um (Ch 9)
        - ir120 (2d numpy array):    SEVIRI IR 12.0 um (Ch 10)
        - ir134 (2d numpy array):    SEVIRI IR 13.4 um  (Ch 11)
        - lsm (2d numpy array):      Land-sea mask
        - skt (2d numpy array):      (ERA5) Skin Temperature
        - solzen (2d numpy array):   Solar zenith angle
        - undo_true_refl (bool):     Remove true reflectances
                                     from VIS channels (* solzen)
        - correct_vis_cal_nasa_to_impf (bool/str):
                                     Whether to apply linear correction
                                     to convert NASA calibrated VIS
                                     channel data to IMPF calibration.
                                     0 (not applying) or
                                     [1, 2, 3, 4].
        - cldmask (2d numpy array or None): External cloud mask.

        Return:
        - prediction (list): NN output list
                             [CMA_reg, CMA_bin, CMA_unc,
                              CPH_reg, CPH_bin, CPH_unc]
    """

    logging.info('---------- RUNNING CMA/CPH ANN ----------')

    # setup networks
    networks = _select_network(opts)

    v = 'CTP'

    # check if model versions of COT and CPH are equal
    model_version = networks[v].version

    # get parameters corresponding to variable and model version
    parameters = hf.get_parameters(model_version, v)

    # check if solzen is available if true refl should be removed
    if undo_true_refl:
        if solzen is None:
            raise Exception(RuntimeError,
                            'If undo_true_refl is true, '
                            'solzen must not be None!')

    # check if solzen and satzen are available if model version is 3
    if model_version == 3:
        if solzen is None or satzen is None:
            raise Exception(RuntimeError,
                            'If model version is 3, '
                            'solzen and satzen must not be None! '
                            'satzen is type {} and solzen '
                            'is type {}'.format(type(satzen), type(solzen)))

    # change fill value of skt from >> 1000 to SREAL_FILL_VALUE
    skt = np.where(skt > 1000, SREAL_FILL_VALUE, skt)

    # scale and put input arrays into the right format for the model
    prepped = _prepare_input_arrays(vis006, vis008, nir016,
                                    ir039, ir062, ir073,
                                    ir087, ir108, ir120,
                                    ir134, lsm, skt, solzen, satzen,
                                    networks, undo_true_refl,
                                    correct_vis_cal_nasa_to_impf,
                                    model_version)
    (scaled_data, dims, input_is_2d, masks) = prepped

    # load correct models (lower percentile, median, upper percentile)
    models = networks[v].get_model()

    # predict CTP
    start = time.time()
    prediction = _run_prediction(v, models, scaled_data, masks, dims)
    logging.info("Time for prediction CTP: {:.3f}".format(time.time() - start))

    if input_is_2d:
        prediction = prediction.reshape((dims[0], dims[1]))
    prediction = _postproc_prediction(prediction, parameters, masks)

    # get uncertainty
    start = time.time()
    uncertainty = _uncertainty(models, scaled_data, prediction, v,
                               dims, masks, parameters, input_is_2d,
                               opts['CTP_UNCERTAINTY_METHOD'])
    logging.info('Time for calculating uncertainty: '
                 '{:.3f}'.format(time.time() - start))

    results = {}
    results['reg'] = prediction
    results['unc'] = uncertainty

    # optional: mask clear sky pixels with extermal cloudmask
    if cldmask is not None:
        logging.info('Applying external cldmask to CTP')
        # mask CTP pixels where binary CMA is clear (0)
        clear_mask = cldmask == IS_CLEAR
        results['reg'][clear_mask] = SREAL_FILL_VALUE
        results['unc'][clear_mask] = SREAL_FILL_VALUE

    return [results['reg'], results['unc']]
