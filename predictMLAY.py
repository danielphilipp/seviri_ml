"""
    This module contains functions to do neural network COT (CMA) and CPH
    predictions based on SEVIRI measurements.

    - func _all_same(): Helper function. Checks whether all elements in a
                      list are equal.
    - func prepare_input_array(): Bring arrays in the right format for the
                                  Neural Network.
    - func getdata_dummy(): For DWD Servers. Load collocated dummy test data.
    - func predict_ANN(): To be called to execute the prediction.

    First author: Daniel Philipp (DWD)
    ---------------------------------------------------------------------------
    2020/07/20, DP: Initial version
    2020/07/23, DP: Added a neural network driver file containing the
                    backend, paths and thresholds
    2020/08/13, DP: Added masking of invalid (space) pixels before prediction.
                    Only valid pixels are predicted to increase efficiency.
                    Implemented correct fill value usage.
    2020/08/18, DP: Implemented fully working uncertainty estimation.
"""

import neuralnet
import numpy as np
import time
import os
import logging
import helperfuncs as hf
from definitions import (SREAL_FILL_VALUE, BYTE_FILL_VALUE, SREAL,
                         BYTE, IS_CLEAR, IS_CLOUD, IS_WATER, IS_ICE,
                         IS_SLAY, IS_MLAY)
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
    logging.info('SEVIRI_ML_BACKEND env variable not defined. '
                 'Setting backend to default {}'.format(backend))

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
                          model_version, cldmask):
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
                     'IMPF for MSG{:d}.'.format(correct_vis_cal_nasa_to_impf))
        c = correct_nasa_impf(vis006, vis008p, nir016p,
                              correct_vis_cal_nasa_to_impf)
        vis006p, vis008p, nir016p = c

    elif correct_vis_cal_nasa_to_impf == 0:
        logging.info('Not correcting VIS channel calibration '
                     'from NASA to IMPF.')
    else:
        logging.info('correct_vis_cal_nasa_to_impf value {} not known.'
                     'However, not correcting VIS calibration from NASA to '
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

    all_channels_valid = ~all_channels_invalid

    if cldmask is not None:
        # check if optional cloudmask shape is matching input data shape
        assert cldmask.shape == vis006p.shape
        # if optional cloudmask is not None mask clear pixels as well
        all_channels_valid = np.logical_and(all_channels_valid, cldmask == IS_CLOUD)

    all_channels_valid_indxs = np.nonzero(all_channels_valid.ravel())

    # dictionary of invalid pixel masks
    masks = {'hii': has_invalid_item,
             'aci': all_channels_invalid,
             'acvi': all_channels_valid_indxs[0]}

    scaled_data = {'MLAY': networks['MLAY'].scale_input(idata)}

    # apply scaling to input array
    return scaled_data, (xdim, ydim), input_is_2d, masks


def _select_networks(opts):
    """ Setup configured networks """
    networks = {'MLAY': neuralnet.NetworkMLAY(opts)}
    return networks


def _thresholding(prediction, opts):
    """ Determine binary array by applying thresholding. """
    # read threshold from driver file content
    threshold = opts.NN_MLAY_THRESHOLD
    binary = np.where(prediction > threshold, IS_MLAY, IS_SLAY)
    # mask pixels where regression array has fill value
    binary[prediction == SREAL_FILL_VALUE] = BYTE_FILL_VALUE
    return binary


def _unc_approx_1(pred, th, unc_params):
    """ Calculate uncertainty for cloudy/ice pixels. """
    norm_diff = (pred-th) / (th - 1)

    minunc = unc_params['min1']
    maxunc = unc_params['max1']

    minunc = max(minunc, 0)

    return (maxunc - minunc) * norm_diff + maxunc


def _unc_approx_0(pred, th, unc_params):
    """ Calculate uncertainty for clear/water pixels """
    norm_diff = (pred-th) / th

    minunc = unc_params['min0']
    maxunc = unc_params['max0']
    
    minunc = max(minunc, 0)

    return (maxunc - minunc) * norm_diff + maxunc


def _uncertainty(prediction, binary, variable, opts):
    """ Calculate CMA/CPH uncertainy. """
    #threshold = opts.NN_MLAY_THRESHOLD

    unc = np.ones(prediction.shape)
    unc = np.where(unc < 0, 0, unc)
    unc = np.where(unc > 100, 100, unc)
    return unc


def _check_prediction(prediction, parameters, masks):
    """ Check neural net regression for invalid values. """
    prediction = np.where(prediction > 1, 1, prediction)
    # mask pixels where all channels are invalid (i.e. space pixels)
    prediction = np.where(masks['aci'] == 1, SREAL_FILL_VALUE, prediction)
    return prediction


def _postproc_prediction(prediction, variable, parameters, masks):
    """ Check invalid predictions, apply thresholding and get uncertainty. """
    # regression
    reg = _check_prediction(prediction, parameters, masks)
    # binary cloud flag
    binary = _thresholding(prediction, parameters)
    # uncertainty
    unc = _uncertainty(prediction, binary, variable, parameters)

    # penalize cases where at least 1 input variable is invalid with higher unc
    unc = np.where(masks['hii'] == 1, unc * 1.1, unc)
    # mask cases where all channels are invalid
    binary = np.where(masks['aci'] == 1, BYTE_FILL_VALUE, binary)
    unc = np.where(masks['aci'] == 1, SREAL_FILL_VALUE, unc)

    reg = np.where(~np.isfinite(reg), SREAL_FILL_VALUE, reg)
    binary = np.where(~np.isfinite(binary), BYTE_FILL_VALUE, binary)
    unc = np.where(~np.isfinite(unc), SREAL_FILL_VALUE, unc)

    res = {'reg': reg.astype(SREAL),
           'bin': binary.astype(BYTE),
           'unc': unc.astype(SREAL)}

    return res


def _run_prediction(variable, networks, scaled_data, masks, dims):
    """ Run prediction with neural network. """
    # load correct model
    model = networks[variable].get_model()
    # select scaled data for correct variable
    idata = scaled_data[variable]
    # predict only pixels indices where all channels are valid
    idata = idata[masks['acvi'], :]
    # run prediction on valid pixels
    prediction = np.squeeze(model.predict_proba(idata)).astype(SREAL)
    # empty results array
    pred = np.ones((dims[0]*dims[1]), dtype=SREAL) * SREAL_FILL_VALUE
    # fill indices of predicted pixels with predicted value
    pred[masks['acvi']] = prediction

    return pred


def predict_MLAY(vis006, vis008, nir016, ir039, ir062, ir073, ir087,
                    ir108, ir120, ir134, lsm, skt, solzen=None, satzen=None,
                    undo_true_refl=False, correct_vis_cal_nasa_to_impf=0,
                    cldmask=None):
    """
        Main function that calls the neural network for COT and
        CPH prediction.

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

        Return:
        - prediction (list): NN output list
                             [CMA_reg, CMA_bin, CMA_unc,
                              CPH_reg, CPH_bin, CPH_unc]
    """

    logging.info('---------- RUNNING MULTILAYER ANN ----------')

    v = 'MLAY'

    # setup networks
    networks = _select_networks(opts)

    # check if model versions of COT and CPH are equal
    model_version = networks[v].version

    # get parameters such as uncertainty characterization
    # and thresholds corresponding to variable and version
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
                                    model_version, cldmask
                                    )
    (scaled_data, dims, input_is_2d, masks) = prepped

    # predict MLAY
    start = time.time()
    pred = _run_prediction(v, networks, scaled_data, masks, dims)
    if input_is_2d:
        pred = pred.reshape((dims[0], dims[1]))
    results = _postproc_prediction(pred,
                                   v,
                                   parameters,
                                   masks)
    logging.info("Time for prediction MLAY: {:.3f}".format(time.time() - start))

    return [results['reg'], results['bin'], results['unc']]
