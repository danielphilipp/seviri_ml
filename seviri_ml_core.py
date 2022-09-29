import helperfuncs as hf
import numpy as np
from definitions import (SREAL_FILL_VALUE, BYTE_FILL_VALUE, SREAL,
                         BYTE, IS_CLEAR, IS_CLOUD, IS_WATER, IS_ICE,
                         IS_MLAY, IS_SLAY)
from nasa_impf_correction import correct_nasa_impf
import neuralnet
import logging

fmt = '%(levelname)s : %(filename)s : %(message)s'
logging.basicConfig(level=logging.DEBUG,
                    format=fmt)


class ProcessorBase:
    def __init__(self, data, networks, undo_true_refl,
                 correct_vis_cal_nasa_to_impf, cldmask, variable):

        self.data = data
        self.networks = networks
        self.undo_true_refl = undo_true_refl
        self.do_correct_nasa_impf = correct_vis_cal_nasa_to_impf
        self.cldmask = cldmask
        self.variable = variable

        self.has_invalid_item = None
        self.all_channels_invalid = None
        self.all_channels_valid_idxs = None
        self.ir039_invalid = None
        self.xdim = None
        self.ydim = None
        self.input_is_2d = None
        self.scaled_data = None

    def prepare_input_arrays(self):
        # set reflectances below 0 to 0
        vis006p = self.data.vis006.copy()
        vis008p = self.data.vis008.copy()
        nir016p = self.data.nir016.copy()

        if self.do_correct_nasa_impf in [1, 2, 3, 4]:
            logging.info('Correcting VIS calibration from NASA to '
                         'IMPF for MSG{:d}'.format(self.do_correct_nasa_impf))

            c = correct_nasa_impf(vis006p, vis008p, nir016p,
                                  self.do_correct_nasa_impf)
            vis006p, vis008p, nir016p = c

        elif self.do_correct_nasa_impf == 0:
            logging.info('Not correcting VIS calibration from NASA to IMPF.')
        else:
            logging.info('correct_vis_cal_nasa_to_impf value {} '
                         'not known. However, not correcting VIS channel '
                         'calibration from NASA to '
                         'IMPF.'.format(self.do_correct_nasa_impf))

        vis006p[vis006p < 0] = 0
        vis008p[vis008p < 0] = 0
        nir016p[nir016p < 0] = 0

        # multiply reflectances by 100 to convert from 0-1
        # to 0-100 range as training data. Satpy outputs
        # 0-100 whereas SEVIRI util outputs 0-1.
        vis006p = vis006p * 100.
        vis008p = vis008p * 100.
        nir016p = nir016p * 100.

        # change fill value of skt from >> 1000 to SREAL_FILL_VALUE
        skt = np.where(self.data.skt > 1000, SREAL_FILL_VALUE, self.data.skt)

        # remove true reflectances
        if self.undo_true_refl:
            logging.info('Removing true reflectances')
            cond = np.logical_and(self.data.solzen >= 0.,
                                  self.data.solzen < 90.)
            cos_sza = np.cos(np.deg2rad(self.data.solzen))
            vis006p = np.where(cond, vis006p * cos_sza, vis006p)
            vis008p = np.where(cond, vis008p * cos_sza, vis008p)
            nir016p = np.where(cond, nir016p * cos_sza, nir016p)
        else:
            logging.info('Not removing true reflectances')

        # calculate channel differences
        ir087_108 = self.data.ir087 - self.data.ir108
        ir108_120 = self.data.ir108 - self.data.ir120

        if self.model_version in [1, 2]:
            # list of arrays must be kept in this order!
            data_lst = [
                self.data.ir039,  # 1
                self.data.ir087,  # 2
                ir087_108,        # 3
                self.data.ir108,  # 4
                ir108_120,        # 5
                self.data.ir120,  # 6
                self.data.ir134,  # 7
                self.data.lsm,    # 8
                nir016p,          # 9
                skt,              # 10
                vis006p,          # 11
                vis008p,          # 12
                self.data.ir062,  # 13
                self.data.ir073   # 14
            ]
        elif self.model_version == 3:
            # list of arrays must be kept in this order!
            data_lst = [
                self.data.ir039,   # 1
                self.data.ir087,   # 2
                ir087_108,         # 3
                self.data.ir108,   # 4
                ir108_120,         # 5
                self.data.ir120,   # 6
                self.data.ir134,   # 7
                self.data.lsm,     # 8
                nir016p,           # 9
                self.data.satzen,  # 10
                skt,               # 11
                self.data.solzen,  # 12
                vis006p,           # 13
                vis008p,           # 14
                self.data.ir062,   # 15
                self.data.ir073    # 16
            ]
        else:
            msg = 'Model version {} invalid. Allowed are ' \
                  '1, 2 and 3.'.format(self.model_version)
            raise Exception(RuntimeError, msg)

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
            self.xdim = data_lst[0].shape[0]
            if len(data_lst[0].shape) == 2:
                self.ydim = data_lst[0].shape[1]
                self.input_is_2d = True
            else:
                self.ydim = 1
                self.input_is_2d = False
        else:
            msg = 'xdim or ydim differ between input arrays.'
            raise Exception(RuntimeError, msg)

        # fill neural network input array with flattened data fields
        idata = np.empty((self.xdim * self.ydim, len(data_lst)))
        for cnt, d in enumerate(data_lst):
            tmp = d.ravel()
            idata[:, cnt] = tmp

        # check for each pixel if any channels is invalid (1), else 0
        has_invalid_item = np.any(np.where(idata < 0, 1, 0), axis=1)
        if self.input_is_2d:
            has_invalid_item = has_invalid_item.reshape((self.xdim, self.ydim))

        if self.opts['CORRECT_IR039_OUT_OF_RANGE']:
            if self.variable in ['CMA', 'CPH', 'MLAY', 'CTP', 'CTT']:
                # check if all channels are valid and ir039 invalid
                # list of all channels except 039
                all_chs_exc_039 = np.array(
                            [self.data.ir087, self.data.ir108, self.data.ir120,
                             self.data.ir134, self.data.ir062, self.data.ir073]
                            )
                # check if all ir channels except 039 are valid
                all_channels_valid_exc_039 = np.all(
                    np.where(all_chs_exc_039 > 0, 1, 0), axis=0)
                # 039 is invalid if it is < 0 or NaN
                ir039_invalid = np.logical_or(
                            np.isnan(self.data.ir039),
                            self.data.ir039 < 0
                            )
                # 039 can be invalid if it is a space pixel.
                # Don't use those cases
                ir039_invalid_disk = np.where(np.logical_and(
                    ir039_invalid,
                    all_channels_valid_exc_039
                    ), 1, 0)

                self.ir039_invalid = ir039_invalid_disk
                self.n_ir039_invalid = np.sum(self.ir039_invalid)
                logging.info('N_IR039_INVALID: ' + str(self.n_ir039_invalid))

                    
        all_chs = np.array([vis006p, vis008p, nir016p, self.data.ir039,
                            self.data.ir087, self.data.ir108, self.data.ir120,
                            self.data.ir134, self.data.ir062, self.data.ir073])

        # pixels with all IR channels invalid = 1, else 0 (as VIS can be
        # at night
        all_channels_invalid = np.all(np.where(all_chs[3:] < 0, 1, 0), axis=0)
        all_channels_valid = ~all_channels_invalid

        if self.cldmask is not None:
            # check if optional cloudmask shape is matching input data shape
            assert self.cldmask.shape == vis006p.shape
            # if optional cloudmask is not None mask clear pixels as well
            all_channels_valid = np.logical_and(all_channels_valid,
                                                self.cldmask == IS_CLOUD)

        all_channels_valid_idxs = np.nonzero(all_channels_valid.ravel())

        self.has_invalid_item = has_invalid_item
        self.all_channels_invalid = all_channels_invalid
        self.all_channels_valid_idxs = all_channels_valid_idxs[0]

        self.scaled_data = self.networks.scale_input(idata)

        if self.opts['CORRECT_IR039_OUT_OF_RANGE']:
            # if CTP or CTT replace invalid 3.9 pixel BT with 10.8 BT
            if self.variable in ['CTP', 'CTT']:
                self.scaled_data[:, 0] = np.where(self.ir039_invalid.ravel() == 1,
                                                  self.scaled_data[:, 3],
                                                  self.scaled_data[:, 0])

class InputData:
    def __init__(self, vis006, vis008, nir016, ir039, ir062, ir073, ir087,
                 ir108, ir120, ir134, lsm, skt, solzen, satzen):
        self.vis006 = vis006
        self.vis008 = vis008
        self.nir016 = nir016
        self.ir039 = ir039
        self.ir062 = ir062
        self.ir073 = ir073
        self.ir087 = ir087
        self.ir108 = ir108
        self.ir120 = ir120
        self.ir134 = ir134
        self.lsm = lsm
        self.skt = skt
        self.solzen = solzen
        self.satzen = satzen


class ProcessorCMA(ProcessorBase):
    def __init__(self, data, undo_true_refl, correct_vis_cal_nasa_to_impf,
                 cldmask, variable, opts):

        self.undo_true_refl = undo_true_refl
        self.do_correct_nasa_impf = correct_vis_cal_nasa_to_impf
        self.cldmask = cldmask
        self.variable = variable
        self.opts = opts

        self.models = None
        self.estimate = None
        self.binary = None
        self.uncertainty = None

        self.prediction_done = False
        self.binary_done = False
        self.uncertainty_done = False

        # setup networks
        self.networks = neuralnet.NetworkCMA(opts)
        # get model version
        self.model_version = self.networks.version
        # get parameters corresponding to variable and model version
        self.parameters = hf.get_parameters(self.model_version, variable)

        # check if solzen is available if true refl should be removed
        if undo_true_refl:
            if data.solzen is None:
                raise Exception(RuntimeError,
                                'If undo_true_refl is true, '
                                'solzen must not be None!')

        # check if solzen and satzen are available if model version is 3
        if self.model_version == 3:
            if data.solzen is None or data.satzen is None:
                raise Exception(RuntimeError,
                                'If model version is 3, '
                                'solzen and satzen must not be None! '
                                'satzen is type {} and solzen '
                                'is type {}'.format(type(data.satzen),
                                                    type(data.solzen)))

        super().__init__(data, self.networks, undo_true_refl,
                         correct_vis_cal_nasa_to_impf, cldmask, variable)

    def _check_prediction(self, prediction):
        prediction = np.where(prediction > 1, 1, prediction)
        if self.opts['CORRECT_IR039_OUT_OF_RANGE']:
            if self.n_ir039_invalid > 0:
                # modify pixels where ir039 is invalid due to really cold
                # clouds (out of range for 039 channel sometimes at night)
                # set predicted COT of invalid ir039 pixels to 1.0
                prediction = np.where(self.ir039_invalid == 1, 1, prediction)

        # mask pixels where all channels are invalid (i.e. space pixels)
        prediction = np.where(self.all_channels_invalid == 1,
                              SREAL_FILL_VALUE,
                              prediction)
        prediction = np.where(~np.isfinite(prediction),
                              SREAL_FILL_VALUE,
                              prediction)
        prediction = prediction.astype(SREAL)
        return prediction

    def get_prediction(self):
        # run data preparation
        self.prepare_input_arrays()
        # load HDF5 model
        models = self.networks.get_model()
        self.models = models

        # select scaled data for correct variable
        idata = self.scaled_data
        # predict only pixels indices where all channels are valid
        idata = idata[self.all_channels_valid_idxs, :]
        # run prediction on valid pixels
        prediction = np.squeeze(models.predict(idata)).astype(SREAL)
        # empty results array
        pred = np.ones((self.xdim * self.ydim),
                       dtype=SREAL) * SREAL_FILL_VALUE
        # fill indices of predicted pixels with predicted value
        pred[self.all_channels_valid_idxs] = prediction

        if self.input_is_2d:
            estimate = pred.reshape((self.xdim, self.ydim))
        else:
            estimate = pred

        estimate = self._check_prediction(estimate)
        self.estimate = estimate
        self.prediction_done = True

        return estimate

    def get_binary(self):
        if self.prediction_done:
            # binary cloud flag
            binary = self._thresholding()
            if self.opts['CORRECT_IR039_OUT_OF_RANGE']:
                if self.n_ir039_invalid > 0:
                    # modify pixels where ir039 is invalid due to really cold
                    # clouds (out of range for 039 channel sometimes at night)
                    # set invalid ir039 pixels to cloudy
                    binary = np.where(self.ir039_invalid == 1, IS_CLOUD,
                                      binary)

            binary = np.where(self.all_channels_invalid, BYTE_FILL_VALUE,
                              binary)
            binary = np.where(~np.isfinite(binary), BYTE_FILL_VALUE,
                              binary)
            binary = binary.astype(BYTE)
            self.binary = binary
            self.binary_done = True
            return binary
        else:
            raise Exception('get_binary(): Prediction must be done before '
                            'making binary classification.')

    def get_uncertainty(self):
        if self.binary_done and self.prediction_done:
            # uncertainty
            unc = self._uncertainty()
            if self.opts['CORRECT_IR039_OUT_OF_RANGE']:
                if self.n_ir039_invalid > 0:
                    # modify pixels where ir039 is invalid due to really cold
                    # clouds (out of range for 039 channel sometimes at night)
                    # set uncertainty to max uncertainty
                    unc = np.where(self.ir039_invalid == 1,
                                   self.parameters.UNC_INTERCEPT_CLD, unc)

            # penalize cases where at least 1 input variable is invalid with
            # higher unc
            unc = np.where(self.has_invalid_item, unc * 1.1, unc)
            # mask cases where all channels are invalid
            unc = np.where(self.all_channels_invalid,
                           SREAL_FILL_VALUE, unc)
            unc = np.where(~np.isfinite(unc), SREAL_FILL_VALUE, unc)
            unc = np.where(self.estimate == SREAL_FILL_VALUE,
                           SREAL_FILL_VALUE,
                           unc)
            unc = unc.astype(SREAL)
            self.uncertainty = unc
            self.uncertainty_done = True
            return unc
        else:
            raise Exception('get_uncertainty(): Prediction and binary '
                            'classificationmust be done before '
                            'calculating uncertainty.')

    def _thresholding(self):
        """ Determine binary array by applying thresholding. """
        # get threshold from driver file content
        threshold = self.parameters.NN_COT_THRESHOLD
        # apply threshold
        binary = np.where(self.estimate > threshold, IS_CLOUD, IS_CLEAR)
        # mask pixels where regression array has fill value
        binary[self.estimate == SREAL_FILL_VALUE] = BYTE_FILL_VALUE
        return binary

    def _uncertainty(self):
        """ Calculate CMA/CPH uncertainy. """
        opts = self.parameters

        threshold = opts.NN_COT_THRESHOLD

        unc_params = {
            'min1': opts.UNC_SLOPE_CLD + opts.UNC_INTERCEPT_CLD,
            'max1': opts.UNC_INTERCEPT_CLD,
            'min0': -opts.UNC_SLOPE_CLR + opts.UNC_INTERCEPT_CLR,
            'max0': opts.UNC_INTERCEPT_CLR
        }

        unc = np.where(self.binary > IS_CLEAR,
                       self._unc_approx_1(self.estimate,
                                          threshold,
                                          unc_params
                                          ),  # where water
                       self._unc_approx_0(self.estimate,
                                          threshold,
                                          unc_params
                                          )  # where ice
                       )

        unc = np.where(unc <= 0, 0, unc)
        unc = np.where(unc > 100, 100, unc)
        return unc

    def _unc_approx_1(self, pred, th, unc_params):
        """ Calculate uncertainty for cloudy/ice pixels. """
        norm_diff = (pred - th) / (th - 1)

        minunc = unc_params['min1']
        maxunc = unc_params['max1']

        minunc = max(minunc, 0)

        return (maxunc - minunc) * norm_diff + maxunc

    def _unc_approx_0(self, pred, th, unc_params):
        """ Calculate uncertainty for clear/water pixels """
        norm_diff = (pred - th) / th

        minunc = unc_params['min0']
        maxunc = unc_params['max0']

        minunc = max(minunc, 0)

        return (maxunc - minunc) * norm_diff + maxunc


class ProcessorCPH(ProcessorBase):
    def __init__(self, data, undo_true_refl, correct_vis_cal_nasa_to_impf,
                 cldmask, variable, opts):

        self.undo_true_refl = undo_true_refl
        self.do_correct_nasa_impf = correct_vis_cal_nasa_to_impf
        self.cldmask = cldmask
        self.variable = variable
        self.opts = opts

        self.models = None
        self.estimate = None
        self.binary = None
        self.uncertainty = None

        self.prediction_done = False
        self.binary_done = False
        self.uncertainty_done = False

        # setup networks
        self.networks = neuralnet.NetworkCPH(opts)
        # get model version
        self.model_version = self.networks.version
        # get parameters corresponding to variable and model version
        self.parameters = hf.get_parameters(self.model_version, variable)

        # check if solzen is available if true refl should be removed
        if undo_true_refl:
            if data.solzen is None:
                raise Exception(RuntimeError,
                                'If undo_true_refl is true, '
                                'solzen must not be None!')

        # check if solzen and satzen are available if model version is 3
        if self.model_version == 3:
            if data.solzen is None or data.satzen is None:
                raise Exception(RuntimeError,
                                'If model version is 3, '
                                'solzen and satzen must not be None! '
                                'satzen is type {} and solzen '
                                'is type {}'.format(type(data.satzen),
                                                    type(data.solzen)))

        super().__init__(data, self.networks, undo_true_refl,
                         correct_vis_cal_nasa_to_impf, cldmask, variable)

    def _check_prediction(self, prediction):
        prediction = np.where(prediction > 1, 1, prediction)
        if self.opts['CORRECT_IR039_OUT_OF_RANGE']:
            if self.n_ir039_invalid > 0:
                # modify pixels where ir039 is invalid due to really cold
                # clouds (out of range for 039 channel sometimes at night)
                # set invalid ir039 pixels to predicted 1.0
                prediction = np.where(self.ir039_invalid == 1, 1, prediction)

        # mask pixels where all channels are invalid (i.e. space pixels)
        prediction = np.where(self.all_channels_invalid == 1,
                              SREAL_FILL_VALUE,
                              prediction)
        prediction = np.where(~np.isfinite(prediction),
                              SREAL_FILL_VALUE,
                              prediction)
        prediction = prediction.astype(SREAL)
        return prediction

    def get_prediction(self):
        # run data preparation
        self.prepare_input_arrays()
        # load HDF5 model
        models = self.networks.get_model()
        self.models = models

        # select scaled data for correct variable
        idata = self.scaled_data
        # predict only pixels indices where all channels are valid
        idata = idata[self.all_channels_valid_idxs, :]
        # run prediction on valid pixels
        prediction = np.squeeze(models.predict(idata)).astype(SREAL)
        # empty results array
        pred = np.ones((self.xdim * self.ydim), dtype=SREAL) * SREAL_FILL_VALUE
        # fill indices of predicted pixels with predicted value
        pred[self.all_channels_valid_idxs] = prediction

        if self.input_is_2d:
            estimate = pred.reshape((self.xdim, self.ydim))
        else:
            estimate = pred

        estimate = self._check_prediction(estimate)
        if self.cldmask is not None:
            estimate = np.where(self.cldmask == IS_CLEAR,
                                SREAL_FILL_VALUE,
                                estimate)
        self.estimate = estimate
        self.prediction_done = True

        return estimate

    def get_binary(self):
        if self.prediction_done:
            # binary cloud flag
            binary = self._thresholding()
            if self.opts['CORRECT_IR039_OUT_OF_RANGE']:
                if self.n_ir039_invalid > 0:
                    # modify pixels where ir039 is invalid due to really cold
                    # clouds (out of range for 039 channel sometimes at night)
                    # set invalid ir039 pixels to ice
                    binary = np.where(self.ir039_invalid == 1, IS_ICE,
                                      binary)

            binary = np.where(self.all_channels_invalid,
                              BYTE_FILL_VALUE,
                              binary)
            binary = np.where(~np.isfinite(binary),
                              BYTE_FILL_VALUE,
                              binary)
            binary = binary.astype(BYTE)

            self.binary = binary
            self.binary_done = True
            return binary
        else:
            raise Exception('get_binary(): Prediction must be done before '
                            'making binary classification.')

    def get_uncertainty(self):
        if self.binary_done and self.prediction_done:
            # uncertainty
            unc = self._uncertainty()
            if self.opts['CORRECT_IR039_OUT_OF_RANGE']:
                if self.n_ir039_invalid > 0:
                    # modify pixels where ir039 is invalid due to really cold
                    # clouds (out of range for 039 channel sometimes at night)
                    # set uncertainty to max uncertainty
                    unc = np.where(self.ir039_invalid == 1,
                                   self.parameters.UNC_INTERCEPT_ICE, unc)

            # penalize cases where at least 1 input variable is invalid with
            # higher unc
            unc = np.where(self.has_invalid_item, unc * 1.1, unc)
            # mask cases where all channels are invalid
            unc = np.where(self.all_channels_invalid == 1,
                           SREAL_FILL_VALUE,
                           unc)
            unc = np.where(~np.isfinite(unc),
                           SREAL_FILL_VALUE,
                           unc)
            unc = unc.astype(SREAL)
            if self.cldmask is not None:
                unc = np.where(self.cldmask == IS_CLEAR,
                               SREAL_FILL_VALUE,
                               unc)
            self.uncertainty = unc
            self.uncertainty_done = True
            return unc
        else:
            raise Exception('get_uncertainty(): Prediction and binary '
                            'classificationmust be done before '
                            'calculating uncertainty.')

    def _thresholding(self):
        """ Determine binary array by applying thresholding. """
        # get threshold from driver file content
        threshold = self.parameters.NN_CPH_THRESHOLD
        # apply threshold
        binary = np.where(self.estimate > threshold, IS_ICE, IS_WATER)
        # mask pixels where regression array has fill value
        binary[self.estimate == SREAL_FILL_VALUE] = BYTE_FILL_VALUE
        return binary

    def _uncertainty(self):
        """ Calculate CMA/CPH uncertainy. """
        opts = self.parameters

        threshold = opts.NN_CPH_THRESHOLD

        unc_params = {
            'min1': opts.UNC_SLOPE_ICE + opts.UNC_INTERCEPT_ICE,
            'max1': opts.UNC_INTERCEPT_ICE,
            'min0': -opts.UNC_SLOPE_LIQ + opts.UNC_INTERCEPT_LIQ,
            'max0': opts.UNC_INTERCEPT_LIQ
                      }

        unc = np.where(self.binary > IS_WATER,
                       self._unc_approx_1(self.estimate,
                                          threshold,
                                          unc_params
                                          ),  # where water
                       self._unc_approx_0(self.estimate,
                                          threshold,
                                          unc_params
                                          )  # where ice
                       )
        unc = np.where(unc < 0, 0, unc)
        unc = np.where(unc > 100, 100, unc)
        return unc

    def _unc_approx_1(self, pred, th, unc_params):
        """ Calculate uncertainty for cloudy/ice pixels. """
        norm_diff = (pred - th) / (th - 1)

        minunc = unc_params['min1']
        maxunc = unc_params['max1']

        minunc = max(minunc, 0)

        return (maxunc - minunc) * norm_diff + maxunc

    def _unc_approx_0(self, pred, th, unc_params):
        """ Calculate uncertainty for clear/water pixels """
        norm_diff = (pred - th) / th

        minunc = unc_params['min0']
        maxunc = unc_params['max0']

        minunc = max(minunc, 0)

        return (maxunc - minunc) * norm_diff + maxunc


class ProcessorCTP(ProcessorBase):
    def __init__(self, data, undo_true_refl, correct_vis_cal_nasa_to_impf,
                 cldmask, variable, opts):

        self.undo_true_refl = undo_true_refl
        self.do_correct_nasa_impf = correct_vis_cal_nasa_to_impf
        self.cldmask = cldmask
        self.variable = variable
        self.opts = opts

        self.models = None
        self.estimate = None
        self.uncertainty = None

        # setup networks
        self.networks = neuralnet.NetworkCTP(opts)
        # get model version
        self.model_version = self.networks.version
        # get parameters corresponding to variable and model version
        self.parameters = hf.get_parameters(self.model_version, variable)

        # check if solzen is available if true refl should be removed
        if undo_true_refl:
            if data.solzen is None:
                raise Exception(RuntimeError,
                                'If undo_true_refl is true, '
                                'solzen must not be None!')

        # check if solzen and satzen are available if model version is 3
        if self.model_version == 3:
            if data.solzen is None or data.satzen is None:
                raise Exception(RuntimeError,
                                'If model version is 3, '
                                'solzen and satzen must not be None! '
                                'satzen is type {} and solzen '
                                'is type {}'.format(type(data.satzen),
                                                    type(data.solzen)))

        super().__init__(data, self.networks, undo_true_refl,
                         correct_vis_cal_nasa_to_impf, cldmask, variable)

    def get_prediction(self):
        # run data preparation
        self.prepare_input_arrays()
        # load HDF5 model
        models = self.networks.get_model()
        self.models = models

        # select scaled data for correct variable
        idata = self.scaled_data
        # predict only pixels indices where all channels are valid
        idata = idata[self.all_channels_valid_idxs, :]
        # run prediction on valid pixels
        prediction = np.squeeze(models['median'].predict(idata)).astype(SREAL)
        # empty results array
        pred = np.ones((self.xdim * self.ydim), dtype=SREAL) * SREAL_FILL_VALUE
        # fill indices of predicted pixels with predicted value
        pred[self.all_channels_valid_idxs] = prediction

        if self.input_is_2d:
            estimate = pred.reshape((self.xdim, self.ydim))
        else:
            estimate = pred

        estimate = self._check_prediction(estimate)
        if self.cldmask is not None:
            estimate = np.where(self.cldmask == IS_CLEAR,
                                SREAL_FILL_VALUE,
                                estimate)
        self.estimate = estimate
        return estimate

    def _check_prediction(self, data):
        # mask pixels outside valid range
        condition = np.logical_or(
            data > self.parameters.VALID_CTP_REGRESSION_MAX,
            data < self.parameters.VALID_CTP_REGRESSION_MIN
            )
        data = np.where(condition, SREAL_FILL_VALUE, data)

        # mask pixels where all channels are invalid (i.e. space pixels)
        data = np.where(self.all_channels_invalid == 1,
                        SREAL_FILL_VALUE,
                        data)
        data = np.where(~np.isfinite(data),
                        SREAL_FILL_VALUE,
                        data)
        return data

    def get_uncertainty(self):
        unc_method = self.opts['CTP_UNCERTAINTY_METHOD']
        median = self.estimate
        # quantile regression.
        if unc_method.lower() in ['percentile', 'quantile', 'qrm']:
            # select scaled data for correct variable
            idata = self.scaled_data
            # predict only pixels indices where all channels are valid
            idata = idata[self.all_channels_valid_idxs, :]

            # run lower and upper percentile prediction on valid pixels
            prediction_lower = np.squeeze(self.models['lower'].predict(idata))
            prediction_upper = np.squeeze(self.models['upper'].predict(idata))
            prediction_lower = prediction_lower.astype(SREAL)
            prediction_upper = prediction_upper.astype(SREAL)

            # empty results array
            p_lower = np.ones((self.xdim * self.ydim),
                              dtype=SREAL) * SREAL_FILL_VALUE
            p_upper = np.ones((self.xdim * self.ydim),
                              dtype=SREAL) * SREAL_FILL_VALUE

            # fill indices of predicted pixels with predicted values
            p_lower[self.all_channels_valid_idxs] = prediction_lower
            p_upper[self.all_channels_valid_idxs] = prediction_upper

            if self.input_is_2d:
                p_lower = p_lower.reshape((self.xdim, self.ydim))
                p_upper = p_upper.reshape((self.xdim, self.ydim))

            # mask invalid pixels and set correct fill values
            p_lower = self._check_prediction(p_lower)
            p_upper = self._check_prediction(p_upper)

            # as the 1 sigma lower/upper interval is not symmetric
            # we take the mean of upper and lower
            lower_sigma = np.abs(p_lower - median)
            upper_sigma = np.abs(p_upper - median)
            mean_sigma = 0.5 * (lower_sigma + upper_sigma)
            if self.cldmask is not None:
                mean_sigma = np.where(self.cldmask == IS_CLEAR,
                                      SREAL_FILL_VALUE,
                                      mean_sigma)
            self.uncertainty = mean_sigma
            return mean_sigma
        else:
            raise Exception('No uncertainty method except prcentile '
                            'regression implemented yet. '
                            'Set CTP_UNCERTAINTY_METHOD to '
                            'Percentile in the nn_driver.txt')


class ProcessorCTT(ProcessorBase):
    def __init__(self, data, undo_true_refl, correct_vis_cal_nasa_to_impf,
                 cldmask, variable, opts):

        self.undo_true_refl = undo_true_refl
        self.do_correct_nasa_impf = correct_vis_cal_nasa_to_impf
        self.cldmask = cldmask
        self.variable = variable
        self.opts = opts

        self.models = None
        self.estimate = None
        self.uncertainty = None

        # setup networks
        self.networks = neuralnet.NetworkCTT(opts)
        # get model version
        self.model_version = self.networks.version
        # get parameters corresponding to variable and model version
        self.parameters = hf.get_parameters(self.model_version, variable)

        # check if solzen is available if true refl should be removed
        if undo_true_refl:
            if data.solzen is None:
                raise Exception(RuntimeError,
                                'If undo_true_refl is true, '
                                'solzen must not be None!')

        # check if solzen and satzen are available if model version is 3
        if self.model_version == 3:
            if data.solzen is None or data.satzen is None:
                raise Exception(RuntimeError,
                                'If model version is 3, '
                                'solzen and satzen must not be None! '
                                'satzen is type {} and solzen '
                                'is type {}'.format(type(data.satzen),
                                                    type(data.solzen)))

        super().__init__(data, self.networks, undo_true_refl,
                         correct_vis_cal_nasa_to_impf, cldmask, variable)

    def get_prediction(self):
        # run data preparation
        self.prepare_input_arrays()
        # load HDF5 model
        models = self.networks.get_model()
        self.models = models

        # select scaled data for correct variable
        idata = self.scaled_data
        # predict only pixels indices where all channels are valid
        idata = idata[self.all_channels_valid_idxs, :]
        # run prediction on valid pixels
        prediction = np.squeeze(models['median'].predict(idata)).astype(SREAL)
        # empty results array
        pred = np.ones((self.xdim * self.ydim), dtype=SREAL) * SREAL_FILL_VALUE
        # fill indices of predicted pixels with predicted value
        pred[self.all_channels_valid_idxs] = prediction

        if self.input_is_2d:
            estimate = pred.reshape((self.xdim, self.ydim))
        else:
            estimate = pred

        estimate = self._check_prediction(estimate)
        if self.cldmask is not None:
            estimate = np.where(self.cldmask == IS_CLEAR,
                                SREAL_FILL_VALUE,
                                estimate)
        self.estimate = estimate
        return estimate

    def _check_prediction(self, data):
        # mask pixels outside valid range
        condition = np.logical_or(
            data > self.parameters.VALID_CTT_REGRESSION_MAX,
            data < self.parameters.VALID_CTT_REGRESSION_MIN
            )
        data = np.where(condition, SREAL_FILL_VALUE, data)

        # mask pixels where all channels are invalid (i.e. space pixels)
        data = np.where(self.all_channels_invalid == 1,
                        SREAL_FILL_VALUE,
                        data)
        data = np.where(~np.isfinite(data),
                        SREAL_FILL_VALUE,
                        data)
        return data

    def get_uncertainty(self):
        unc_method = self.opts['CTT_UNCERTAINTY_METHOD']
        median = self.estimate
        # quantile regression.
        if unc_method.lower() in ['percentile', 'quantile', 'qrm']:
            # select scaled data for correct variable
            idata = self.scaled_data
            # predict only pixels indices where all channels are valid
            idata = idata[self.all_channels_valid_idxs, :]

            # run lower and upper percentile prediction on valid pixels
            prediction_lower = np.squeeze(self.models['lower'].predict(idata))
            prediction_upper = np.squeeze(self.models['upper'].predict(idata))
            prediction_lower = prediction_lower.astype(SREAL)
            prediction_upper = prediction_upper.astype(SREAL)

            # empty results array
            p_lower = np.ones((self.xdim * self.ydim),
                              dtype=SREAL) * SREAL_FILL_VALUE
            p_upper = np.ones((self.xdim * self.ydim),
                              dtype=SREAL) * SREAL_FILL_VALUE

            # fill indices of predicted pixels with predicted values
            p_lower[self.all_channels_valid_idxs] = prediction_lower
            p_upper[self.all_channels_valid_idxs] = prediction_upper

            if self.input_is_2d:
                p_lower = p_lower.reshape((self.xdim, self.ydim))
                p_upper = p_upper.reshape((self.xdim, self.ydim))

            # mask invalid pixels and set correct fill values
            p_lower = self._check_prediction(p_lower)
            p_upper = self._check_prediction(p_upper)

            # as the 1 sigma lower/upper interval is not symmetric
            # we take the mean of upper and lower
            lower_sigma = np.abs(p_lower - median)
            upper_sigma = np.abs(p_upper - median)
            mean_sigma = 0.5 * (lower_sigma + upper_sigma)
            if self.cldmask is not None:
                mean_sigma = np.where(self.cldmask == IS_CLEAR,
                                      SREAL_FILL_VALUE,
                                      mean_sigma)
            self.uncertainty = mean_sigma
            return mean_sigma
        else:
            raise Exception('No uncertainty method except prcentile '
                            'regression implemented yet. '
                            'Set CTP_UNCERTAINTY_METHOD to '
                            'Percentile in the nn_driver.txt')


class ProcessorMLAY(ProcessorBase):
    def __init__(self, data, undo_true_refl, correct_vis_cal_nasa_to_impf,
                 cldmask, variable, opts):

        self.undo_true_refl = undo_true_refl
        self.do_correct_nasa_impf = correct_vis_cal_nasa_to_impf
        self.cldmask = cldmask
        self.variable = variable
        self.opts = opts

        self.models = None
        self.estimate = None
        self.binary = None
        self.uncertainty = None

        self.prediction_done = False
        self.binary_done = False
        self.uncertainty_done = False

        # setup networks
        self.networks = neuralnet.NetworkMLAY(opts)
        # get model version
        self.model_version = self.networks.version
        # get parameters corresponding to variable and model version
        self.parameters = hf.get_parameters(self.model_version, variable)

        # check if solzen is available if true refl should be removed
        if undo_true_refl:
            if data.solzen is None:
                raise Exception(RuntimeError,
                                'If undo_true_refl is true, '
                                'solzen must not be None!')

        # check if solzen and satzen are available if model version is 3
        if self.model_version == 3:
            if data.solzen is None or data.satzen is None:
                raise Exception(RuntimeError,
                                'If model version is 3, '
                                'solzen and satzen must not be None! '
                                'satzen is type {} and solzen '
                                'is type {}'.format(type(data.satzen),
                                                    type(data.solzen)))

        super().__init__(data, self.networks, undo_true_refl,
                         correct_vis_cal_nasa_to_impf, cldmask, variable)

    def _check_prediction(self, prediction):
        prediction = np.where(prediction > 1, 1, prediction)
        # mask pixels where all channels are invalid (i.e. space pixels)
        prediction = np.where(self.all_channels_invalid == 1,
                              SREAL_FILL_VALUE,
                              prediction)
        prediction = np.where(~np.isfinite(prediction),
                              SREAL_FILL_VALUE,
                              prediction)
        prediction = prediction.astype(SREAL)
        return prediction

    def get_prediction(self):
        # run data preparation
        self.prepare_input_arrays()
        # load HDF5 model
        models = self.networks.get_model()
        self.models = models

        # select scaled data for correct variable
        idata = self.scaled_data
        # predict only pixels indices where all channels are valid
        idata = idata[self.all_channels_valid_idxs, :]
        # run prediction on valid pixels
        prediction = np.squeeze(models.predict(idata)).astype(SREAL)
        # empty results array
        pred = np.ones((self.xdim * self.ydim),
                       dtype=SREAL) * SREAL_FILL_VALUE
        # fill indices of predicted pixels with predicted value
        pred[self.all_channels_valid_idxs] = prediction

        if self.input_is_2d:
            estimate = pred.reshape((self.xdim, self.ydim))
        else:
            estimate = pred

        estimate = self._check_prediction(estimate)
        self.estimate = estimate
        self.prediction_done = True

        return estimate

    def get_binary(self):
        if self.prediction_done:
            # binary cloud flag
            binary = self._thresholding()
            binary = np.where(self.all_channels_invalid, BYTE_FILL_VALUE,
                              binary)
            binary = np.where(~np.isfinite(binary), BYTE_FILL_VALUE,
                              binary)
            binary = binary.astype(BYTE)
            self.binary = binary
            self.binary_done = True
            return binary
        else:
            raise Exception('get_binary(): Prediction must be done before '
                            'making binary classification.')

    def get_uncertainty(self):
        if self.binary_done and self.prediction_done:
            # uncertainty
            unc = self._uncertainty()
            if self.opts['CORRECT_IR039_OUT_OF_RANGE']:
                if self.n_ir039_invalid > 0:
                    # modify pixels where ir039 is invalid due to really cold
                    # clouds (out of range for 039 channel sometimes at night)
                    # set uncertainty to max uncertainty
                    unc = np.where(self.ir039_invalid == 1,
                                   self.parameters.UNC_INTERCEPT_MLAY, unc)

            # penalize cases where at least 1 input variable is invalid with
            # higher unc
            unc = np.where(self.has_invalid_item, unc * 1.1, unc)
            # mask cases where all channels are invalid
            unc = np.where(self.all_channels_invalid == 1,
                           SREAL_FILL_VALUE,
                           unc)
            unc = np.where(~np.isfinite(unc),
                           SREAL_FILL_VALUE,
                           unc)
            unc = unc.astype(SREAL)
            if self.cldmask is not None:
                unc = np.where(self.cldmask == IS_CLEAR,
                               SREAL_FILL_VALUE,
                               unc)
            self.uncertainty = unc
            self.uncertainty_done = True
            return unc
        else:
            raise Exception('get_uncertainty(): Prediction and binary '
                            'classificationmust be done before '
                            'calculating uncertainty.')

    def _thresholding(self):
        """ Determine binary array by applying thresholding. """
        # get threshold from driver file content
        threshold = self.parameters.NN_MLAY_THRESHOLD
        # apply threshold
        binary = np.where(self.estimate > threshold, IS_MLAY, IS_SLAY)
        # mask pixels where regression array has fill value
        binary[self.estimate == SREAL_FILL_VALUE] = BYTE_FILL_VALUE
        return binary


    def _uncertainty(self):
        """ Calculate CMA/CPH uncertainy. """
        opts = self.parameters

        threshold = opts.NN_MLAY_THRESHOLD

        unc_params = {
            'min1': opts.UNC_SLOPE_MLAY + opts.UNC_INTERCEPT_MLAY,
            'max1': opts.UNC_INTERCEPT_MLAY,
            'min0': -opts.UNC_SLOPE_SLAY + opts.UNC_INTERCEPT_SLAY,
            'max0': opts.UNC_INTERCEPT_SLAY
                      }

        unc = np.where(self.binary > IS_SLAY,
                       self._unc_approx_1(self.estimate,
                                          threshold,
                                          unc_params
                                          ),  # where water
                       self._unc_approx_0(self.estimate,
                                          threshold,
                                          unc_params
                                          )  # where ice
                       )
        unc = np.where(unc < 0, 0, unc)
        unc = np.where(unc > 100, 100, unc)
        return unc

    def _unc_approx_1(self, pred, th, unc_params):
        """ Calculate uncertainty for cloudy/ice pixels. """
        norm_diff = (pred - th) / (th - 1)

        minunc = unc_params['min1']
        maxunc = unc_params['max1']

        minunc = max(minunc, 0)

        return (maxunc - minunc) * norm_diff + maxunc

    def _unc_approx_0(self, pred, th, unc_params):
        """ Calculate uncertainty for clear/water pixels """
        norm_diff = (pred - th) / th

        minunc = unc_params['min0']
        maxunc = unc_params['max0']

        minunc = max(minunc, 0)

        return (maxunc - minunc) * norm_diff + maxunc
