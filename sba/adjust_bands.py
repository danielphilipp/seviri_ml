"""
This module contains functionalities to modify non-SEVIRI sensor bands
to match the Spectral Response Function (SRF) of SEVIRI spectral bands
onboard Meteosat-11 (MSG4). Supported sensors are AHI on HIMAWARI, ABI
on GOES16 and GOES17 as well as SEVIRI on Meteosat-8 (MSG1).

The Spectral Band Adjustment is a linear correction consisting of a
slope and offset. For information on how the slope and offset were
derived, see

https://climate.esa.int/media/documents/Cloud_Algorithm-Theoretical
-Baseline-Document-ATBD-CC4CL_v6.2.pdf

in section 2.3.

This functionality is helpful if you want to run the SEVIRI_ML
neural networks, which were trained with SEVIRI measurements,
with a non-SEVIRI sensor.

So e.g. (pseudo code):

-------------------------

from seviri_ml.sba import adjust_bands
from seviri_ml import predictCTP
from satpy import Scene

# read AHI with satpy
scn = Scene(..)
scn.load(['VIS006',...])

# bring AHI channels in dictionary structure for SBA
himawari_measurements = {'VIS006': scn['VIS006'].data, ...}

# apply SBA
sba = adjust_bands.SpectralBandAdjustment('HIMAWARI-AHI',
                                          himawari_measurements)
himawari_like_seviri = sba.transform()

# run CTP ANN
results = predictCTP.predict_CTP(himawari_like_seviri)

------------------------
"""


COEFFS_MET11_NIGHT = {
    'HIMAWARI-AHI': {
        'VIS006': {'slope': 1.00000, 'offset': 0.00000},
        'VIS008': {'slope': 1.00000, 'offset': 0.00000},
        'NIR016': {'slope': 1.00000, 'offset': 0.00000},
        'IR_039': {'slope': 0.93871, 'offset': 14.8503},
        'IR_062': {'slope': 1.00526, 'offset': -1.29354},
        'IR_073': {'slope': 1.00413, 'offset': -1.20664},
        'IR_087': {'slope': 1.01050, 'offset': -2.63734},
        'IR_097': {'slope': 1.01338, 'offset': -2.61762},
        'IR_108': {'slope': 0.99540, 'offset': 1.49858},
        'IR_120': {'slope': 1.02742, 'offset': -6.43452},
        'IR_134': {'slope': 0.89684, 'offset': 21.9239}
        },
    'GOES16-ABI': {
        'VIS006': {'slope': 1.00000, 'offset': 0.00000},
        'VIS008': {'slope': 1.00000, 'offset': 0.00000},
        'NIR016': {'slope': 1.00000, 'offset': 0.00000},
        'IR_039': {'slope': 0.94085, 'offset': 14.4615},
        'IR_062': {'slope': 0.99043, 'offset': 1.80946},
        'IR_073': {'slope': 1.00483, 'offset': -1.50406},
        'IR_087': {'slope': 1.01947, 'offset': -4.94939},
        'IR_097': {'slope': 1.01835, 'offset': -3.61878},
        'IR_108': {'slope': 0.99627, 'offset': 1.23287},
        'IR_120': {'slope': 1.02239, 'offset': -5.28908},
        'IR_134': {'slope': 0.89005, 'offset': 23.4361}
        },
    'GOES17-ABI': {
        'VIS006': {'slope': 1.00000, 'offset': 0.00000},
        'VIS008': {'slope': 1.00000, 'offset': 0.00000},
        'NIR016': {'slope': 1.00000, 'offset': 0.00000},
        'IR_039': {'slope': 0.93929, 'offset': 14.7561},
        'IR_062': {'slope': 0.99050, 'offset': 1.80263},
        'IR_073': {'slope': 1.00481, 'offset': -1.48189},
        'IR_087': {'slope': 1.01907, 'offset': -4.84116},
        'IR_097': {'slope': 1.01759, 'offset': -3.46801},
        'IR_108': {'slope': 0.99613, 'offset': 1.27653},
        'IR_120': {'slope': 1.02298, 'offset': -5.42165},
        'IR_134': {'slope': 0.91761, 'offset': 17.3323}
        },
    'METEOSAT8-SEVIRI': {
        'VIS006': {'slope': 1.00000, 'offset': 0.00000},
        'VIS008': {'slope': 1.00000, 'offset': 0.00000},
        'NIR016': {'slope': 1.00000, 'offset': 0.00000},
        'IR_039': {'slope': 1.00882, 'offset': -1.94343},
        'IR_062': {'slope': 1.00146, 'offset': -0.302762},
        'IR_073': {'slope': 1.00121, 'offset': -0.259928},
        'IR_087': {'slope': 1.00194, 'offset': -0.471424},
        'IR_097': {'slope': 0.99932, 'offset': 0.131567},
        'IR_108': {'slope': 0.99951, 'offset': 0.104622},
        'IR_120': {'slope': 0.99968, 'offset': 0.065808},
        'IR_134': {'slope': 0.95540, 'offset': 9.71430}
        }
}

AVAILABLE_CHANNELS = ['VIS006', 'VIS008', 'NIR016', 'IR_039', 'IR_062',
                      'IR_073', 'IR_087', 'IR_097', 'IR_108', 'IR_120',
                      'IR_134']

AVAILABLE_INPUT_SATS = ['HIMAWARI-AHI',
                        'GOES16-ABI',
                        'GOES17-ABI',
                        'METEOSAT8-SEVIRI']

AVAILABLE_TARGET_SATS = ['METEOSAT11-SEVIRI']


class SpectralBandAdjustment:
    def __init__(self, input_sat, input_sat_data,
                 target_sat='METEOSAT11-SEVIRI', verbose=False):
        """ input_sat (str):       Name of input satellite to be transformed.
                                   Select from AVAILABLE INPUT_SATS
            input_sat_data (dict): Dictionary of input satellite data for
                                   each channel. Channel name (from
                                   AVAILABLE_CHANNELS) should be mapped
                                   to corresponding numpy arrays.
                                   {'ch1': array1, 'ch2': array2, ...}
            target_sat (str):      Name of satellite to be mimiced. Until now
                                   only METEOSAT11-SEVIRI is available (MSG4).
        """

        self.input_sat = input_sat
        self.target_sat = target_sat
        self.transformed = input_sat_data
        self.verbose = verbose

        self._check_sats()

        self.is_transformed = False

    def _check_sats(self):
        """ Check if satellites for transformation are available """
        if self.input_sat not in AVAILABLE_INPUT_SATS:
            raise Exception('input_sat {} not available. '
                            'Choose from {}'.format(self.input_sat,
                                                    AVAILABLE_INPUT_SATS))

        if self.target_sat not in AVAILABLE_TARGET_SATS:
            raise Exception('target_sat {} not available. '
                            'Choose from {}'.format(self.target_sat,
                                                    AVAILABLE_TARGET_SATS))

    def transform(self):
        """ Linearily transform input_sat_data to mimic the target satellite's
        spectral band.
        """
        if self.verbose:
            print('   ----- TRANSFORMING DATA FROM {} TO MIMIC {} ---'
                  '--\n'.format(self.input_sat, self.target_sat))
        if self.is_transformed:
            print('   >>> WARNING: For transform, the data already have '
                  'been transformed.')

        for ch in AVAILABLE_CHANNELS:
            if ch in self.transformed.keys():
                slope = COEFFS_MET11_NIGHT[self.input_sat][ch]['slope']
                offset = COEFFS_MET11_NIGHT[self.input_sat][ch]['offset']
                self.transformed[ch] = self._apply_transform(
                                                        slope,
                                                        offset,
                                                        self.transformed[ch]
                                                        )
                if self.verbose:
                    print('   >>> Transformed {} with slope={:.4f} and '
                          'offset={:.4f}'.format(ch, slope, offset))
            else:
                if self.verbose:
                    print('   >>> Channel {} not found in input_sat_data '
                          '=> Skipping'.format(ch))

        self.is_transformed = True
        return self.transformed

    def _apply_transform(self, slope, offset, data):
        """ Apply transformation to a single channel. """
        return slope * data + offset

    def inverse_transform(self):
        """ Linearily re-transform data to mimic input satellite's
        spectral band.
        """
        if self.verbose:
            print('   ----- RE-TRANSFORMING DATA FROM {} TO {} -----'
                  '\n'.format(self.target_sat, self.input_sat))
        if not self.is_transformed:
            print('   >>> WARNING: For inverse_transform, the data have not '
                  'been transformed yet.')

        for ch in AVAILABLE_CHANNELS:
            if ch in self.transformed.keys():
                slope = COEFFS_MET11_NIGHT[self.input_sat][ch]['slope']
                offset = COEFFS_MET11_NIGHT[self.input_sat][ch]['offset']
                self.transformed[ch] = self._apply_inverse_transform(
                                                        slope,
                                                        offset,
                                                        self.transformed[ch])
                if self.verbose:
                    print('   >>> Inverse transformed {} with slope={:.4f} '
                          'and offset={:.4f}'.format(ch, slope, offset))
            else:
                if self.verbose:
                    print('   >>> Channel {} not found in input_sat_data '
                          '=> Skipping'.format(ch))

        self.is_transformed = False
        return self.transformed

    def _apply_inverse_transform(self, slope, offset, data):
        """ Apply inverse transformation to a single channel """
        return (data-offset)/slope
