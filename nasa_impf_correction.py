import numpy as np

PARAMETERS = {'MSG1': {'SLOPE': {'ch1': 0.85628,'ch2': 0.91591,'ch3': 1.05427},
                       'INTERCEPT': {'ch1': 0.0, 'ch2': 0.0, 'ch3': 0.0}},
             {'MSG2': {'SLOPE': {'ch1': 0.91689,'ch2': 0.93875,'ch3': 1.04209},
                       'INTERCEPT': {'ch1': 0.0, 'ch2': 0.0, 'ch3': 0.0}},
             {'MSG3': {'SLOPE': {'ch1': 0.89353,'ch2': 0.95154,'ch3': 1.03764},
                       'INTERCEPT': {'ch1': 0.0, 'ch2': 0.0, 'ch3': 0.0}},
             {'MSG4': {'SLOPE': {'ch1': 0.90091,'ch2': 0.97954,'ch3': 1.06372},
                       'INTERCEPT': {'ch1': 0.0, 'ch2': 0.0, 'ch3': 0.0}}
             }


def _apply_correction(data, slope, intercept):
     lin_corr = data - (slope * data + intercept)
     return np.where(data > 0, data - lin_corr, data)


def correct_nasa_impf(vis006, vis008, nir016, msg_index):
              
    msg = 'MSG{:d}'.format(msg_index)
    params = PARAMETERS[msg]

    vis006 = _apply_correction(vis006, 
                               params['SLOPE']['ch1'],
                               params['INTERCEPT']['ch1'])

    vis008 = _apply_correction(vis008,
                               params['SLOPE']['ch2'],
                               params['INTERCEPT']['ch2'])

    nir016 = _apply_correction(nir016,
                               params['SLOPE']['ch3'],
                               params['INTERCEPT']['ch3'])

    return vis006, vis008, nir016 
