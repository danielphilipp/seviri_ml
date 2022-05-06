import numpy as np
from adjust_bands import SpectralBandAdjustment

testsats = ['GOES16-ABI', 'GOES17-ABI', 'HIMAWARI-AHI', 'METEOSAT8-SEVIRI']
for sat in testsats:
    print('\n################## TESING {} -> METEOSAT11-SEVIRI ##################'.format(sat))

    # ---- make up test data -----
    # for VIS channels
    array_VIS = np.ones((10,10))
    array_VIS[::2, ::2] = 0.5

    # for IR channels
    array_IR = np.ones((10,10)) * 180.
    array_IR[::2, ::2] = 273.15

    # make input_sat_data dict
    input_sat_data = {'VIS006': array_VIS, 'VIS008': array_VIS, 
                     'IR_087': array_IR, 'IR_108': array_IR,
                     'IR_120': array_IR, 'IR_134': array_IR}
    # create instance of SpectralBandAdjustment for transformation
    sba = SpectralBandAdjustment(sat, input_sat_data, verbose=True)
    print('Instance created')
    
    # transform the input satellite data
    transformed = sba.transform()
    print('Data transformed')

    # re-transform adjusted data to original form
    original = sba.inverse_transform()
    print('Data inverse transformed back to original')

    for ch in input_sat_data.keys():
        all_same = np.all(original[ch]==input_sat_data[ch])
        if all_same:
            print('Channel {}: Transformed and re-transformed data are equal to input_sat data! OK'.format(ch))
        else:
            print('Channel {}: Transformed and re-transformed data are NOT equal to input_sat data! FAIL'.format(ch))


