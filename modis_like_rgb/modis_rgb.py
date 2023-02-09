""" 
This module generates an SEVIRI MODIS-like true color RGB 
based on an Artificial Neural Networks (ANNs). SEVIRI 
measurements have been trained to mimic the MOD_143D_RR and
MYD_143D_RR 1-day true color MODIS composite RGB values. 
MOD_143D_RR and MYD_143D_RR have been generated from
MOD09CMG and MYD09CMG products.

Author: Daniel Philipp (DWD)

MODIS Daily composit True color RGBs:
  - Aqua: https://neo.gsfc.nasa.gov/archive/rgb/MYD_143D_RR
  - Terra: https://neo.gsfc.nasa.gov/archive/rgb/MOD_143D_RR

MOD09CMG v6.1: https://lpdaac.usgs.gov/products/mod09cmgv061/
MYD09CMG v6.1: https://lpdaac.usgs.gov/products/myd09cmgv061/

"""


import numpy as np
import matplotlib.pyplot as plt
import joblib
import PIL
import os
import dask
import seviri_ml.helperfuncs as hf


# set backend name from environment variable
backend = hf.get_backend_name(os.environ.get('SEVIRI_ML_BACKEND'))

if backend == 'THEANO':
    from keras.models import load_model
elif backend == 'TENSORFLOW2':
    from tensorflow.keras.models import load_model


class ModisLikeRGB:
    def __init__(self, vis006, vis008, ir_016, 
                 ir_108, ir_134, sza):
        self.vis006 = vis006
        self.vis008 = vis008
        self.ir_016 = ir_016
        self.ir_108 = ir_108
        self.ir_134 = ir_134
        self.sza = sza

        self.size = vis006.size
        self.shape = vis006.shape
        self.n_features = 5

        path = os.path.dirname(__file__)

        self.scaler_file = self.scaler_file = os.path.join(path,
                                  'SCALER_MODIS-LIKE_RGB_v1.pkl')
        if backend == 'THEANO':
            self.model_file = os.path.join(path, 
                                  'MODEL_MODIS-LIKE_RGB_THEANO__1.0.4__v1.h5')
        elif backend == 'TENSORFLOW2':
            self.model_file = os.path.join(path,
                                  'MODEL_MODIS-LIKE_RGB_TF2__2.4.1__v1.h5')            

        self.rgb = None

    def _correct_night_pixels(self, rgb):
        """ 
        Make night pixels look like a nice greyscale
        based on IR_108 BT with a color to greyscale
        fade-in through the twilight area. 
        """

        r = rgb[:,:,0]*255.
        g = rgb[:,:,1]*255.
        b = rgb[:,:,2]*255.

        ir_108 = self.ir_108
        sza = self.sza

        # greyscale conversion
        ir_minmax = (180., 310.)
        irb = np.clip(ir_108, ir_minmax[0], ir_minmax[1])
        irb = ir_minmax[1] - irb
        irb = (irb * 255.) / (ir_minmax[1] - ir_minmax[0])
        irb = np.clip(irb, 0, 255)
        night = sza >= 88
        r = np.where(night, irb, r)
        g = np.where(night, irb, g)
        b = np.where(night, irb, b)

        # fade-in through twilight
        weight = (sza-70)/18.
        fade_sza = (sza > 70) & (sza < 88)
        wirb = weight * irb
        r = np.where(fade_sza, wirb + (1-weight)*r, r)
        g = np.where(fade_sza, wirb + (1-weight)*g, g)
        b = np.where(fade_sza, wirb + (1-weight)*b, b)

        return (np.dstack((r,g,b))/255.).astype(np.float32)

    def make_rgb(self, fmt='float'):
        """ Apply the ANN to make the RGB and make greyscale night pixels. """

        # ANN input in correct format
        inputs = (self.vis006, self.vis008, self.ir_016, 
                  self.ir_108, self.ir_134)
        idata = np.squeeze(np.dstack(inputs))
        idata = idata.reshape((self.size, self.n_features))

        # scale ANN input
        scaler = joblib.load(self.scaler_file)
        idata = scaler.transform(idata)

        # load model and predict
        model = load_model(self.model_file, compile=False)
        rgb = np.squeeze(model.predict(idata, verbose=1))
        rgb = rgb.reshape(self.shape + (3,))
        rgb = np.clip(rgb, 0, 1)

        # make greyscale night pixels with twilight fade-in
        rgb = self._correct_night_pixels(rgb)

        # should RGB be [0,1] float32 or [0, 255] ubyte
        if fmt == 'int':
            rgb *= 255.
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        elif fmt == 'float':
            rgb = np.clip(rgb, 0, 1).astype(np.float32)
        else:
            raise Exception('fmt has to float (0-1 range) ' \
                            'or int(0-255 range).')
        
        self.rgb = rgb

        return rgb

    def save_to_png(self, filename='modis-like_ann_rgb.png', img_size=None):
        """ Save RGB to a full-frame PNG image."""

        if self.rgb is None:
            raise Exception('Please run ModisLikeRGB.make_rgb()'
                            'before saving.')
        else:
            if not filename.endswith('.png'):
                raise Exception('Please save image as .png')
            if isinstance(self.rgb, dask.array.core.Array):
                rgb = np.array(self.rgb)
            else:
                rgb = self.rgb
            
            if rgb.dtype == np.float32:
                rgb *= 255.
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
 
            img = PIL.Image.fromarray(rgb)

            if img_size is not None:
                if isinstance(img_size, tuple):
                    img = img.resize(img_size)
                else:
                    raise Exception('If providing img_size it has to be a tuple: ' \
                                    '(num_x_pixel, num_y_pixel)')

            img.save(filename)

    def show(self):
        """  Quickly show RGB in this instance (quicklook). """

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        ax.imshow(self.rgb)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.show()
        plt.close()

