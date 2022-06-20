import prediction_funcs as preds
import numpy as np
import xarray as xr
import sys

import logging


IFILE = 'seviri_ml_test_201907011200.nc'
TESTPLOT = 'testfigure_seviri_ml.png'

ds = xr.open_dataset(IFILE)

# example data
vis006 = ds['VIS006'].data
vis008 = ds['VIS008'].data
ir016 = ds['IR_016'].data
ir039 = ds['IR_039'].data
ir062 = ds['WV_062'].data
ir073 = ds['WV_073'].data
ir082 = ds['IR_087'].data
ir108 = ds['IR_108'].data
ir120 = ds['IR_120'].data
ir134 = ds['IR_134'].data
lsm = ds['lsm'].data
skt = ds['skt'].data
solzen = ds['solzen'].data
satzen = ds['satzen'].data
print('---------------- CHECK CMA -----------------')
cma = preds.predict_cma(vis006, vis008, ir016, ir039, ir062, ir073, 
                        ir082, ir108, ir120, ir134, lsm, skt, 
                        solzen=solzen, satzen=satzen, 
                        undo_true_refl=False, correct_vis_cal_nasa_to_impf=0)

cldmask = cma[1]
print('PYTHON3 CMA mean: ', np.mean(cma[1][cma[1] >= 0]))

print('---------------- CHECK CPH WITH CLDMASK -----------------')
cph = preds.predict_cph(vis006, vis008, ir016, ir039, ir062, ir073,
                        ir082, ir108, ir120, ir134, lsm, skt,
                        solzen=solzen, satzen=satzen,
                        undo_true_refl=False, correct_vis_cal_nasa_to_impf=0,
                        cldmask=cldmask)

print('---------------- CHECK CTP WITH CLDMASK -----------------')
ctp = preds.predict_ctp(vis006, vis008, ir016, ir039, ir062, ir073,
                        ir082, ir108, ir120, ir134, lsm, skt,
                        solzen=solzen, satzen=satzen,
                        undo_true_refl=False, correct_vis_cal_nasa_to_impf=0,
                        cldmask=cldmask)

print('---------------- CHECK MLAY WITH CLDMASK -----------------')
mlay = preds.predict_mlay(vis006, vis008, ir016, ir039, ir062, ir073,
                        ir082, ir108, ir120, ir134, lsm, skt,
                        solzen=solzen, satzen=satzen,
                        undo_true_refl=False, correct_vis_cal_nasa_to_impf=0, 
                        cldmask=cldmask)

logging.disable(logging.CRITICAL)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

cot_prediction = np.where(cma[0] < 0, np.nan, cma[0])
cldmask = np.where(cma[1] < 0, np.nan, cma[1])
cma_uncertainty = np.where(cma[2] < 0, np.nan, cma[2])

cph_prediction = np.where(cph[0] < 0, np.nan, cph[0])
phase = np.where(cph[1] < 0, np.nan, cph[1])
cph_uncertainty = np.where(cph[2] <= 0, np.nan, cph[2])

pressure = np.where(ctp[0] < 0, np.nan, ctp[0])
ctp_uncertainty = np.where(ctp[1] <= 0, np.nan, ctp[1])

mlay_prediction = np.where(mlay[0] < 0, np.nan, mlay[0])
mlay_flag = np.where(mlay[1] < 0, np.nan, mlay[1])
mlay_uncertainty = np.where(mlay[2] <0, np.nan, mlay[2])

# -------------- PLOTTING ---------------
IPROJ = ccrs.Geostationary()
OPROJ = ccrs.Geostationary()
fig = plt.figure(figsize=(13,10))

ax = fig.add_subplot(431, projection=OPROJ)
ims = ax.imshow(cot_prediction, transform=IPROJ)
ax.set_title('COT prediction')
plt.colorbar(ims)

ax = fig.add_subplot(432, projection=OPROJ)
ims = ax.imshow(cldmask, transform=IPROJ, interpolation='none')
ax.set_title('Binary CMA')
plt.colorbar(ims)

ax = fig.add_subplot(433, projection=OPROJ)
ims = ax.imshow(cma_uncertainty, transform=IPROJ)
ax.set_title('CMA uncertainty')
plt.colorbar(ims)

ax = fig.add_subplot(434, projection=OPROJ)
ims = ax.imshow(cph_prediction, transform=IPROJ, interpolation='none')
ax.set_title('CPH prediction')
plt.colorbar(ims)

ax = fig.add_subplot(435, projection=OPROJ)
ims = ax.imshow(phase, transform=IPROJ, interpolation='none')
ax.set_title('Binary CPH')
plt.colorbar(ims)

ax = fig.add_subplot(436, projection=OPROJ)
ims = ax.imshow(cph_uncertainty, transform=IPROJ, interpolation='none')
ax.set_title('CPH uncertainty')
plt.colorbar(ims)

ax = fig.add_subplot(437, projection=OPROJ)
ims = ax.imshow(pressure, transform=IPROJ, interpolation='none')
ax.set_title('CTP')
plt.colorbar(ims)

ax = fig.add_subplot(438, projection=OPROJ)
ims = ax.imshow(ctp_uncertainty, transform=IPROJ, interpolation='none')
ax.set_title('CTP uncertainty')
plt.colorbar(ims)

ax = fig.add_subplot(4,3,10, projection=OPROJ)
ims = ax.imshow(mlay_prediction, transform=IPROJ, interpolation='none')
ax.set_title('MLAY prediction')
plt.colorbar(ims)

ax = fig.add_subplot(4,3,11, projection=OPROJ)
ims = ax.imshow(mlay_flag, transform=IPROJ, interpolation='none')
ax.set_title('MLAY flag')
plt.colorbar(ims)

ax = fig.add_subplot(4,3,12, projection=OPROJ)
ims = ax.imshow(mlay_uncertainty, transform=IPROJ, interpolation='none')
ax.set_title('MLAY uncertainty')
plt.colorbar(ims)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig(TESTPLOT)
print('SAVED: ', TESTPLOT)
