#import predictCPHCOT as pCPHCOT
import prediction_funcs as preds
#import predictMLAY as pMLAY
import numpy as np

dims = (100, 100)
arr1 = np.ones(dims)
arr2 = np.ones(dims)
arr3 = np.ones(dims)
arr4 = np.ones(dims)
arr5 = np.ones(dims)
arr6 = np.ones(dims)
arr7 = np.ones(dims)
arr8 = np.ones(dims)
arr9 = np.ones(dims)
arr10 = np.ones(dims)
arr11 = np.ones(dims)
arr12 = np.ones(dims)
solzen = np.ones(dims)
satzen = np.ones(dims)

# artificial cloudmask
cldmask = np.ones(dims)
cldmask[18:, ::3] = 0

#res = pCPHCOT.predict_CPH_COT(arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8,
#        arr9, arr10, arr11, arr12, solzen, satzen, False, 0)
#print(len(res), type(res))

# ---- WITHOUT EXTERNAL CMA ----
res = preds.predict_ctp(arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8,
        arr9, arr10, arr11, arr12, solzen=solzen, satzen=satzen, undo_true_refl=False, 
        correct_vis_cal_nasa_to_impf=0)
print(len(res), type(res))

res = preds.predict_ctp(arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8,
        arr9, arr10, arr11, arr12, solzen=solzen, satzen=satzen, undo_true_refl=False, 
        correct_vis_cal_nasa_to_impf=0, cldmask=cldmask)
print(len(res), type(res))

res = preds.predict_mlay(arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8,
        arr9, arr10, arr11, arr12, solzen, satzen, False, 0)
print(len(res), type(res))

res = preds.predict_mlay(arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8,
        arr9, arr10, arr11, arr12, solzen, satzen, False, 0, cldmask)
print(len(res), type(res))

res = preds.predict_cph(arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8,
        arr9, arr10, arr11, arr12, solzen=solzen, satzen=satzen, undo_true_refl=False,
        correct_vis_cal_nasa_to_impf=0)
print(len(res), type(res))

res = preds.predict_cph(arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8,
        arr9, arr10, arr11, arr12, solzen=solzen, satzen=satzen, undo_true_refl=False,
        correct_vis_cal_nasa_to_impf=0, cldmask=cldmask)
print(len(res), type(res))

res = preds.predict_cma(arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8,
        arr9, arr10, arr11, arr12, solzen=solzen, satzen=satzen, undo_true_refl=False,
        correct_vis_cal_nasa_to_impf=0)
print(len(res), type(res))
# ---- WITH EXTERNAL CMA ----
#res = pCTP.predict_CTP(arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8,
#        arr9, arr10, arr11, arr12, solzen, satzen, False, 0, cldmask)
#print(len(res), type(res))

#res = pMLAY.predict_MLAY(arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8,
#        arr9, arr10, arr11, arr12, solzen, satzen, False, 0, cldmask)
#print(len(res), type(res))



