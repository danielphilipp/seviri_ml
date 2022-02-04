import predictCPHCOT as pCPHCOT
import predictCTP as pCTP
import numpy as np

dims = (37, 37)
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
res = pCPHCOT.predict_CPH_COT(arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8,
        arr9, arr10, arr11, arr12, solzen, satzen, False, 0)

print(len(res), type(res))

res = pCTP.predict_CTP(arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8,
        arr9, arr10, arr11, arr12, solzen, satzen, False, 0)

print(len(res), type(res))

