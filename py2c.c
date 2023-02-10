/* 
 * C module which takes input data from a Fortran script and
 * and calls the Python Neural network for CPH/COT, CTP and MLAY. Results
 * are stored in a 3d array in linear representation.
 *
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdlib.h>
#include <arrayobject.h>
#include <stdbool.h>
#include <py2c.h>


void py_ann_mlay(void *vis006, void *vis008, void *nir016, void *ir039,
                    void *ir062, void *ir073, void *ir087, void *ir108,
                    void *ir120, void *ir134, void *lsm, void *skt,
                    void *solzen, void *satzen, int *nx, int *ny,
                    float *reg_mlay, char *bin_mlay, float *unc_mlay,
                    void *cldmask, char *msg_index,
                    bool *undo_true_reflectances)
{

    // initialize Python interpreter
    if (!Py_IsInitialized()){
        Py_Initialize();
    }

    init_numpy();

    // declare Python Objects
    npy_intp dims[2];
    dims[0] = *nx;
    dims[1] = *ny;

    PyObject *undo_true_refl_py;
    if (*undo_true_reflectances){
        undo_true_refl_py = Py_True;
    } else {
        undo_true_refl_py = Py_False;
    }

    PyObject *msg_index_py = Py_BuildValue("b", *msg_index);
    PyObject *mName, *pModule, *pFunc, *args_var;
    PyObject *vis006py, *vis008py, *nir016py, *ir039py, *ir062py, *ir073py;
    PyObject *ir087py, *ir108py, *ir120py, *ir134py, *lsmpy, *sktpy, *solzenpy;
    PyObject *satzenpy, *cldmaskpy;
    PyObject *res;

    // define and import Python module
    pModule = PyImport_ImportModule("prediction_funcs");

    if (pModule != NULL){
        // define function name
        pFunc = PyObject_GetAttrString(pModule, (char*)"predict_mlay");

        if (PyCallable_Check(pFunc)){
            // create numpy arrays from C array
            vis006py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, vis006);
            vis008py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, vis008);
            nir016py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, nir016);
            ir039py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir039);
            ir062py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir062);
            ir073py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir073);
            ir087py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir087);
            ir108py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir108);
            ir120py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir120);
            ir134py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir134);
            lsmpy = PyArray_SimpleNewFromData(2, dims, NPY_BYTE, lsm);
            sktpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, skt);
            solzenpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, solzen);
            satzenpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, satzen);
            cldmaskpy = PyArray_SimpleNewFromData(2, dims, NPY_BYTE, cldmask);

            PyArray_ENABLEFLAGS((PyArrayObject*) vis006py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) vis008py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) nir016py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir039py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir062py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir073py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir087py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir108py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir120py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir134py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) lsmpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) sktpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) solzenpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) satzenpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) cldmaskpy, NPY_ARRAY_OWNDATA);

            // generate args tuple for function call
            args_var = PyTuple_Pack(17, vis006py, vis008py, nir016py, ir039py,
                                    ir062py, ir073py, ir087py, ir108py, ir120py,
                                    ir134py, lsmpy, sktpy, solzenpy, satzenpy,
                                    undo_true_refl_py, msg_index_py, cldmaskpy);

            // call python function for COT
            res = PyObject_CallObject(pFunc, args_var);
            /* Function call returns list in  the form of:
             *  [COT_regression, COT_binary, COT_uncertainty,
             *   CPH_regression, CPH_binary, CPH_uncertainty] */

            int idx;
            // assign numpy result arrays to pre-allocated Fortran arrays
            if (res != NULL){
                for (int i=0; i < *nx; i++){
                    for (int j=0; j < *ny; j++){

                        idx = i * *ny + j;

                        reg_mlay[idx] = *(float *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 0),
                                                                  (npy_int)i, (npy_int)j);

                        bin_mlay[idx] = *(char *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 1),
                                                                 (npy_int)i, (npy_int)j);

                        unc_mlay[idx] = *(float *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 2),
                                                                  (npy_int)i, (npy_int)j);
                    }
			    }
                // decrement reference counter of this objects
                Py_DECREF(res);
                //Py_DECREF(tmp_var);
                Py_DECREF(args_var);

            } else {
                // decrement reference counter of this objects
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                Py_DECREF(vis006py);
                Py_DECREF(vis008py);
                Py_DECREF(nir016py);
                Py_DECREF(ir039py);
                Py_DECREF(ir062py);
                Py_DECREF(ir073py);
                Py_DECREF(ir087py);
                Py_DECREF(ir108py);
                Py_DECREF(ir120py);
                Py_DECREF(ir134py);
                Py_DECREF(lsmpy);
                Py_DECREF(sktpy);
                Py_DECREF(solzenpy);
                Py_DECREF(satzenpy);
                Py_DECREF(cldmaskpy);
                PyErr_Print();

                fprintf(stderr, "Call failed\n");
            }
        } else{
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function\n");
        }
        // decrement reference counter of this objects
        Py_DECREF(pModule);
        Py_DECREF(pFunc);
    } else {
        PyErr_Print();
        fprintf(stderr, "Failed to load module\n");
    }
    // finalize Python interpreter
    // if finalizing interpreter the ctp_ann cannot be called
    // subsequently (segfault when importing predictCTP because of numpy being
    // imported twice), thus Py_FinalizeEx() is commented out (not ideal but solves problem)
    // known bug in Python community.
    //Py_FinalizeEx();
}


void py_ann_ctp(void *vis006, void *vis008, void *nir016, void *ir039,
                void *ir062, void *ir073, void *ir087, void *ir108,
                void *ir120, void *ir134, void *lsm, void *skt,
                void *solzen, void *satzen, int *nx, int *ny,
                float *reg_ctp, float *unc_ctp, void *cldmask, char *msg_index,
                bool *undo_true_reflectances)
{
    //initialize Python interpreter
    if (!Py_IsInitialized()){
        Py_Initialize();
    }
 
    init_numpy();
    
    // declare Python Objects    
    npy_intp dims[2];
    dims[0] = *nx;
    dims[1] = *ny;

    PyObject *undo_true_refl_py;
    if (*undo_true_reflectances){
        undo_true_refl_py = Py_True;
    } else {
        undo_true_refl_py = Py_False;
    }

    PyObject *msg_index_py = Py_BuildValue("b", *msg_index);
    PyObject *mName, *pModule, *pFunc, *args_var;
    PyObject *vis006py, *vis008py, *nir016py, *ir039py, *ir062py, *ir073py;
    PyObject *ir087py, *ir108py, *ir120py, *ir134py, *lsmpy, *sktpy;
    PyObject *solzenpy, *satzenpy, *cldmaskpy;
    PyObject *res;
   
    pModule = PyImport_ImportModule("prediction_funcs");

    if (pModule != NULL){
        // define function name
        pFunc = PyObject_GetAttrString(pModule, (char*)"predict_ctp");

        if (PyCallable_Check(pFunc)){
            // create numpy arrays from C array
            vis006py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, vis006);
            vis008py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, vis008);
            nir016py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, nir016);
            ir039py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir039);
            ir062py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir062);
            ir073py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir073);
            ir087py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir087);
            ir108py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir108);
            ir120py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir120);
            ir134py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir134);
            lsmpy = PyArray_SimpleNewFromData(2, dims, NPY_BYTE, lsm);
            sktpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, skt);
            solzenpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, solzen);
            satzenpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, satzen);
            cldmaskpy = PyArray_SimpleNewFromData(2, dims, NPY_BYTE, cldmask);

            PyArray_ENABLEFLAGS((PyArrayObject*) vis006py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) vis008py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) nir016py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir039py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir062py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir073py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir087py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir108py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir120py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir134py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) lsmpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) sktpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) solzenpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) satzenpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) cldmaskpy, NPY_ARRAY_OWNDATA);

            // generate args tuple for function call
            args_var = PyTuple_Pack(17, vis006py, vis008py, nir016py, ir039py,
                                    ir062py, ir073py, ir087py, ir108py, ir120py,
                                    ir134py, lsmpy, sktpy, solzenpy, satzenpy,
                                    undo_true_refl_py, msg_index_py, cldmaskpy);

            // call python function for COT
            res = PyObject_CallObject(pFunc, args_var);
            /* Function call returns list in  the form of:
             *  [CTP_regression, CTP_uncertainty] */
                 
            int idx;
            // assign numpy result arrays to pre-allocated Fortran arrays
            if (res != NULL){
                for (int i=0; i < *nx; i++){
                    for (int j=0; j < *ny; j++){

                        idx = i * *ny + j;

                        reg_ctp[idx] = *(float *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 0),
                                                                  (npy_int)i, (npy_int)j);

                        unc_ctp[idx] = *(float *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 1),
                                                                  (npy_int)i, (npy_int)j);
                     }
	        }

                // decrement reference counter of this objects
                Py_DECREF(res);
                Py_DECREF(args_var);

            } else {
                // decrement reference counter of this objects
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                Py_DECREF(vis006py);
                Py_DECREF(vis008py);
                Py_DECREF(nir016py);
                Py_DECREF(ir039py);
                Py_DECREF(ir062py);
                Py_DECREF(ir073py);
                Py_DECREF(ir087py);
                Py_DECREF(ir108py);
                Py_DECREF(ir120py);
                Py_DECREF(ir134py);
                Py_DECREF(lsmpy);
                Py_DECREF(sktpy);
                Py_DECREF(solzenpy);
                Py_DECREF(satzenpy);
                Py_DECREF(cldmaskpy);
                PyErr_Print();

                fprintf(stderr, "Call failed\n");
            }
        } else{
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function\n");
        }
        // decrement reference counter of this objects
        Py_DECREF(pModule);
        Py_DECREF(pFunc);
    } else {
        PyErr_Print();
    }
    // finalize Python interpreter
    // if finalizing interpreter the ctp_ann cannot be called
    // subsequently (segfault when importing predictCTP because of numpy being
    // imported twice), thus Py_FinalizeEx() is commented out (not ideal but solves problem)
    // known bug in Python community.
    //Py_FinalizeEx();
}

void py_ann_ctt(void *vis006, void *vis008, void *nir016, void *ir039,
                void *ir062, void *ir073, void *ir087, void *ir108,
                void *ir120, void *ir134, void *lsm, void *skt,
                void *solzen, void *satzen, int *nx, int *ny,
                float *reg_ctt, float *unc_ctt, void *cldmask, char *msg_index,
                bool *undo_true_reflectances)
{
    //initialize Python interpreter
    if (!Py_IsInitialized()){
        Py_Initialize();
    }

    init_numpy();

    // declare Python Objects
    npy_intp dims[2];
    dims[0] = *nx;
    dims[1] = *ny;

    PyObject *undo_true_refl_py;
    if (*undo_true_reflectances){
        undo_true_refl_py = Py_True;
    } else {
        undo_true_refl_py = Py_False;
    }

    PyObject *msg_index_py = Py_BuildValue("b", *msg_index);
    PyObject *mName, *pModule, *pFunc, *args_var;
    PyObject *vis006py, *vis008py, *nir016py, *ir039py, *ir062py, *ir073py;
    PyObject *ir087py, *ir108py, *ir120py, *ir134py, *lsmpy, *sktpy;
    PyObject *solzenpy, *satzenpy, *cldmaskpy;
    PyObject *res;

    pModule = PyImport_ImportModule("prediction_funcs");

    if (pModule != NULL){
        // define function name
        pFunc = PyObject_GetAttrString(pModule, (char*)"predict_ctt");

        if (PyCallable_Check(pFunc)){
            // create numpy arrays from C array
            vis006py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, vis006);
            vis008py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, vis008);
            nir016py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, nir016);
            ir039py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir039);
            ir062py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir062);
            ir073py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir073);
            ir087py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir087);
            ir108py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir108);
            ir120py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir120);
            ir134py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir134);
            lsmpy = PyArray_SimpleNewFromData(2, dims, NPY_BYTE, lsm);
            sktpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, skt);
            solzenpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, solzen);
            satzenpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, satzen);
            cldmaskpy = PyArray_SimpleNewFromData(2, dims, NPY_BYTE, cldmask);

            PyArray_ENABLEFLAGS((PyArrayObject*) vis006py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) vis008py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) nir016py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir039py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir062py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir073py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir087py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir108py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir120py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir134py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) lsmpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) sktpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) solzenpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) satzenpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) cldmaskpy, NPY_ARRAY_OWNDATA);

            // generate args tuple for function call
            args_var = PyTuple_Pack(17, vis006py, vis008py, nir016py, ir039py,
                                    ir062py, ir073py, ir087py, ir108py, ir120py,
                                    ir134py, lsmpy, sktpy, solzenpy, satzenpy,
                                    undo_true_refl_py, msg_index_py, cldmaskpy);

            // call python function for COT
            res = PyObject_CallObject(pFunc, args_var);
            /* Function call returns list in  the form of:
             *  [CTP_regression, CTP_uncertainty] */

            int idx;
            // assign numpy result arrays to pre-allocated Fortran arrays
            if (res != NULL){
                for (int i=0; i < *nx; i++){
                    for (int j=0; j < *ny; j++){

                        idx = i * *ny + j;

                        reg_ctt[idx] = *(float *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 0),
                                                                  (npy_int)i, (npy_int)j);

                        unc_ctt[idx] = *(float *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 1),
                                                                  (npy_int)i, (npy_int)j);
                     }
	        }

                // decrement reference counter of this objects
                Py_DECREF(res);
                Py_DECREF(args_var);

            } else {
                // decrement reference counter of this objects
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                Py_DECREF(vis006py);
                Py_DECREF(vis008py);
                Py_DECREF(nir016py);
                Py_DECREF(ir039py);
                Py_DECREF(ir062py);
                Py_DECREF(ir073py);
                Py_DECREF(ir087py);
                Py_DECREF(ir108py);
                Py_DECREF(ir120py);
                Py_DECREF(ir134py);
                Py_DECREF(lsmpy);
                Py_DECREF(sktpy);
                Py_DECREF(solzenpy);
                Py_DECREF(satzenpy);
                Py_DECREF(cldmaskpy);
                PyErr_Print();

                fprintf(stderr, "Call failed\n");
            }
        } else{
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function\n");
        }
        // decrement reference counter of this objects
        Py_DECREF(pModule);
        Py_DECREF(pFunc);
    } else {
        PyErr_Print();
    }
    // finalize Python interpreter
    // if finalizing interpreter the ctp_ann cannot be called
    // subsequently (segfault when importing predictCTP because of numpy being
    // imported twice), thus Py_FinalizeEx() is commented out (not ideal but solves problem)
    // known bug in Python community.
    //Py_FinalizeEx();
}


void py_ann_cbh(void *ir108, void *ir120, void *ir134, void *solzen, void *satzen, 
		int *nx, int *ny, float *reg_cbh, float *unc_cbh, void *cldmask)
{
    //initialize Python interpreter
    if (!Py_IsInitialized()){
        Py_Initialize();
    }

    init_numpy();

    // declare Python Objects
    npy_intp dims[2];
    dims[0] = *nx;
    dims[1] = *ny;

    PyObject *mName, *pModule, *pFunc, *args_var;
    PyObject *ir108py, *ir120py, *ir134py;
    PyObject *solzenpy, *satzenpy, *cldmaskpy;
    PyObject *res;

   pModule = PyImport_ImportModule("prediction_funcs");

    if (pModule != NULL){
        // define function name
        pFunc = PyObject_GetAttrString(pModule, (char*)"predict_cbh");

        if (PyCallable_Check(pFunc)){
            // create numpy arrays from C array
            ir108py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir108);
            ir120py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir120);
            ir134py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir134);
            solzenpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, solzen);
            satzenpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, satzen);
            cldmaskpy = PyArray_SimpleNewFromData(2, dims, NPY_BYTE, cldmask);

            PyArray_ENABLEFLAGS((PyArrayObject*) ir108py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir120py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir134py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) solzenpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) satzenpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) cldmaskpy, NPY_ARRAY_OWNDATA);

          // generate args tuple for function call
            args_var = PyTuple_Pack(6, ir108py, ir120py, ir134py, solzenpy, 
			            satzenpy, cldmaskpy);

            // call python function for COT
            res = PyObject_CallObject(pFunc, args_var);
            /* Function call returns list in  the form of:
             *  [CTP_regression, CTP_uncertainty] */

            int idx;
            // assign numpy result arrays to pre-allocated Fortran arrays
            if (res != NULL){
                for (int i=0; i < *nx; i++){
                    for (int j=0; j < *ny; j++){

                        idx = i * *ny + j;

                        reg_cbh[idx] = *(float *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 0),
                                                                  (npy_int)i, (npy_int)j);

                        unc_cbh[idx] = *(float *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 1),
                                                                  (npy_int)i, (npy_int)j);
                     }
                }

                // decrement reference counter of this objects
                Py_DECREF(res);
                Py_DECREF(args_var);

            } else {
                // decrement reference counter of this objects
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                Py_DECREF(ir108py);
                Py_DECREF(ir120py);
                Py_DECREF(ir134py);
                Py_DECREF(solzenpy);
                Py_DECREF(satzenpy);
                Py_DECREF(cldmaskpy);
                PyErr_Print();

                fprintf(stderr, "Call failed\n");
            }
        } else{
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function\n");
        }
        // decrement reference counter of this objects
        Py_DECREF(pModule);
        Py_DECREF(pFunc);
    } else {
        PyErr_Print();
    }
    // finalize Python interpreter
    // if finalizing interpreter the ctp_ann cannot be called
    // subsequently (segfault when importing predictCTP because of numpy being
    // imported twice), thus Py_FinalizeEx() is commented out (not ideal but solves problem)
    // known bug in Python community.
    //Py_FinalizeEx();
}


void py_ann_cma(void *vis006, void *vis008, void *nir016, void *ir039,
                void *ir062, void *ir073, void *ir087, void *ir108,
                void *ir120, void *ir134, void *lsm, void *skt,
                void *solzen, void *satzen, int *nx, int *ny,
                float *reg_cot, char *bin_cot, float *unc_cot,
                char *msg_index, bool *undo_true_reflectances)
{
	
    // initialize Python interpreter
    if (!Py_IsInitialized()){
        Py_Initialize();
    }
    
    init_numpy();

    // declare Python Objects 
    npy_intp dims[2];
    dims[0] = *nx;
    dims[1] = *ny;

    PyObject *undo_true_refl_py; 
    if (*undo_true_reflectances){
        undo_true_refl_py = Py_True;
    } else {
        undo_true_refl_py = Py_False;
    }
    
    PyObject *msg_index_py = Py_BuildValue("b", *msg_index); 
    PyObject *mName, *pModule, *pFunc, *args_var;
    PyObject *vis006py, *vis008py, *nir016py, *ir039py, *ir062py, *ir073py; 
    PyObject *ir087py, *ir108py, *ir120py, *ir134py, *lsmpy, *sktpy, *solzenpy;
    PyObject *satzenpy;
    PyObject *res;
	       
    // define and import Python module 
    pModule = PyImport_ImportModule("prediction_funcs");

    if (pModule != NULL){
        // define function name
        pFunc = PyObject_GetAttrString(pModule, (char*)"predict_cma");
		
        if (PyCallable_Check(pFunc)){
            // create numpy arrays from C array			
            vis006py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, vis006);
            vis008py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, vis008);
            nir016py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, nir016);
            ir039py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir039);
            ir062py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir062);
            ir073py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir073);
            ir087py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir087);
            ir108py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir108);
            ir120py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir120);
            ir134py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir134);
            lsmpy = PyArray_SimpleNewFromData(2, dims, NPY_BYTE, lsm);
            sktpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, skt);
            solzenpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, solzen);
            satzenpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, satzen);

            PyArray_ENABLEFLAGS((PyArrayObject*) vis006py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) vis008py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) nir016py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir039py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir062py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir073py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir087py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir108py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir120py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir134py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) lsmpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) sktpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) solzenpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) satzenpy, NPY_ARRAY_OWNDATA);

            // generate args tuple for function call
            args_var = PyTuple_Pack(16, vis006py, vis008py, nir016py, ir039py, 
                                    ir062py, ir073py, ir087py, ir108py, ir120py,
                                    ir134py, lsmpy, sktpy, solzenpy, satzenpy,
                                    undo_true_refl_py, msg_index_py);
               
            // call python function for COT              
            res = PyObject_CallObject(pFunc, args_var);
            /* Function call returns list in  the form of:
             *  [COT_regression, COT_binary, COT_uncertainty,
             *   CPH_regression, CPH_binary, CPH_uncertainty] */

            int idx;
            // assign numpy result arrays to pre-allocated Fortran arrays     
            if (res != NULL){
                for (int i=0; i < *nx; i++){
                    for (int j=0; j < *ny; j++){
                            
                        idx = i * *ny + j;

                        reg_cot[idx] = *(float *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 0),
                                                                  (npy_int)i, (npy_int)j);

                        bin_cot[idx] = *(char *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 1),
                                                                 (npy_int)i, (npy_int)j);

                        unc_cot[idx] = *(float *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 2), 
                                                                  (npy_int)i, (npy_int)j);
                    }
			    }
                // decrement reference counter of this objects
                Py_DECREF(res);
                //Py_DECREF(tmp_var);
                Py_DECREF(args_var);
                    
            } else {
                // decrement reference counter of this objects
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                Py_DECREF(vis006py);
                Py_DECREF(vis008py);
                Py_DECREF(nir016py);
                Py_DECREF(ir039py);
                Py_DECREF(ir062py);
                Py_DECREF(ir073py);
                Py_DECREF(ir087py);
                Py_DECREF(ir108py);
                Py_DECREF(ir120py);
                Py_DECREF(ir134py);
                Py_DECREF(lsmpy);
                Py_DECREF(sktpy);
                Py_DECREF(solzenpy);
                Py_DECREF(satzenpy);
                PyErr_Print();
                
                fprintf(stderr, "Call failed\n");
            }
        } else{
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function\n");
        }
        // decrement reference counter of this objects
        Py_DECREF(pModule);
        Py_DECREF(pFunc);
    } else {
        PyErr_Print();
        fprintf(stderr, "Failed to load module\n");
    }
    // finalize Python interpreter
    // if finalizing interpreter the ctp_ann cannot be called
    // subsequently (segfault when importing predictCTP because of numpy being 
    // imported twice), thus Py_FinalizeEx() is commented out (not ideal but solves problem)
    // known bug in Python community.
    //Py_FinalizeEx(); 
}


void py_ann_cph(void *vis006, void *vis008, void *nir016, void *ir039,
                void *ir062, void *ir073, void *ir087, void *ir108,
                void *ir120, void *ir134, void *lsm, void *skt,
                void *solzen, void *satzen, int *nx, int *ny,
                float *reg_cph, char *bin_cph, float *unc_cph, void *cldmask,
                char *msg_index, bool *undo_true_reflectances)
{

    // initialize Python interpreter
    if (!Py_IsInitialized()){
        Py_Initialize();
    }

    init_numpy();

    // declare Python Objects
    npy_intp dims[2];
    dims[0] = *nx;
    dims[1] = *ny;

    PyObject *undo_true_refl_py;
    if (*undo_true_reflectances){
        undo_true_refl_py = Py_True;
    } else {
        undo_true_refl_py = Py_False;
    }

    PyObject *msg_index_py = Py_BuildValue("b", *msg_index);
    PyObject *mName, *pModule, *pFunc, *args_var;
    PyObject *vis006py, *vis008py, *nir016py, *ir039py, *ir062py, *ir073py;
    PyObject *ir087py, *ir108py, *ir120py, *ir134py, *lsmpy, *sktpy, *solzenpy;
    PyObject *satzenpy, *cldmaskpy;
    PyObject *res;

    // define and import Python module
    pModule = PyImport_ImportModule("prediction_funcs");

    if (pModule != NULL){
        // define function name
        pFunc = PyObject_GetAttrString(pModule, (char*)"predict_cph");

        if (PyCallable_Check(pFunc)){
            // create numpy arrays from C array
            vis006py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, vis006);
            vis008py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, vis008);
            nir016py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, nir016);
            ir039py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir039);
            ir062py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir062);
            ir073py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir073);
            ir087py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir087);
            ir108py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir108);
            ir120py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir120);
            ir134py = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ir134);
            lsmpy = PyArray_SimpleNewFromData(2, dims, NPY_BYTE, lsm);
            sktpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, skt);
            solzenpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, solzen);
            satzenpy = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, satzen);
            cldmaskpy = PyArray_SimpleNewFromData(2, dims, NPY_BYTE, cldmask);

            PyArray_ENABLEFLAGS((PyArrayObject*) vis006py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) vis008py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) nir016py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir039py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir062py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir073py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir087py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir108py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir120py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) ir134py, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) lsmpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) sktpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) solzenpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) satzenpy, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject*) cldmaskpy, NPY_ARRAY_OWNDATA);

            // generate args tuple for function call
            args_var = PyTuple_Pack(17, vis006py, vis008py, nir016py, ir039py,
                                    ir062py, ir073py, ir087py, ir108py, ir120py,
                                    ir134py, lsmpy, sktpy, solzenpy, satzenpy,
                                    undo_true_refl_py, msg_index_py, cldmaskpy);

            // call python function for COT
            res = PyObject_CallObject(pFunc, args_var);
            /* Function call returns list in  the form of:
             *  [COT_regression, COT_binary, COT_uncertainty,
             *   CPH_regression, CPH_binary, CPH_uncertainty] */

            int idx;
            // assign numpy result arrays to pre-allocated Fortran arrays
            if (res != NULL){
                for (int i=0; i < *nx; i++){
                    for (int j=0; j < *ny; j++){

                        idx = i * *ny + j;

                        reg_cph[idx] = *(float *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 0),
                                                                  (npy_int)i, (npy_int)j);

                        bin_cph[idx] = *(char *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 1),
                                                                 (npy_int)i, (npy_int)j);

                        unc_cph[idx] = *(float *) PyArray_GETPTR2((PyArrayObject *)PyList_GetItem(res, 2),
                                                                  (npy_int)i, (npy_int)j);
                    }
			    }
                // decrement reference counter of this objects
                Py_DECREF(res);
                //Py_DECREF(tmp_var);
                Py_DECREF(args_var);

            } else {
                // decrement reference counter of this objects
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                Py_DECREF(vis006py);
                Py_DECREF(vis008py);
                Py_DECREF(nir016py);
                Py_DECREF(ir039py);
                Py_DECREF(ir062py);
                Py_DECREF(ir073py);
                Py_DECREF(ir087py);
                Py_DECREF(ir108py);
                Py_DECREF(ir120py);
                Py_DECREF(ir134py);
                Py_DECREF(lsmpy);
                Py_DECREF(sktpy);
                Py_DECREF(solzenpy);
                Py_DECREF(satzenpy);
                Py_DECREF(cldmaskpy);
                PyErr_Print();

                fprintf(stderr, "Call failed\n");
            }
        } else{
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function\n");
        }
        // decrement reference counter of this objects
        Py_DECREF(pModule);
        Py_DECREF(pFunc);
    } else {
        PyErr_Print();
        fprintf(stderr, "Failed to load module\n");
    }
    // finalize Python interpreter
    // if finalizing interpreter the ctp_ann cannot be called
    // subsequently (segfault when importing predictCTP because of numpy being
    // imported twice), thus Py_FinalizeEx() is commented out (not ideal but solves problem)
    // known bug in Python community.
    //Py_FinalizeEx();
}
