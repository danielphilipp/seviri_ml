#include <stdbool.h> 
#ifndef PY2C_H
#define PY2C_H

#if PY_MAJOR_VERSION >=3
int init_numpy()
{
    if(PyArray_API == NULL)
    {
        import_array();
    }
}
#else
void init_numpy()
{
    if(PyArray_API == NULL)
    {
        import_array();
    }
}
#endif

void py_ann_mlay(void *vis006, void *vis008, void *nir016, void *ir039,
                    void *ir062, void *ir073, void *ir087, void *ir108,
                    void *ir120, void *ir134, void *lsm, void *skt,
                    void *solzen, void *satzen, int *nx, int *ny,
                    float *reg_mlay, char *bin_mlay, float *unc_mlay,
                    void *cldmask, char *msg_index,
                    bool *undo_true_reflectances);

void py_ann_ctp(void *vis006, void *vis008, void *nir016, void *ir039,
                void *ir062, void *ir073, void *ir087, void *ir108,
                void *ir120, void *ir134, void *lsm, void *skt,
                void *solzen, void *satzen, int *nx, int *ny,
                float *reg_ctp, float *unc_ctp, void *cldmask, char *msg_index,
                bool *undo_true_reflectances);

void py_ann_ctt(void *vis006, void *vis008, void *nir016, void *ir039,
                void *ir062, void *ir073, void *ir087, void *ir108,
                void *ir120, void *ir134, void *lsm, void *skt,
                void *solzen, void *satzen, int *nx, int *ny,
                float *reg_ctt, float *unc_ctt, void *cldmask, char *msg_index,
                bool *undo_true_reflectances);

void py_ann_cma(void *vis006, void *vis008, void *nir016, void *ir039,
                void *ir062, void *ir073, void *ir087, void *ir108,
                void *ir120, void *ir134, void *lsm, void *skt,
                void *solzen, void *satzen, int *nx, int *ny,
                float *reg_cot, char *bin_cot, float *unc_cot,
                char *msg_index, bool *undo_true_reflectances);

void py_ann_cbh(void *ir108, void *ir120, void *ir134, void *solzen, 
		void *satzen, int *nx, int *ny, float *reg_cbh, 
		float *unc_cbh, void *cldmask);


void py_ann_cph(void *vis006, void *vis008, void *nir016, void *ir039,
                void *ir062, void *ir073, void *ir087, void *ir108,
                void *ir120, void *ir134, void *lsm, void *skt,
                void *solzen, void *satzen, int *nx, int *ny,
                float *reg_cph, char *bin_cph, float *unc_cph, void *cldmask,
                char *msg_index, bool *undo_true_reflectances);

#endif
