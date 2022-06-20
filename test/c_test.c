#include <stdlib.h>
#include <stdio.h>
#include <netcdf.h>
#include <py2c.h>
#include <stdbool.h>

#define FILE_NAME "seviri_ml_test_201907011200.nc"
#define NX 100
#define NY 100

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

int main() {

    int ncid, vis006_id, vis008_id, ir_016_id, ir_039_id, \
        ir_062_id, ir_073_id, ir_087_id, ir_108_id, ir_120_id, \
        ir_134_id, lsm_id, skt_id, solzen_id, satzen_id;

    float vis006[NX][NY], vis008[NX][NY], ir_016[NX][NY], \
          ir_039[NX][NY], ir_062[NX][NY], ir_073[NX][NY], \
          ir_087[NX][NY], ir_108[NX][NY], ir_120[NX][NY], \
          ir_134[NX][NY], skt[NX][NY], satzen[NX][NY], \
          solzen[NX][NY], cot[NX][NY], cma_unc[NX][NY];

    char lsm[NX][NY], cma[NX][NY];
    char msg_num=0;
    int nx = NX, ny = NY;
    int retval;
    bool undo_true_refl=false;
    float meanval;

    if ((retval = nc_open(FILE_NAME, NC_NOWRITE, &ncid)))
        ERR(retval);

    if ((retval = nc_inq_varid(ncid, "VIS006", &vis006_id)))
        ERR(retval);
    if ((retval = nc_inq_varid(ncid, "VIS008", &vis008_id)))
        ERR(retval);
    if ((retval = nc_inq_varid(ncid, "IR_016", &ir_016_id)))
        ERR(retval);
    if ((retval = nc_inq_varid(ncid, "IR_039", &ir_039_id)))
        ERR(retval);
    if ((retval = nc_inq_varid(ncid, "WV_062", &ir_062_id)))
        ERR(retval);
    if ((retval = nc_inq_varid(ncid, "WV_073", &ir_073_id)))
        ERR(retval);
    if ((retval = nc_inq_varid(ncid, "IR_087", &ir_087_id)))
        ERR(retval);
    if ((retval = nc_inq_varid(ncid, "IR_108", &ir_108_id)))
        ERR(retval);
    if ((retval = nc_inq_varid(ncid, "IR_120", &ir_120_id)))
        ERR(retval);
    if ((retval = nc_inq_varid(ncid, "IR_134", &ir_134_id)))
        ERR(retval);
    if ((retval = nc_inq_varid(ncid, "lsm", &lsm_id)))
        ERR(retval);
    if ((retval = nc_inq_varid(ncid, "skt", &skt_id)))
        ERR(retval);
    if ((retval = nc_inq_varid(ncid, "solzen", &solzen_id)))
        ERR(retval);
    if ((retval = nc_inq_varid(ncid, "satzen", &satzen_id)))
        ERR(retval);

    if ((retval = nc_get_var_float(ncid, vis006_id, &vis006[0][0])))
        ERR(retval);
    if ((retval = nc_get_var_float(ncid, vis008_id, &vis008[0][0])))
        ERR(retval);
    if ((retval = nc_get_var_float(ncid, ir_016_id, &ir_016[0][0])))
        ERR(retval);
    if ((retval = nc_get_var_float(ncid, ir_039_id, &ir_039[0][0])))
        ERR(retval);
    if ((retval = nc_get_var_float(ncid, ir_062_id, &ir_062[0][0])))
        ERR(retval);
    if ((retval = nc_get_var_float(ncid, ir_073_id, &ir_073[0][0])))
        ERR(retval);
    if ((retval = nc_get_var_float(ncid, ir_087_id, &ir_087[0][0])))
        ERR(retval);
    if ((retval = nc_get_var_float(ncid, ir_108_id, &ir_108[0][0])))
        ERR(retval);
    if ((retval = nc_get_var_float(ncid, ir_120_id, &ir_120[0][0])))
        ERR(retval);
    if ((retval = nc_get_var_float(ncid, ir_134_id, &ir_134[0][0])))
        ERR(retval);
    if ((retval = nc_get_var(ncid, lsm_id, &lsm[0][0])))
        ERR(retval);
    if ((retval = nc_get_var_float(ncid, skt_id, &skt[0][0])))
        ERR(retval);
    if ((retval = nc_get_var_float(ncid, solzen_id, &solzen[0][0])))
        ERR(retval);
    if ((retval = nc_get_var_float(ncid, satzen_id, &satzen[0][0])))
        ERR(retval);

    if ((retval = nc_close(ncid)))
        ERR(retval);

    py_ann_cma(*vis006, *vis008, *ir_016, *ir_039, *ir_062, *ir_073, 
               *ir_087, *ir_108, *ir_120, *ir_134, *lsm, *skt, *solzen, 
               *satzen, &nx, &ny, *cot, *cma, *cma_unc, &msg_num, 
               &undo_true_refl);

    int i, j, cnt;
    cnt = 0;
    for (i=0; i<nx; i++)
    {
        for (j=0; j<ny; j++)
            if (cma[i][j] >= 0)
            {
                meanval = meanval + cma[i][j];
                cnt = cnt +1;
            }
    }
    meanval = meanval/cnt;
    printf("\n  C CMA mean: %f\n\n", meanval);

    return 0;
}
