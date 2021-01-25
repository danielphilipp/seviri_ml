SCRIPTPATH=/perm/ms/de/sf7/esa_cci_c_proc/orac_repository/ectrans_DP.R
#REMOTEPATH=/cmsaf/cloud_cci/esa_cloud_cci/software/model_nn15_lsm_skt_full_cph_linp.h5
#HOSTPATH=/perm/ms/de/sf7/esa_cci_c_proc/orac_repository/orac_devel/orac/seviri_neuralnet/data/model_nn15_lsm_skt_full_cph_linp.h5

#FILENAME=CMA_14_150_150_1_LOSS-MSE_OPT-ADAM_LR-0001_NE-300_BS-200_THEANO_GSICS.h5
FILENAME=cma_nn_output_20100715_1200.png
REMOTEPATH=/cmsaf/cloud_cci/esa_cloud_cci/software/${FILENAME}
HOSTPATH=/perm/ms/de/sf7/esa_cci_c_proc/orac_repository/orac_devel/orac/external_ml/seviri/data/${FILENAME}

echo Rscript $SCRIPTPATH $REMOTEPATH $HOSTPATH get
Rscript $SCRIPTPATH $REMOTEPATH $HOSTPATH get
