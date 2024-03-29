!------------------------------------------------------------------------------
! Name: seviri_neural_net.F90
!
! Purpose:
! Module for SEVIRI neural network cloud detection and cloud phase 
! determination. Neural network prediction is done in Python using 
! the Keras library with Tensorflow or Theano backends. This module
! establishes an interface to a C layer from which the Python neural 
! network module is called. Neural net input (radiances and auxuiliary 
! data) are passed to the Python module through the C interface. The 
! predicted field is then passed back to this module.
!
! History:
! 2020/07/20, DP: Initial version
! 2020/09/30, DP: Moved assignment of NN results to Fortran arrays to C.
!                 Fixed memory leak. Major Simplifications + improvements 
!                 + cleanup. Revised subroutine header.
!
! Bugs:
! None known.
!------------------------------------------------------------------------------

module seviri_neural_net_m
    use iso_c_binding
     
    implicit none
     
    interface
        ! interface to C function py_ann_cot_cph
        subroutine py_ann_cma(vis006, vis008, nir016, ir039, &
                              ir062, ir073, ir087, ir108, ir120, &
                              ir134, lsm, skt, solzen, satzen, nx, ny, reg_cot, &
                              bin_cot, unc_cot, msg_index, &
                              undo_true_reflectances) bind(C, name="py_ann_cma")
            import :: c_ptr
            import :: c_int
            import :: c_float
            import :: c_signed_char
            import :: c_bool
            type(c_ptr), value :: vis006
            type(c_ptr), value :: vis008
            type(c_ptr), value :: nir016
            type(c_ptr), value :: ir039
            type(c_ptr), value :: ir062
            type(c_ptr), value :: ir073
            type(c_ptr), value :: ir087
            type(c_ptr), value :: ir108
            type(c_ptr), value :: ir120
            type(c_ptr), value :: ir134
            type(c_ptr), value :: lsm
            type(c_ptr), value :: skt
            type(c_ptr), value :: solzen
            type(c_ptr), value :: satzen
            real(c_float), dimension(*), intent(out) :: reg_cot, unc_cot
            integer(c_signed_char), dimension(*), intent(out) :: bin_cot
            integer(c_int) :: nx, ny
            integer(c_signed_char) :: msg_index
            logical(c_bool) :: undo_true_reflectances
        end subroutine py_ann_cma


         ! interface to C function py_ann_cot_cph
        subroutine py_ann_cph(vis006, vis008, nir016, ir039, &
                              ir062, ir073, ir087, ir108, ir120, &
                              ir134, lsm, skt, solzen, satzen, nx, ny, &
                              reg_cph, bin_cph, unc_cph, cldmask, msg_index, &
                              undo_true_reflectances) bind(C, name="py_ann_cph")
            import :: c_ptr
            import :: c_int
            import :: c_float
            import :: c_signed_char
            import :: c_bool
            type(c_ptr), value :: vis006
            type(c_ptr), value :: vis008
            type(c_ptr), value :: nir016
            type(c_ptr), value :: ir039
            type(c_ptr), value :: ir062
            type(c_ptr), value :: ir073
            type(c_ptr), value :: ir087
            type(c_ptr), value :: ir108
            type(c_ptr), value :: ir120
            type(c_ptr), value :: ir134
            type(c_ptr), value :: lsm
            type(c_ptr), value :: skt
            type(c_ptr), value :: solzen
            type(c_ptr), value :: satzen
            type(c_ptr), value :: cldmask
            real(c_float), dimension(*), intent(out) :: reg_cph, unc_cph
            integer(c_signed_char), dimension(*), intent(out) :: bin_cph
            integer(c_int) :: nx, ny
            integer(c_signed_char) :: msg_index
            logical(c_bool) :: undo_true_reflectances
        end subroutine py_ann_cph


        ! interface to C function py_ann_ctp
        subroutine py_ann_ctp(vis006, vis008, nir016, ir039, &
                              ir062, ir073, ir087, ir108, ir120, &
                              ir134, lsm, skt, solzen, satzen, nx, ny, ctp, &
                              ctp_unc, cldmask, msg_index, &
                              undo_true_reflectances) bind(C, name="py_ann_ctp")
            import :: c_ptr
            import :: c_int
            import :: c_float
            import :: c_signed_char
            import :: c_bool
            type(c_ptr), value :: vis006
            type(c_ptr), value :: vis008
            type(c_ptr), value :: nir016
            type(c_ptr), value :: ir039
            type(c_ptr), value :: ir062
            type(c_ptr), value :: ir073
            type(c_ptr), value :: ir087
            type(c_ptr), value :: ir108
            type(c_ptr), value :: ir120
            type(c_ptr), value :: ir134
            type(c_ptr), value :: lsm
            type(c_ptr), value :: skt
            type(c_ptr), value :: solzen
            type(c_ptr), value :: satzen
            type(c_ptr), value :: cldmask
            real(c_float), dimension(*), intent(out) :: ctp, ctp_unc
            integer(c_int) :: nx, ny
            integer(c_signed_char) :: msg_index
            logical(c_bool) :: undo_true_reflectances
        end subroutine py_ann_ctp

        ! interface to C function py_ann_ctt
        subroutine py_ann_ctt(vis006, vis008, nir016, ir039, &
                              ir062, ir073, ir087, ir108, ir120, &
                              ir134, lsm, skt, solzen, satzen, nx, ny, ctt, &
                              ctt_unc, cldmask, msg_index, &
                              undo_true_reflectances) bind(C, name="py_ann_ctt")
            import :: c_ptr
            import :: c_int
            import :: c_float
            import :: c_signed_char
            import :: c_bool
            type(c_ptr), value :: vis006
            type(c_ptr), value :: vis008
            type(c_ptr), value :: nir016
            type(c_ptr), value :: ir039
            type(c_ptr), value :: ir062
            type(c_ptr), value :: ir073
            type(c_ptr), value :: ir087
            type(c_ptr), value :: ir108
            type(c_ptr), value :: ir120
            type(c_ptr), value :: ir134
            type(c_ptr), value :: lsm
            type(c_ptr), value :: skt
            type(c_ptr), value :: solzen
            type(c_ptr), value :: satzen
            type(c_ptr), value :: cldmask
            real(c_float), dimension(*), intent(out) :: ctt, ctt_unc
            integer(c_int) :: nx, ny
            integer(c_signed_char) :: msg_index
            logical(c_bool) :: undo_true_reflectances
        end subroutine py_ann_ctt


        ! interface to C function py_ann_ctt
        subroutine py_ann_cbh(ir108, ir120, ir134, solzen, satzen, nx, &
                              ny, cbh, cbh_unc, cldmask) bind(C, name="py_ann_cbh")
            import :: c_ptr
            import :: c_int
            import :: c_float
            import :: c_char
            import :: c_bool
            type(c_ptr), value :: ir108
            type(c_ptr), value :: ir120
            type(c_ptr), value :: ir134
            type(c_ptr), value :: solzen
            type(c_ptr), value :: satzen
            type(c_ptr), value :: cldmask
            real(c_float), dimension(*), intent(out) :: cbh, cbh_unc
            integer(c_int) :: nx, ny
        end subroutine py_ann_cbh


        ! interface to C function py_ann_mlay
        subroutine py_ann_mlay(vis006, vis008, nir016, ir039, &
                              ir062, ir073, ir087, ir108, ir120, &
                              ir134, lsm, skt, solzen, satzen, nx, ny, &
                              mlay_reg, mlay_bin, mlay_unc, cldmask, msg_index, &
                              undo_true_reflectances) bind(C, name="py_ann_mlay")
            import :: c_ptr
            import :: c_int
            import :: c_float
            import :: c_signed_char
            import :: c_bool
            type(c_ptr), value :: vis006
            type(c_ptr), value :: vis008
            type(c_ptr), value :: nir016
            type(c_ptr), value :: ir039
            type(c_ptr), value :: ir062
            type(c_ptr), value :: ir073
            type(c_ptr), value :: ir087
            type(c_ptr), value :: ir108
            type(c_ptr), value :: ir120
            type(c_ptr), value :: ir134
            type(c_ptr), value :: lsm
            type(c_ptr), value :: skt
            type(c_ptr), value :: solzen
            type(c_ptr), value :: satzen
            type(c_ptr), value :: cldmask
            real(c_float), dimension(*), intent(out) :: mlay_reg, mlay_unc
            integer(c_signed_char), dimension(*), intent(out) :: mlay_bin
            integer(c_int) :: nx, ny
            integer(c_signed_char) :: msg_index
            logical(c_bool) :: undo_true_reflectances
        end subroutine py_ann_mlay

    end interface
contains


!------------------------------------------------------------------------------
! Name: seviri_ann_cph_cot
!
! Purpose:
! Subroutine accepting neural network input data from ORAC which are 
! passed to C and the Python neural network subsequently. Calls the C 
! interface function
!
! Arguments:
! Name                 Type  I/O Description
!------------------------------------------------------------------------------
! nx                   int   In  Dimension in x direction
! ny                   int   In  Dimension in y direction
! vis006               2darr In  SEVIRI VIS006 measurements
! vis008               2darr In  SEVIRI VIS008 measurements
! nir016               2darr In  SEVIRI NIR016 measurements
! ir039                2darr In  SEVIRI IR039 measurements
! ir062                2darr In  SEVIRI IR062 measurements
! ir073                2darr In  SEVIRI IR073 measurements
! ir087                2darr In  SEVIRI IR087 measurements
! ir108                2darr In  SEVIRI IR108 measurements
! ir120                2darr In  SEVIRI IR120 measurements
! ir134                2darr In  SEVIRI IR134 measurements
! lsm                  2darr In  Land-sea mask
! skt                  2darr In  Skin temperature
! solzen               2darr In  Solar Zenith Angle
! regression_cot       2darr Out COT NN regression value
! binary_cot           2darr Out COT binary value after thresholding (CMA)
! uncertainty_cot      2darr Out COT uncertainty of CMA
! regression_cph       2darr Out CPH NN regression value
! binary_cph           2darr Out CPH binary value after thresholding
! uncertainty_cph      2darr Out CPH uncertainty of CPH
!------------------------------------------------------------------------------

subroutine seviri_ann_cma(nx, ny, vis006, vis008, nir016, ir039, ir062, ir073, &
                          ir087, ir108, ir120, ir134, lsm, skt, solzen, satzen, &
                          regression_cot, binary_cot, uncertainty_cot, &
                          msg_index, undo_true_reflectances)
    use iso_c_binding
    
    ! output arrays 
    real(c_float), intent(out) :: regression_cot(:,:), uncertainty_cot(:,:)
    integer(c_signed_char), intent(out) :: binary_cot(:,:)

    ! C-types
    integer(c_int) :: nx ,ny
    integer(c_signed_char) :: msg_index
    real(c_float), dimension(nx,ny), target :: vis006, vis008, nir016, ir039, &
                                               & ir062, ir073, ir087, ir108, &
                                               & ir120, ir134, skt, solzen, &
                                               & satzen
    integer(c_signed_char), dimension(nx,ny), target :: lsm
    logical(kind=1) :: undo_true_reflectances
 
    ! Call Python neural network via Python C-API
    call py_ann_cma(c_loc(vis006(1,1)), c_loc(vis008(1,1)), c_loc(nir016(1,1)), &
                    c_loc(ir039(1,1)), c_loc(ir062(1,1)), c_loc(ir073(1,1)), &
                    c_loc(ir087(1,1)), c_loc(ir108(1,1)), c_loc(ir120(1,1)), &
                    c_loc(ir134(1,1)), c_loc(lsm(1,1)), c_loc(skt(1,1)), &
                    c_loc(solzen(1,1)), c_loc(satzen(1,1)), nx, ny, &
                    regression_cot, binary_cot, uncertainty_cot, &
                    msg_index, undo_true_reflectances)

end subroutine seviri_ann_cma


subroutine seviri_ann_cph(nx, ny, vis006, vis008, nir016, ir039, ir062, ir073, &
                          ir087, ir108, ir120, ir134, lsm, skt, solzen, satzen, &
                          regression_cph, binary_cph, uncertainty_cph, cldmask, &
                          msg_index, undo_true_reflectances)
    use iso_c_binding

    ! output arrays
    real(c_float), intent(out) :: regression_cph(:,:), uncertainty_cph(:,:)
    integer(c_signed_char), intent(out) :: binary_cph(:,:)

    ! C-types
    integer(c_int) :: nx ,ny
    integer(c_signed_char) :: msg_index
    real(c_float), dimension(nx,ny), target :: vis006, vis008, nir016, ir039, &
                                               & ir062, ir073, ir087, ir108, &
                                               & ir120, ir134, skt, solzen, &
                                               & satzen

    integer(c_signed_char), dimension(nx,ny), target :: lsm, cldmask
    logical(kind=1) :: undo_true_reflectances

    ! Call Python neural network via Python C-API
    call py_ann_cph(c_loc(vis006(1,1)), c_loc(vis008(1,1)), c_loc(nir016(1,1)), &
                    c_loc(ir039(1,1)), c_loc(ir062(1,1)), c_loc(ir073(1,1)), &
                    c_loc(ir087(1,1)), c_loc(ir108(1,1)), c_loc(ir120(1,1)), &
                    c_loc(ir134(1,1)), c_loc(lsm(1,1)), c_loc(skt(1,1)), &
                    c_loc(solzen(1,1)), c_loc(satzen(1,1)), &
                    nx, ny, regression_cph, binary_cph, uncertainty_cph, &
                    c_loc(cldmask(1,1)), msg_index, undo_true_reflectances)

end subroutine seviri_ann_cph


subroutine seviri_ann_ctp(nx, ny, vis006, vis008, nir016, ir039, ir062, ir073, &
                        ir087, ir108, ir120, ir134, lsm, skt, solzen, satzen, &
                        ctp, ctp_unc, cldmask, msg_index, undo_true_reflectances)
    use iso_c_binding

    ! output arrays
    real(c_float), intent(out) :: ctp(:,:), ctp_unc(:,:)

    ! C-types
    integer(c_int) :: nx ,ny
    integer(c_signed_char) :: msg_index
    real(c_float), dimension(nx,ny), target :: vis006, vis008, nir016, ir039, &
                                               & ir062, ir073, ir087, ir108, &
                                               & ir120, ir134, skt, solzen, &
                                               & satzen
    integer(c_signed_char), dimension(nx,ny), target :: lsm, cldmask
    logical(kind=1) :: undo_true_reflectances

    ! Call Python neural network via Python C-API
    call py_ann_ctp(c_loc(vis006(1,1)), c_loc(vis008(1,1)),c_loc(nir016(1,1)), &
                    c_loc(ir039(1,1)), c_loc(ir062(1,1)), c_loc(ir073(1,1)), &
                    c_loc(ir087(1,1)), c_loc(ir108(1,1)), c_loc(ir120(1,1)), &
                    c_loc(ir134(1,1)), c_loc(lsm(1,1)), c_loc(skt(1,1)), &
                    c_loc(solzen(1,1)), c_loc(satzen(1,1)),nx, ny, ctp, &
                    ctp_unc, c_loc(cldmask(1,1)), msg_index, undo_true_reflectances)

end subroutine seviri_ann_ctp


subroutine seviri_ann_ctt(nx, ny, vis006, vis008, nir016, ir039, ir062, ir073, &
                        ir087, ir108, ir120, ir134, lsm, skt, solzen, satzen, &
                        ctt, ctt_unc, cldmask, msg_index, undo_true_reflectances)
    use iso_c_binding

    ! output arrays
    real(c_float), intent(out) :: ctt(:,:), ctt_unc(:,:)

    ! C-types
    integer(c_int) :: nx ,ny
    integer(c_signed_char) :: msg_index
    real(c_float), dimension(nx,ny), target :: vis006, vis008, nir016, ir039, &
                                               & ir062, ir073, ir087, ir108, &
                                               & ir120, ir134, skt, solzen, &
                                               & satzen
    integer(c_signed_char), dimension(nx,ny), target :: lsm, cldmask
    logical(kind=1) :: undo_true_reflectances

    ! Call Python neural network via Python C-API
    call py_ann_ctt(c_loc(vis006(1,1)), c_loc(vis008(1,1)),c_loc(nir016(1,1)), &
                    c_loc(ir039(1,1)), c_loc(ir062(1,1)), c_loc(ir073(1,1)), &
                    c_loc(ir087(1,1)), c_loc(ir108(1,1)), c_loc(ir120(1,1)), &
                    c_loc(ir134(1,1)), c_loc(lsm(1,1)), c_loc(skt(1,1)), &
                    c_loc(solzen(1,1)), c_loc(satzen(1,1)),nx, ny, ctt, &
                    ctt_unc, c_loc(cldmask(1,1)), msg_index, undo_true_reflectances)

end subroutine seviri_ann_ctt


subroutine seviri_ann_cbh(nx, ny, ir108, ir120, ir134, solzen, satzen, cbh, cbh_unc, &
                          cldmask)
    use iso_c_binding

    ! output arrays
    real(c_float), intent(out) :: cbh(:,:), cbh_unc(:,:)

    ! C-types
    integer(c_int) :: nx ,ny
    integer(c_signed_char) :: msg_index
    real(c_float), dimension(nx,ny), target :: ir108, ir120, ir134, solzen, satzen
    integer(c_signed_char), dimension(nx,ny), target :: cldmask

    ! Call Python neural network via Python C-API
    call py_ann_cbh(c_loc(ir108(1,1)), c_loc(ir120(1,1)), c_loc(ir134(1,1)), &
                    c_loc(solzen(1,1)), c_loc(satzen(1,1)),nx, ny, cbh, &
                    cbh_unc, c_loc(cldmask(1,1)))

end subroutine seviri_ann_cbh


subroutine seviri_ann_mlay(nx, ny, vis006, vis008, nir016, ir039, ir062, ir073, &
                        ir087, ir108, ir120, ir134, lsm, skt, solzen, satzen, &
                        regression_mlay, binary_mlay, uncertainty_mlay, cldmask, &
                        msg_index, undo_true_reflectances)
    use iso_c_binding

    ! output arrays
    real(c_float), intent(out) :: regression_mlay(:,:), uncertainty_mlay(:,:)
    integer(c_signed_char), intent(out) :: binary_mlay(:,:)

    ! C-types
    integer(c_int) :: nx ,ny
    integer(c_signed_char) :: msg_index
    real(c_float), dimension(nx,ny), target :: vis006, vis008, nir016, ir039, &
                                               & ir062, ir073, ir087, ir108, &
                                               & ir120, ir134, skt, solzen, &
                                               & satzen
    integer(c_signed_char), dimension(nx,ny), target :: lsm, cldmask
    logical(kind=1) :: undo_true_reflectances

    ! Call Python neural network via Python C-API
    call py_ann_mlay(c_loc(vis006(1,1)), c_loc(vis008(1,1)), c_loc(nir016(1,1)), &
                     c_loc(ir039(1,1)), c_loc(ir062(1,1)), c_loc(ir073(1,1)), &
                     c_loc(ir087(1,1)), c_loc(ir108(1,1)), c_loc(ir120(1,1)), &
                     c_loc(ir134(1,1)), c_loc(lsm(1,1)), c_loc(skt(1,1)), &
                     c_loc(solzen(1,1)), c_loc(satzen(1,1)), nx, ny, &
                     regression_mlay, binary_mlay, uncertainty_mlay, &
                     c_loc(cldmask(1,1)), msg_index, undo_true_reflectances)

end subroutine seviri_ann_mlay

end module seviri_neural_net_m
