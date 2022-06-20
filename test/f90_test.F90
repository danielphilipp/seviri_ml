program test_f90

use netcdf
use seviri_neural_net_m

implicit none

character(len=*), parameter :: FILE_NAME = "seviri_ml_test_201907011200.nc"

integer, parameter :: nx=100, ny=100
integer :: ncid, vis006_id, vis008_id, ir_016_id, ir_039_id, &
           & ir_062_id, ir_073_id, ir_087_id, ir_108_id, ir_120_id, &
           & ir_134_id, lsm_id, skt_id, solzen_id, satzen_id

real(kind=4), dimension(:,:), pointer :: vis006, vis008, ir_016, &
                                         & ir_039, ir_062, ir_073, &
                                         & ir_087, ir_108, ir_120, &
                                         & ir_134, skt, satzen, solzen, &
                                         & cot, cma_unc
integer(kind=1), dimension(:,:), pointer :: lsm, cma
integer(kind=1) :: msg_num
logical(kind=1) :: undo_true_refl
real(kind=4) :: meanval
msg_num = 0

allocate(vis006(nx, ny))
allocate(vis008(nx, ny))
allocate(ir_016(nx, ny))
allocate(ir_039(nx, ny))
allocate(ir_062(nx, ny))
allocate(ir_073(nx, ny))
allocate(ir_087(nx, ny))
allocate(ir_108(nx, ny))
allocate(ir_120(nx, ny))
allocate(ir_134(nx, ny))
allocate(skt(nx, ny))
allocate(lsm(nx, ny))
allocate(satzen(nx, ny))
allocate(solzen(nx, ny))
allocate(cot(nx, ny))
allocate(cma(nx, ny))
allocate(cma_unc(nx, ny))

call check( nf90_open(FILE_NAME, NF90_NOWRITE, ncid) )

call check( nf90_inq_varid(ncid, 'VIS006', vis006_id) )
call check( nf90_inq_varid(ncid, 'VIS008', vis008_id) )
call check( nf90_inq_varid(ncid, 'IR_016', ir_016_id) )
call check( nf90_inq_varid(ncid, 'IR_039', ir_039_id) )
call check( nf90_inq_varid(ncid, 'WV_062', ir_062_id) )
call check( nf90_inq_varid(ncid, 'WV_073', ir_073_id) )
call check( nf90_inq_varid(ncid, 'IR_087', ir_087_id) )
call check( nf90_inq_varid(ncid, 'IR_108', ir_108_id) )
call check( nf90_inq_varid(ncid, 'IR_120', ir_120_id) )
call check( nf90_inq_varid(ncid, 'IR_134', ir_134_id) )
call check( nf90_inq_varid(ncid, 'lsm', lsm_id) )
call check( nf90_inq_varid(ncid, 'skt', skt_id) )
call check( nf90_inq_varid(ncid, 'solzen', solzen_id) )
call check( nf90_inq_varid(ncid, 'satzen', satzen_id) )

call check( nf90_get_var(ncid, vis006_id, vis006) )
call check( nf90_get_var(ncid, vis008_id, vis008) )
call check( nf90_get_var(ncid, ir_016_id, ir_016) )
call check( nf90_get_var(ncid, ir_039_id, ir_039) )
call check( nf90_get_var(ncid, ir_062_id, ir_062) )
call check( nf90_get_var(ncid, ir_073_id, ir_073) )
call check( nf90_get_var(ncid, ir_087_id, ir_087) )
call check( nf90_get_var(ncid, ir_108_id, ir_108) )
call check( nf90_get_var(ncid, ir_120_id, ir_120) )
call check( nf90_get_var(ncid, ir_134_id, ir_134) )
call check( nf90_get_var(ncid, lsm_id, lsm) )
call check( nf90_get_var(ncid, skt_id, skt) )
call check( nf90_get_var(ncid, solzen_id, solzen) )
call check( nf90_get_var(ncid, satzen_id, satzen) )

call seviri_ann_cma(nx, ny, vis006, vis008, ir_016, ir_039, &
                    ir_062, ir_073, ir_087, ir_108, ir_120, &
                    ir_134, lsm, skt, solzen, satzen, cot, &
                    cma, cma_unc, msg_num, undo_true_refl)

call get_mean(cma, meanval, nx, ny)
write(*,*) " "
write(*,*) " FORTRAN90 CMA mean: ", meanval
write(*,*) " "

contains

    subroutine check(status)
        integer, intent(in) :: status
    
        if(status /= nf90_noerr) then 
            print *, trim(nf90_strerror(status))
            stop "Stopped"
        end if
    end subroutine check


    subroutine get_mean(arr, meanval, nx, ny)
        real(kind=4), intent(inout) :: meanval
        integer :: nx, ny, cnt, i, j
        integer(kind=1), pointer, intent(in) :: arr(:,:)

        meanval = 0
        cnt = 0
        do i=1, nx
            do j=1, ny
                if (arr(i,j) >= 0) then
                    meanval = meanval + arr(i,j)
                    cnt = cnt + 1
                endif
            enddo
        enddo

        meanval = meanval/cnt
    end subroutine get_mean
    
end program test_f90
