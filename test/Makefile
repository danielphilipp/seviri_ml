# set compilers GNU
CC=gcc
F90=gfortran

# full path to your numpy core includes
# get your numpy include path with:
# ../get_py_config.sh
NUMPYINCLUDE=/path/to/your/numpy/includes

# full path to your python includes
# get your Python include path with:
# ../get_py_config.sh
PYINCLUDE=/path/to/your/python/includes
PYLIB=/path/to/your/python/libs

# path to your local seviri_ml directory
SEVIRI_ML=/path/to/your/seviri_ml

# NetCDF
NCDF_LIB = /path/to/your/netcdf/lib
NCDF_INCLUDE = /path/to/your/netcdf/include
NCDF_FORTRAN_LIB = /path/to/your/netcdf_fortran/lib
NCDF_FORTRAN_INCLUDE = /path/to/your/netcdf_fortran/include


# set Fortran compiler flags for GNU
FFLAGS=-O3 -ffree-line-length-0

# set C compiler flags for GNU
CFLAGS=-O3 -std=c99


# -----------------------------
all: fort90testexe ctestexe

# set includes
INC=-I${SEVIRI_ML} \
    -I${PYINCLUDE} \
    -I${NUMPYINCLUDE} \
    -I${NCDF_FORTRAN_INCLUDE} \
    -I${NCDF_INCLUDE}

LIB=-L${SEVIRI_ML} -lsevann \
    -L${PYLIB} -lpython3.8 \
    -L${NCDF_LIB} -lnetcdf \
    -L${NCDF_FORTRAN_LIB} -lnetcdff

ctestexe: c_test.c
	$(CC) $(INC) $(CFLAGS) c_test.c -o ctestexe $(LIB)

fort90testexe: f90_test.F90
	$(F90) $(INC) $(FFLAGS) f90_test.F90 -o fort90testexe $(LIB)

clean:
	rm -f *.a *.o *.out ctestexe fort90testexe
