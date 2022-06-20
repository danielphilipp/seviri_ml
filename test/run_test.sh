echo ""
echo "################### CHECK PYTHON ####################"
echo ""
python3 py_test.py 
echo ""
echo "################### CHECK FORTRAN INTERFACE ####################"
echo ""
./fort90testexe
echo ""
echo "################### CHECK C INTERFACE ####################"
echo ""
./ctestexe


