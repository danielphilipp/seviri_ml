echo "########### YOUR NUMPY INCLUDE PATH ##########"
python3 -c "import numpy as np; import os; print(os.path.join(np.get_include(), 'numpy'))"
echo "########### YOUR PYTHON INCLUDE PATH ##########"
python3-config --includes
echo "########### YOUR PYTHON LIBRARY PATH ###########"
python3-config --ldflags
