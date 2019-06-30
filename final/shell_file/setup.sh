cd ../src
pip install . --user
python setup.py build_ext --inplace
cd ../data
python preprocess.py
