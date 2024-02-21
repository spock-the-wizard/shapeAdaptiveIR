
cmake -D CMAKE_C_COMPILER=gcc-9                       \
      -D CMAKE_CXX_COMPILER=g++-9                     \
      -D PYTHON_INCLUDE_PATH=/usr/include/python3.8/  \
      -D OpenEXR_ROOT=/usr/include/OpenEXR/           \
      -D ENOKI_DIR=/home/enoki \
      ..
make -j