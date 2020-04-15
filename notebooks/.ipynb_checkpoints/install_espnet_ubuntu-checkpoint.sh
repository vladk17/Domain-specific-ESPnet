#!/bin/bash

# apt-get install cmake
# apt-get install sox
# apt-get install ffmpeg
# apt-get install flac

# NUMBER_OF_CPUs=4 # Edit number of CPUs

# echo 'CUDAROOT=/path/to/cuda' >> ~/.bashrc
# echo 'export PATH=$CUDAROOT/bin:$PATH' >> ~/.bashrc

# echo 'export LD_LIBRARY_PATH=$CUDAROOT/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# echo 'export CFLAGS="-I$CUDAROOT/include $CFLAGS"' >> ~/.bashrc
# echo 'export CUDA_HOME=$CUDAROOT' >> ~/.bashrc
# echo 'export CUDA_PATH=$CUDAROOT' >> ~/.bashrc

# git clone https://github.com/kaldi-asr/kaldi
# cd kaldi

# cd tools
# make -j $NUMBER_OF_CPUs 

# ./extras/install_openblas.sh
# ./extras/install_mkl.sh
# apt-get install libatlas-base-dev

# cd src
# ./configure --openblas-root=../tools/OpenBLAS/install --use-cuda=no
# make -j clean depend; make -j $NUMBER_OF_CPUs

git clone https://github.com/espnet/espnet
cd espnet
cd tools
make KALDI=../kaldi
make check_install