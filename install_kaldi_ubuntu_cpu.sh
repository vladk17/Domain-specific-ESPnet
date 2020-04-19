#!/bin/bash

WORK_DIR=$PWD

apt-get install -y cmake
apt-get install -y sox
apt-get install -y ffmpeg
apt-get install -y flac
apt-get install -y automake autoconf gfortran libtool subversion

NUMBER_OF_CPUs=4 # Edit number of CPUs

echo 'CUDAROOT=/path/to/cuda' >> ~/.bashrc
echo 'export PATH=$CUDAROOT/bin:$PATH' >> ~/.bashrc

echo 'export LD_LIBRARY_PATH=$CUDAROOT/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CFLAGS="-I$CUDAROOT/include $CFLAGS"' >> ~/.bashrc
echo 'export CUDA_HOME=$CUDAROOT' >> ~/.bashrc
echo 'export CUDA_PATH=$CUDAROOT' >> ~/.bashrc

git clone https://github.com/kaldi-asr/kaldi

cd kaldi
cd tools
make -j $NUMBER_OF_CPUs 

./extras/install_openblas.sh
./extras/install_mkl.sh
apt-get install -y libatlas-base-dev

cd src
./configure --openblas-root=../tools/OpenBLAS/install --use-cuda=no
make -j clean depend; make -j $NUMBER_OF_CPUs

echo 'Everything installed successfully'
