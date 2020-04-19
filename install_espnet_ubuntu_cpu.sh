
git clone https://github.com/espnet/espnet
cd espnet
cd tools

make KALDI='../../kaldi' PYTHON=/usr/bin/python3.6 CUPY_VERSION='' -j 10
make check_install CUPY_VERSION=''
