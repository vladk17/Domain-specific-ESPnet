
git clone https://github.com/espnet/espnet
cd espnet
cd tools

make KALDI='../..kaldi' CUPY_VERSION=''
make check_install CUPY_VERSION=''
