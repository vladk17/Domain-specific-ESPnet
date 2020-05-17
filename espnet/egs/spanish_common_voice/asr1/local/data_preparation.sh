#!/bin/bash

cd "$(dirname "$0")"
pip install -r requirements.txt

add-apt-repository ppa:mc3man/trusty-media
apt-get -y update
apt install -y ffmpeg

python3 make_data.py || exit 1

echo "Downsampling files to 16000 kHz"

cd ..
mkdir -p resampled_downloads
cd downloads/

for file in *.wav
do
    sox $file -r 16000 "../resampled_downloads/${file}"
done

cd ..
rm -rf downloads
mv resampled_downloads downloads

chmod -R 777 ../data
chmod -R 777 ../downloads