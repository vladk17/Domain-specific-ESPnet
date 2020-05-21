#!/bin/bash

cd "$(dirname "$0")"
pip install -r requirements.txt

add-apt-repository ppa:mc3man/trusty-media
apt-get -y update
apt install -y ffmpeg

python3 make_data.py

echo "Downsampling files to 16000 kHz"

cd ..
mkdir -p resampled_crowdsource
cd downloads/crowdsource

for file in *.wav
do
    sox $file -r 16000 "../../resampled_crowdsource/${file}"
done

cd ..
rm -rf downloads/crowdsource
mv resampled_downloads downloads/crowdsource

chmod -R 777 ../data
chmod -R 777 ../downloads

