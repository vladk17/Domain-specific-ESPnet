#!/bin/bash

cd "$(dirname "$0")"
pip install -r requirements.txt

#add-apt-repository ppa:mc3man/trusty-media
#apt-get -y update
#apt install -y ffmpeg

python3 _spanish_common_voice_runner.py || exit 1

chmod -R 777 ../data
chmod -R 777 ../downloads

