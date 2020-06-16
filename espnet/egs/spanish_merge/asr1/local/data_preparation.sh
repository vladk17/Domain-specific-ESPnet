#!/bin/bash

cd "$(dirname "$0")"
pip install -r requirements.txt

add-apt-repository ppa:mc3man/trusty-media
apt-get -y update
apt install -y ffmpeg

#python3 make_data.py

chmod -R 777 ../data
chmod -R 777 ../downloads
