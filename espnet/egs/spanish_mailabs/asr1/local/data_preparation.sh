#!/bin/bash

cd "$(dirname "$0")"
pip install -r requirements.txt
sudo apt install ffmpeg
python3 -m utils.prepare_data || exit 1

chmod -R 777 ../data
chmod -R 777 ../downloads

