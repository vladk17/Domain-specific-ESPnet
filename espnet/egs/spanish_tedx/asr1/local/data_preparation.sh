#!/bin/bash

cd "$(dirname "$0")"
pip install -r requirements.txt
python -m utils.prepare_tedx_spanish

chmod -R 777 ../data
chmod -R 777 ../downloads

