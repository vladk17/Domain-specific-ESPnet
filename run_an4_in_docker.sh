#!/bin/bash

git clone https://github.com/espnet/espnet
cd espnet/docker
./run.sh --docker_gpu -1 --docker_egs an4/asr1 --ngpu 0


