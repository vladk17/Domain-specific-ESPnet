#!/bin/bash

./run.sh --docker_gpu 8,9,10,11 --docker_egs spanish_merge/asr1 --ngpu 4 --stage 5 --stop_stage 999 \
--docker_env "ITERATION_NAME=5_tedx,ITERATION_TRAIN=data/train_tedx,ITERATION_VAL=data/test_tedx"