#!/bin/bash

./run.sh --docker_gpu 12,13,14,15 --docker_egs spanish_merge/asr1 --ngpu 4  \
--docker_env "ITERATION_NAME=6_mailabs,ITERATION_TRAIN=data/train_mailabs,ITERATION_VAL=data/test_mailabs"