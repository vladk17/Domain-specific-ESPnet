#!/bin/bash

./run.sh --docker_gpu 0,1,2,3 --docker_egs spanish_merge/asr1 --ngpu 4 --stage 1 --stop_stage 2 \
--docker_env "ITERATION_NAME=3_crowdsource,ITERATION_TRAIN=data/train_crowdsources,ITERATION_VAL=data/test_crowdsource"