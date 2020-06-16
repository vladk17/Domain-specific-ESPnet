#!/bin/bash

./run.sh --docker_gpu 8,9,10,11 --docker_egs spanish_merge/asr1 --ngpu 4 --iteration 5_tedx \
--iteration_train_datasets "data/train_tedx" --iteration_val_datasets "data/test_tedx"