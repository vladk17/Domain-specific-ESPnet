#!/bin/bash

./run.sh --docker_gpu 12,13,14,15 --docker_egs spanish_merge/asr1 --ngpu 4 --iteration 6_mailabs \
--iteration_train_datasets "data/train_mailabs" --iteration_val_datasets "data/test_mailabs"