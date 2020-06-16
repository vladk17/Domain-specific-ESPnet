#!/bin/bash

./run.sh --docker_gpu 4,5,6,7 --docker_egs spanish_merge/asr1 --docker_env --ngpu 4 --iteration 4_comvoice \
--iteration_train_datasets "data/train_comvoice" --iteration_val_datasets "data/test_comvoice"