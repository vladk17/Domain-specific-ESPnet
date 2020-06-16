#!/bin/bash

./run.sh --docker_gpu 0,1,2,3 --docker_egs spanish_merge/asr1 --docker_env --ngpu 4 --iteration 3_crowdsource \
--iteration_train_datasets "data/train_crowdsource" --iteration_val_datasets "data/test_crowdsource"