#!/bin/bash -l
source ./ardt-env/ardt/bin/activate && python3 ~/action-robust-decision-transformer/ardt/pipeline.py --config_name ardt_vanilla-halfcheetah-d4rl_expert --is_test_run
