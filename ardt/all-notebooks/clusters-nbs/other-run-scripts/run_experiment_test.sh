#!/bin/bash -l
source ./envs/ardt-env/bin/activate && python3 ~/action-robust-decision-transformer/ardt/pipeline.py --config_name ardt_vanilla-halfcheetah-d4rl_expert --is_test_run
