#!/bin/bash -l
source ./ardt-env/ardt/bin/activate && python3 action-robust-decision-transformer/ardt/pipeline.py --config_name ardt_vanilla-halfcheetah-rarl_train_v1
