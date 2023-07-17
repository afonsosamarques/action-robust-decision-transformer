#!/bin/bash -l
source ./envs/ardt-env/bin/activate && cd ~/action-robust-decision-transformer/codebase/ && python3 ardt.pipeline --config_name pipeline-example --is_test_run
