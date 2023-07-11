#!/bin/bash -l

#$ -N TestRun
#$ -l gpu=true
#$ -l tmem=4G
#$ -l h_rt=02:00:00

#$ -wd /home/amarques

nvidia-smi
python3 ./action-robust-decision-transformer/ardt/pipeline.py --config_name ardt_vanilla-halfcheetah-d4rl_expert --is_test_run
