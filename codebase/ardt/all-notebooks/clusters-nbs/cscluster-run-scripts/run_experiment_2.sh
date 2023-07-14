#!/bin/bash -l

#$ -N ardt_simplest-halfcheetah-rarl_train_v2-envadv
#$ -l gpu=1
#$ -l tmem=4G
#$ -l h_rt=6:00:00

#$ -wd /home/amarques

nvidia-smi
python3 ./action-robust-decision-transformer/ardt/pipeline.py --config_name ardt_simplest-halfcheetah-rarl_train_v2-envadv
