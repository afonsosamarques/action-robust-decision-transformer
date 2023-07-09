#!/bin/bash -l

#$ -N FirstRealLongIterativeRun
#$ -l gpu=2
#$ -l mem=8G
#$ -l tmpfs=2G
#$ -l h_rt=46:00:00

#$ -wd /home/ucabscm

module purge
module load default-modules
module remove mpi compilers
module remove git/2.32.0
module load git/2.41.0-lfs-3.3.0
module load python/3.9.10

source ardt-env/ardt/bin/activate

nvidia-smi
python3 action-robust-decision-transformer/ardt/pipeline.py --config_name ardt_vanilla-halfcheetah-d4rl_expert
