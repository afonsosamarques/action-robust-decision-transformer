#!/bin/bash -l

#$ -N ardt_vanilla-halfcheetah-dataset_combo
#$ -l gpu=1
#$ -l mem=8G
#$ -l tmpfs=2G
#$ -l h_rt=12:00:00

#$ -wd /home/ucabscm/action-robust-decision-transformer/codebase

module purge
module load default-modules
module remove mpi compilers
module remove git/2.32.0
module load git/2.41.0-lfs-3.3.0
module load python/3.9.10

source /home/ucabscm/envs/ardt-env/bin/activate

nvidia-smi
python3 -m ardt.pipeline --config_name ardt_vanilla-halfcheetah-dataset_combo
