#$ -S /bin/bash

#$ -N ardt_vanilla-halfcheetah-dataset_combo_v4
#$ -l gpu=1
#$ -l tmem=8G
#$ -l h_rt=55:00:00

#$ -wd /home/amarques/action-robust-decision-transformer/codebase

source /share/apps/source_files/python/python-3.9.5.source
source /share/apps/source_files/cuda/cuda-11.2.source
export PATH=$PATH:/share/apps/git-lfs-2.11.0/bin/

source /home/amarques/envs/ardt-env/bin/activate

nvidia-smi
python3 -m ardt.pipeline --config_name ardt_vanilla-halfcheetah-dataset_combo_v4
