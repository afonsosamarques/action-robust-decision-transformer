#$ -S /bin/bash

#$ -N ardt_simplest-halfcheetah-dataset_combo_v6
#$ -l gpu=1
#$ -l tmem=4G
#$ -l h_rt=8:00:00

#$ -wd /home/amarques/action-robust-decision-transformer/codebase

source /share/apps/source_files/python/python-3.9.5.source

export PATH=$PATH:/share/apps/git-lfs-2.11.0/bin/

source /home/amarques/envs/ardt-env/bin/activate

nvidia-smi
python3 -m ardt.pipeline --config_name ardt_simplest-halfcheetah-dataset_combo_v6
