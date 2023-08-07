#$ -S /bin/bash

#$ -N ardt_full_v2
#$ -l gpu=1
#$ -l h_vmem=20G
#$ -l tmem=20G
#$ -l h_rt=36:00:00

#$ -wd /home/amarques/action-robust-decision-transformer/codebase

source /share/apps/source_files/python/python-3.9.5.source
export PATH=$PATH:/share/apps/git-lfs-2.11.0/bin/

source /home/amarques/envs/ardt-env/bin/activate

nvidia-smi
python3 -m ardt.pipeline --config_name ardt_full_v2