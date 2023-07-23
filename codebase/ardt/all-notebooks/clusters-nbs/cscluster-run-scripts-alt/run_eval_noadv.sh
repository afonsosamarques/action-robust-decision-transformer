$ -S /bin/bash

#$ -N ardt-simplest-eval-noadv
#$ -l gpu=1
#$ -l tmem=4G
#$ -l h_rt=6:00:00

#$ -wd /home/amarques/action-robust-decision-transformer/codebase

source /share/apps/source_files/python/python-3.9.5.source
source /share/apps/source_files/cuda/cuda-11.2.source
export PATH=$PATH:/share/apps/git-lfs-2.11.0/bin/

source /home/amarques/envs/ardt-env/bin/activate

nvidia-smi
python3 -m evaluation_protocol.evaluate --config_name ardt-simplest-eval-noadv
