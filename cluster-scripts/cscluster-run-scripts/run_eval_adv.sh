$ -S /bin/bash

#$ -N evaluation_agentadv
#$ -l gpu=1
#$ -l tmem=6G
#$ -l h_rt=8:00:00

#$ -wd /home/amarques/action-robust-decision-transformer/codebase

source /share/apps/source_files/python/python-3.9.5.source
source /share/apps/source_files/cuda/cuda-11.4.source
export PATH=$PATH:/share/apps/git-lfs-2.11.0/bin/

source /home/amarques/envs/ardt-env/bin/activate

nvidia-smi
python3 -m evaluation_protocol.evaluate --config_name evaluation_agentadv
