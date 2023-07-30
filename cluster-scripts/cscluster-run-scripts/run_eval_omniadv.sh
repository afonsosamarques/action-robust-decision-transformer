$ -S /bin/bash

#$ -N evaluation_agentadv_omni
#$ -l gpu=1
#$ -l tmem=10G
#$ -l h_rt=24:00:00

#$ -wd /home/amarques/action-robust-decision-transformer/codebase

source /share/apps/source_files/python/python-3.9.5.source
export PATH=$PATH:/share/apps/git-lfs-2.11.0/bin/

source /home/amarques/envs/ardt-env/bin/activate

nvidia-smi
python3 -m evaluation_protocol.evaluate --config_name evaluation_batch_agentadv --omni_adv
