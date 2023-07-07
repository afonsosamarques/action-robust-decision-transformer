#!/bin/bash -l

module purge
module load default-modules
module remove mpi compilers
module load python/3.9.10
mkdir ardt-env
cd ardt-env
virtualenv ardt
cd /home/ucabscm
source /home/ucabscm/ardt-env/ardt/bin/activate

pip3 install --upgrade pip
pip3 install -r /home/ucabscm/requirements.txt

cd /home/ucabscm
chmod +x run_experiment.sh
qsub run_experiment.sh
