#$ -S /bin/bash

#$ -N evaluation_batchenv_robust_hopper
#$ -l gpu=True
#$ -l tmem=40G
#$ -l h_rt=12:00:00

#$ -wd /home/amarques/action-robust-decision-transformer/codebase

export PATH=$PATH:/share/apps/git-lfs-2.11.0/bin/
source /share/apps/source_files/python/python-3.9.5.source
source /home/amarques/envs/ardt-env/bin/activate

# initialise variables
version=0
result_file="results.txt"
lock_file="results.lock"
config_name=evaluation_batchenv_robust_hopper

# create the files, if necessary
touch $result_file
touch $lock_file

# determine a unique name for the script, adding a version if necessary
exec 200>$lock_file
flock -e 200

unique_script_name=$config_name
while grep -q "^$unique_script_name:" $result_file; do
  version=$((version + 1))
  unique_script_name="${unique_script_name}_v${version}"
done

flock -u 200
exec 200>&-

# run the code
nvidia-smi
python3 -m evaluation_protocol.evaluate --config_name $config_name
result=$?

# check the result and write to the file
exec 200>$lock_file
flock -e 200

if [ $result -eq 0 ]; then
  echo "$unique_script_name: SUCCESS" >> $result_file
else
  echo "$unique_script_name: FAILURE" >> $result_file
fi

flock -u 200
exec 200>&-
