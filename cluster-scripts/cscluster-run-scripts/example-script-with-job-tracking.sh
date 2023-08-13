#$ -S /bin/bash

#$ -N ardt_full_special
#$ -l gpu=true
#$ -l h_vmem=20G
#$ -l h_rt=8:00:00

#$ -wd /home/amarques/action-robust-decision-transformer/codebase

source /share/apps/source_files/python/python-3.9.5.source
export PATH=$PATH:/share/apps/git-lfs-2.11.0/bin/
source /home/amarques/envs/ardt-env/bin/activate

version=0
result_file="results.txt"
lock_file="results.lock"
script_name="ardt_full_special"

# Create the file, if necessary
touch $result_file
touch $lock_file

# Determine a unique name for the script, adding a version if necessary
exec 200>$lock_file
flock -e 200

unique_script_name=$script_name
while grep -q "^$unique_script_name:" $result_file; do
  version=$((version + 1))
  unique_script_name="${script_name}_v${version}"
done

flock -u 200
exec 200>&-


nvidia-smi
python3 -m ardt.pipeline --config_name ardt_full_special
result=$?


# Check the result and write to the file
exec 200>$lock_file
flock -e 200

if [ $result -eq 0 ]; then
  echo "$unique_script_name: SUCCESS" >> $result_file
else
  echo "$unique_script_name: FAILURE" >> $result_file
fi

flock -u 200
exec 200>&-