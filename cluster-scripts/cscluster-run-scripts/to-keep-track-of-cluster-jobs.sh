#!/bin/bash

script_name="test.py"
version=0
result_file="results.txt"

# Create the file, if necessary
touch $result_file

# Determine a unique name for the script, adding a version if necessary
unique_script_name=$script_name
while grep -q "^$unique_script_name" $result_file; do
  version=$((version + 1))
  unique_script_name="${script_name}_v${version}"
done

# Run the Python script
python $script_name

# Check the result and write to the file
if [ $? -eq 0 ]; then
  echo "$unique_script_name: SUCCESS" >> $result_file
else
  echo "$unique_script_name: FAILURE" >> $result_file
fi
