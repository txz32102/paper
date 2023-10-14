#!/bin/bash

# Input and output file paths
input_file="/home/musong/python/stu/data/raw_data/missing_id.txt"
output_file="/home/musong/python/stu/data/raw_data/result.csv"

# Create an empty result CSV file with headers
echo "ID,sequence" > "$output_file"

# Loop through each ID in the input file
while IFS= read -r id; do
  # Use grep and awk to extract the sequence
  sequence=$(grep -A 1 "$id" /home/musong/python/dataset/swissprot | awk '/^>/ {next} {print}')
  
  # Check if the sequence is empty
  if [ -z "$sequence" ]; then
    sequence="-1"
  fi

  # Write the ID and sequence to the result CSV file
  echo "$id,$sequence" >> "$output_file"
done < "$input_file"
