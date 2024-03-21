#!/bin/bash

# Loop through files 1.blend to 50.blend
for i in {1..50}
do
  # Start CUPTI_receiver in the background and output to CSV
  ./CUPTI_receiver 0 1 1 32 > "${i}.csv" &
  
  # Wait for 2 seconds to ensure it's ready
  sleep 2

  # Run Blender in batch mode for the current .blend file
  blender -b "/home/mrdev/data/${i}.blend" -o "//img/${i}_###" -F PNG -x 1 -f 1..20 -- --cycles-device CUDA --cycles-print-stats

  # Kill the CUPTI_receiver process
  pkill -f "./CUPTI_receiver"

  # Wait for 2 seconds before starting the next iteration
  sleep 2
done

echo "Processing completed."
