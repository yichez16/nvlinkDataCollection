#!/bin/bash

# Collect nvlink side-channel leakage of 8 openmm benchmarks:
# rf,pme,apoa1rf,apoa1pme,apoa1ljpme,amber20-dhfr,amber20-cellulose

# Start program A in the background
sudo ./CUPTI_receiver 0 1 0 > rf.csv &

# Wait for 2 seconds
sleep 2

for i in $(seq 1 2);
do
    # Run the second command in the foreground (concurrently with the first command)
    python benchmark.py --platform CUDA --test rf --device 0,1 
    # Sleep for 2 seconds
    sleep 2
done

# Find and kill program A
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

##########################################################






