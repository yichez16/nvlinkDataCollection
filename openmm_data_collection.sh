#!/bin/bash

# Collect nvlink side-channel leakage of 8 openmm benchmarks:
# rf,pme,apoa1rf,apoa1pme,apoa1ljpme,amber20-dhfr,amber20-cellulose

# Start program A in the background
sudo ./CUPTI_receiver 0 1 0 > rf.csv &

# Wait for 5 seconds
sleep 5

for i in {1..3}
do
    python benchmark.py --platform CUDA --test rf --device 0,1 
    sleep 2
done

# # Wait for all instances of program B to finish
# wait

# Find and kill program A
sudo pkill -f "./CUPTI_receiver"

# Wait for 5 seconds
sleep 5

##########################################################






