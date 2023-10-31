#!/bin/bash

# Collect nvlink side-channel leakage of 8 openmm benchmarks:
# rf,pme,apoa1rf,apoa1pme,apoa1ljpme,amber20-dhfr,amber20-cellulose

# Start program A in the background
sudo ./CUPTI_receiver 0 1 0 > rf.csv &

# Wait for 2 seconds
sleep 2

# Launch program B ten times with a 5-second gap in between
for i in {1..1}
do
    python benchmark.py --platform CUDA --test rf --device 0,1 &
    sleep 2
done

# Find and kill program A
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

##########################################################




