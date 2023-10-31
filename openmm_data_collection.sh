#!/bin/bash

# Start program A in the background
./CUPTI_receiver 0 1 0 > rf.csv &


# Wait for 5 seconds
sleep 5

# Launch program B ten times with a 5-second gap in between
for i in {1..10}
do
    python ~/openmm/examples/benchmark.py --platform CUDA --test rf --device 0,1  
    &
    sleep 5
done

# Wait for all instances of program B to finish
wait

# Find and kill program A
pkill ./CUPTI_receiver 0 1 0 > rf.csv
