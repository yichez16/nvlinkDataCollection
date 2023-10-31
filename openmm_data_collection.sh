#!/bin/bash

# Collect nvlink side-channel leakage of 8 openmm benchmarks:
# rf,pme,apoa1rf,apoa1pme,apoa1ljpme,amber20-dhfr,amber20-cellulose

# Start program A in the background
sudo ./CUPTI_receiver 0 1 0 > rf.csv &

# Wait for 5 seconds
sleep 5

for i in $(seq 1 10);
do
    echo "Launching benchmark for the $i time"
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


# Start program A in the background
sudo ./CUPTI_receiver 0 1 0 > pme.csv &

# Wait for 5 seconds
sleep 5

for i in $(seq 1 10);
do
    echo "Launching benchmark for the $i time"
    python benchmark.py --platform CUDA --test pme --device 0,1  
    sleep 2
done

# # Wait for all instances of program B to finish
# wait

# Find and kill program A
sudo pkill -f "./CUPTI_receiver"

# Wait for 5 seconds
sleep 5

##########################################################

# Start program A in the background
sudo ./CUPTI_receiver 0 1 0 > apoa1rf.csv &

# Wait for 5 seconds
sleep 5

for i in $(seq 1 10);
do
    echo "Launching benchmark for the $i time"
    python benchmark.py --platform CUDA --test apoa1rf --device 0,1  
    sleep 2
done

# # Wait for all instances of program B to finish
# wait

# Find and kill program A
sudo pkill -f "./CUPTI_receiver"

# Wait for 5 seconds
sleep 5

##########################################################

# Start program A in the background
sudo ./CUPTI_receiver 0 1 0 > apoa1pme.csv &

# Wait for 5 seconds
sleep 5

for i in $(seq 1 10);
do
    echo "Launching benchmark for the $i time"
    python benchmark.py --platform CUDA --test apoa1pme --device 0,1  
    sleep 2
done

# # Wait for all instances of program B to finish
# wait

# Find and kill program A
sudo pkill -f "./CUPTI_receiver"

# Wait for 5 seconds
sleep 5

##########################################################

# Start program A in the background
sudo ./CUPTI_receiver 0 1 0 > apoa1ljpme.csv &

# Wait for 5 seconds
sleep 5

for i in $(seq 1 10);
do
    echo "Launching benchmark for the $i time"
    python benchmark.py --platform CUDA --test apoa1ljpme --device 0,1  
    sleep 2
done

# # Wait for all instances of program B to finish
# wait

# Find and kill program A
sudo pkill -f "./CUPTI_receiver"

# Wait for 5 seconds
sleep 5

##########################################################

# Start program A in the background
sudo ./CUPTI_receiver 0 1 0 > amber20-dhfr.csv &

# Wait for 5 seconds
sleep 5

for i in $(seq 1 10);
do
    echo "Launching benchmark for the $i time"
    python benchmark.py --platform CUDA --test amber20-dhfr --device 0,1  
    sleep 2
done

# # Wait for all instances of program B to finish
# wait

# Find and kill program A
sudo pkill -f "./CUPTI_receiver"

# Wait for 5 seconds
sleep 5

##########################################################
# Start program A in the background
sudo ./CUPTI_receiver 0 1 0 > amber20-cellulose.csv &

# Wait for 5 seconds
sleep 5

for i in $(seq 1 10);
do
    echo "Launching benchmark for the $i time"
    python benchmark.py --platform CUDA --test amber20-cellulose --device 0,1  
    sleep 2
done

# # Wait for all instances of program B to finish
# wait

# Find and kill program A
sudo pkill -f "./CUPTI_receiver"

# Wait for 5 seconds
sleep 5

##########################################################
