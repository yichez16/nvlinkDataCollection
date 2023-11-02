#!/bin/bash

# Collect nvlink side-channel leakage of 5 nccl operations:
# allreduce broadcast reduce allGather reduceScatter

# Run the first command in the background
sudo ./CUPTI_receiver 0 1 0 >> allreduce.csv &

# Sleep for 5 seconds
sleep 5

for i in $(seq 1 5);
do
    # Run the second command in the foreground (concurrently with the first command)
    echo "Launching op for the $i time"
    ./nccl_test 0 10

    # Sleep for 2 seconds
    sleep 2
done


# Find and kill the first command 
pkill -f "./CUPTI_receiver"

sleep 5

##########################################################

# Run the first command in the background
sudo ./CUPTI_receiver 0 1 0 >> broadcast.csv &

# Sleep for 5 seconds
sleep 5

for i in $(seq 1 5);
do
    # Run the second command in the foreground (concurrently with the first command)
    echo "Launching op for the $i time"
    ./nccl_test 1 10

    # Sleep for 2 seconds
    sleep 2
done


# Find and kill the first command 
pkill -f "./CUPTI_receiver"

sleep 5

##########################################################

# Run the first command in the background
sudo ./CUPTI_receiver 0 1 0 >> reduce.csv &

# Sleep for 5 seconds
sleep 5

for i in $(seq 1 5);
do
    # Run the second command in the foreground (concurrently with the first command)
    echo "Launching op for the $i time"
    ./nccl_test 2 10
    # Sleep for 2 seconds
    sleep 2
done


# Find and kill the first command 
pkill -f "./CUPTI_receiver"

sleep 5

##########################################################

# Run the first command in the background
sudo ./CUPTI_receiver 0 1 0 >> allGather.csv &

# Sleep for 5 seconds
sleep 5

for i in $(seq 1 5);
do
    # Run the second command in the foreground (concurrently with the first command)
    echo "Launching op for the $i time"
    ./nccl_test 3 10

    # Sleep for 2 seconds
    sleep 2
done


# Find and kill the first command 
pkill -f "./CUPTI_receiver"

sleep 5

##########################################################

# Run the first command in the background
sudo ./CUPTI_receiver 0 1 0 >> reduceScatter.csv &

# Sleep for 5 seconds
sleep 5

for i in $(seq 1 5);
do
    # Run the second command in the foreground (concurrently with the first command)
    echo "Launching op for the $i time"
    ./nccl_test 4 10

    # Sleep for 2 seconds
    sleep 2
done


# Find and kill the first command 
pkill -f "./CUPTI_receiver"

sleep 5

##########################################################
