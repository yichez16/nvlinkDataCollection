#!/bin/bash

# Collect nvlink side-channel leakage of 8 openmm benchmarks:
# rf,pme,apoa1rf,apoa1pme,apoa1ljpme,amber20-dhfr,amber20-cellulose

# Start program A in the background
sudo ./CUPTI_receiver 0 1 0 > rf.csv &
pidA=$!

# Wait for 5 seconds
sleep 5

# Launch program B ten times with a 5-second gap in between
for i in {1..5}
do
    # python benchmark.py --platform CUDA --test rf --device 0,1 &
    sleep 1
done

# Wait for all instances of program B to finish
wait

# Find and kill program A
kill $pidA


# Wait for 5 seconds
sleep 5

##########################################################


# # Start program A in the background
# sudo ./CUPTI_receiver 0 1 0 > pme.csv &


# # Wait for 5 seconds
# sleep 5

# # Launch program B ten times with a 5-second gap in between
# for i in {1..10}
# do
#     python benchmark.py --platform CUDA --test pme --device 0,1 &
#     sleep 5
# done

# # Wait for all instances of program B to finish
# wait

# # Find and kill program A
# pkill ./CUPTI_receiver 0 1 0 > pme.csv


# # Wait for 5 seconds
# sleep 5

# ##########################################################

# # Start program A in the background
# sudo ./CUPTI_receiver 0 1 0 > apoa1rf.csv &


# # Wait for 5 seconds
# sleep 5

# # Launch program B ten times with a 5-second gap in between
# for i in {1..10}
# do
#     python benchmark.py --platform CUDA --test apoa1rf --device 0,1 &
#     sleep 5
# done

# # Wait for all instances of program B to finish
# wait

# # Find and kill program A
# pkill ./CUPTI_receiver 0 1 0 > apoa1rf.csv


# # Wait for 5 seconds
# sleep 5

# ##########################################################

# # Start program A in the background
# sudo ./CUPTI_receiver 0 1 0 > apoa1pme.csv &


# # Wait for 5 seconds
# sleep 5

# # Launch program B ten times with a 5-second gap in between
# for i in {1..10}
# do
#     python benchmark.py --platform CUDA --test apoa1pme --device 0,1 &
#     sleep 5
# done

# # Wait for all instances of program B to finish
# wait

# # Find and kill program A
# pkill ./CUPTI_receiver 0 1 0 > apoa1pme.csv


# # Wait for 5 seconds
# sleep 5

# ##########################################################

# # Start program A in the background
# sudo ./CUPTI_receiver 0 1 0 > apoa1ljpme.csv &


# # Wait for 5 seconds
# sleep 5

# # Launch program B ten times with a 5-second gap in between
# for i in {1..10}
# do
#     python benchmark.py --platform CUDA --test apoa1ljpme --device 0,1 &
#     sleep 5
# done

# # Wait for all instances of program B to finish
# wait

# # Find and kill program A
# pkill ./CUPTI_receiver 0 1 0 > apoa1ljpme.csv


# # Wait for 5 seconds
# sleep 5

# ##########################################################

# # Start program A in the background
# sudo ./CUPTI_receiver 0 1 0 > amber20-dhfr.csv &


# # Wait for 5 seconds
# sleep 5

# # Launch program B ten times with a 5-second gap in between
# for i in {1..10}
# do
#     python benchmark.py --platform CUDA --test amber20-dhfr --device 0,1 &
#     sleep 5
# done

# # Wait for all instances of program B to finish
# wait

# # Find and kill program A
# pkill ./CUPTI_receiver 0 1 0 > amber20-dhfr.csv


# # Wait for 5 seconds
# sleep 5

# ##########################################################

# # Start program A in the background
# sudo ./CUPTI_receiver 0 1 0 > amber20-cellulose.csv &


# # Wait for 5 seconds
# sleep 5

# # Launch program B ten times with a 5-second gap in between
# for i in {1..10}
# do
#     python benchmark.py --platform CUDA --test amber20-cellulose --device 0,1 &
#     sleep 5
# done

# # Wait for all instances of program B to finish
# wait

# # Find and kill program A
# pkill ./CUPTI_receiver 0 1 0 > amber20-cellulose.csv


# # Wait for 5 seconds
# sleep 5

# ##########################################################


