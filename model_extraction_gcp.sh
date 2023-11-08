

# Start program A in the background
sudo ./CUPTI_receiver 0 1 0 >> rf.csv &

# Wait for 5 seconds
sleep 2

sudo ./CUPTI_receiver 2 0 2 >> rf.csv &

# Wait for 5 seconds
sleep 2

sudo ./CUPTI_receiver 3 2 3 >> rf.csv &

# Wait for 5 seconds
sleep 2
