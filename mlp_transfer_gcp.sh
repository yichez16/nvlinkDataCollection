# Start profiler in the background
sudo ./CUPTI_receiver 1 0 1 nvlink_total_data_received > mlp_nvlink_1.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python mlp_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 1 0 3 nvlink_total_data_received > mlp_nvlink_2.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python mlp_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################


# Start profiler in the background
sudo ./CUPTI_receiver 1 0 2 nvlink_total_data_received > mlp_nvlink_3.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python mlp_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################
