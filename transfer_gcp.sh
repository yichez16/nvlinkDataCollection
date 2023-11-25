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


# Start profiler in the background
sudo ./CUPTI_receiver 1 3 1 nvlink_total_data_received > cnn_nvlink_1.csv &

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
sudo ./CUPTI_receiver 1 3 3 nvlink_total_data_received > cnn_nvlink_2.csv &

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
sudo ./CUPTI_receiver 1 3 2 nvlink_total_data_received > cnn_nvlink_3.csv &

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
sudo ./CUPTI_receiver 4 6 7 nvlink_total_data_received > cnn_nvlink_4.csv &

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
sudo ./CUPTI_receiver 4 6 6 nvlink_total_data_received > cnn_nvlink_5.csv &

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
sudo ./CUPTI_receiver 4 6 4 nvlink_total_data_received > cnn_nvlink_6.csv &

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
sudo ./CUPTI_receiver 4 6 5 nvlink_total_data_received > cnn_nvlink_7.csv &

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
