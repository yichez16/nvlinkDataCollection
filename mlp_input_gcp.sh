# # Start profiler in the background
# sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_128.csv &

# # Wait for 2 seconds
# sleep 2

# # Start mlp training
# echo "Launching mlp."
# python mlp_train_mnist.py 128

# # Wait for 2 seconds
# sleep 2

# # kill profiler
# sudo pkill -f "./CUPTI_receiver"

# # Wait for 2 seconds
# sleep 2

# ################################################################################################

# # Start profiler in the background
# sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_256.csv &

# # Wait for 2 seconds
# sleep 2

# # Start mlp training
# echo "Launching mlp."
# python mlp_train_mnist.py 256

# # Wait for 2 seconds
# sleep 2

# # kill profiler
# sudo pkill -f "./CUPTI_receiver"

# # Wait for 2 seconds
# sleep 2

# ################################################################################################


# # Start profiler in the background
# sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_384.csv &

# # Wait for 2 seconds
# sleep 2

# # Start mlp training
# echo "Launching mlp."
# python mlp_train_mnist.py 384

# # Wait for 2 seconds
# sleep 2

# # kill profiler
# sudo pkill -f "./CUPTI_receiver"

# # Wait for 2 seconds
# sleep 2

# ################################################################################################

# # Start profiler in the background
# sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_512.csv &

# # Wait for 2 seconds
# sleep 2

# # Start mlp training
# echo "Launching mlp."
# python mlp_train_mnist.py 512

# # Wait for 2 seconds
# sleep 2

# # kill profiler
# sudo pkill -f "./CUPTI_receiver"

# # Wait for 2 seconds
# sleep 2

# ################################################################################################

# # Start profiler in the background
# sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_640.csv &

# # Wait for 2 seconds
# sleep 2

# # Start mlp training
# echo "Launching mlp."
# python mlp_train_mnist.py 640


# # Wait for 2 seconds
# sleep 2

# # kill profiler
# sudo pkill -f "./CUPTI_receiver"

# # Wait for 2 seconds
# sleep 2

# ################################################################################################

# # Start profiler in the background
# sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_768.csv &

# # Wait for 2 seconds
# sleep 2

# # Start mlp training
# echo "Launching mlp."
# python mlp_train_mnist.py 768


# # Wait for 2 seconds
# sleep 2

# # kill profiler
# sudo pkill -f "./CUPTI_receiver"

# # Wait for 2 seconds
# sleep 2

# ################################################################################################

################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_896.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python mlp_train_mnist.py 896


# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_1024.csv &

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