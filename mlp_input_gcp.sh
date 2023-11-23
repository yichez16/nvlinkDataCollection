# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_64.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python mlp_train_mnist.py 64

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

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


# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_192.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python mlp_train_mnist.py 192

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

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

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_320.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python mlp_train_mnist.py 320


# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

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

###############################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_448.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python mlp_train_mnist.py 448


# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

###############################################################################################

# Start profiler in the background
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







# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > cnn_input_received_64.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python cnn_train_mnist.py 64

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > cnn_input_received_128.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python cnn_train_mnist.py 128

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################


# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > cnn_input_received_192.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python cnn_train_mnist.py 192

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > cnn_input_received_256.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python cnn_train_mnist.py 256

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > cnn_input_received_320.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python cnn_train_mnist.py 320


# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > cnn_input_received_384.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python cnn_train_mnist.py 384


# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > cnn_input_received_448.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python cnn_train_mnist.py 448


# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > cnn_input_received_512.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python cnn_train_mnist.py 512


# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################