# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_128.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python mlp_train_mnist.py 128

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_256.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python mlp_train_mnist.py 256

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################


# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_512.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python mlp_train_mnist.py 512

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

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

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_2048.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python mlp_train_mnist.py 2048


# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_4096.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python mlp_train_mnist.py 4096


# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > mlp_input_received_8192.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python mlp_train_mnist.py 8192


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

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > cnn_input_received_1024.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python cnn_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > cnn_input_received_2048.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python cnn_train_mnist.py 2048


# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > cnn_input_received_2048.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python cnn_train_mnist.py 2048


# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > cnn_input_received_4096.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python cnn_train_mnist.py 4096


# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################

################################################################################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > cnn_input_received_8192.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python cnn_train_mnist.py 8192


# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

################################################################################################