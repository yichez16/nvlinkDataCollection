###############MLP##########################################

###############Input##########################################

# Start profiler in the background
sudo ./CUPTI_receiver 0 1 0 pcie_total_data_received > case_study/mlp_1024.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python Newmlp_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

###############Layer1##########################################

# Start profiler in the background
sudo ./conv_100 0 > case_study/mlp_layer1.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python Newmlp_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

###############Layer2##########################################

# Start profiler in the background
sudo ./conv_100 1 > case_study/mlp_layer2.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python Newmlp_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

###############Layer3##########################################

# Start profiler in the background
sudo ./conv_100 3 > case_study/mlp_layer3.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python Newmlp_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2


###############Layer4##########################################

# Start profiler in the background
sudo ./conv_100 2 > case_study/mlp_layer4.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python Newmlp_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2


###############nvlink##########################################

# Start profiler in the background
sudo ./CUPTI_receiver 1 0 1 nvlink_total_data_received > case_study/mlp_nvlink_1.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python Newmlp_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2


###############nvlink##########################################

# Start profiler in the background
sudo ./CUPTI_receiver 1 0 3 nvlink_total_data_received > case_study/mlp_nvlink_2.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python Newmlp_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

###############nvlink##########################################

# Start profiler in the background
sudo ./CUPTI_receiver 1 0 2 nvlink_total_data_received > case_study/mlp_nvlink_3.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching mlp."
python Newmlp_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./CUPTI_receiver"

# Wait for 2 seconds
sleep 2

