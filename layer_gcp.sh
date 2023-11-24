# Start profiler in the background
sudo ./conv_100 0 > layer_conv_1.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./conv_100 1 > layer_pooling_1.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./conv_100 2 >layer_conv_2.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./conv_100 3 > layer_pooling_2.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################


# Start profiler in the background
sudo ./conv_100 4 > layer_conv_3.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./conv_100 5 > layer_fc_1.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################


# Start profiler in the background
sudo ./conv_100 6 > layer_fc_2.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./conv_100 7 > layer_fc_3.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 1024

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################


