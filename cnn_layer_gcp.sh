# Start profiler in the background
sudo ./conv_100 0 > cnn_layer_0.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 512

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./conv_100 1 > cnn_layer_1.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 512

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./conv_100 2 > cnn_layer_2.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 512

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./conv_100 3 > cnn_layer_3.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 512

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################


# Start profiler in the background
sudo ./conv_100 4 > cnn_layer_4.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 512

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./conv_100 5 > cnn_layer_5.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 512

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################


# Start profiler in the background
sudo ./conv_100 6 > cnn_layer_6.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 512

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################

# Start profiler in the background
sudo ./conv_100 7 > cnn_layer_7.csv &

# Wait for 2 seconds
sleep 2

# Start mlp training
echo "Launching."
python cnn_train_mnist.py 512

# Wait for 2 seconds
sleep 2

# kill profiler
sudo pkill -f "./conv_100"

# Wait for 2 seconds
sleep 2

################################################################################################


