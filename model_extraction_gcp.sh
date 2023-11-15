

# Start program A in the background
sudo ./CUPTI_receiver 0 1 0 > cnn_layer1.csv &

# Wait for 5 seconds
sleep 2

sudo ./CUPTI_receiver 0 1 2 > cnn_layer2.csv &

# Wait for 5 seconds
sleep 2

sudo ./CUPTI_receiver 0 1 3 > cnn_layer3.csv &

# Wait for 5 seconds
sleep 2


echo "Launching CNN models."
python train_mnist.py
sleep 2

sudo pkill -f "./CUPTI_receiver"
