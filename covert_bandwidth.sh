
for i in $(seq 1 1);
do
    # Start receiver in the background
    ./CUPTI_receiver_covert 1 0 1 nvlink_total_data_received 1000 > covert_receiver_$i.csv &
    # Wait 
    sleep 2
    ./CUPTI_sender 0 1 10000 10 
    # kill receiver
    sudo pkill -f "./CUPTI_receiver_covert"
    # Wait for 2 seconds
    sleep 2
done


