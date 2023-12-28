
for i in $(seq 1 30);
do
    # Start receiver in the background
    ./CUPTI_receiver_covert 1 0 1 nvlink_total_data_received 1000 > covert_receiver_$i.csv &
    # Wait 
    sleep 1
    ./CUPTI_sender 0 1 10000 10 > covert_sender_$i.csv
    
done


