# for i in $(seq 1 1000);
# do
#     nvprof  --profile-from-start off --devices 0 --aggregate-mode off --csv --log-file ./transmitted/$i.csv --event-collection-mode kernel -m nvlink_total_data_transmitted,nvlink_total_data_received ./nvlink_re 0 1 $i 

#     # Sleep for 2 seconds
#     sleep 1
# done

for i in $(seq 1 1000);
do
    nvprof  --profile-from-start off --devices 0 --aggregate-mode off --csv --log-file ./throughput/$i.csv --event-collection-mode kernel -m nvlink_transmit_throughput,nvlink_receive_throughput ./nvlink_re 0 1 $i 

    # Sleep for 2 seconds
    sleep 1
done
