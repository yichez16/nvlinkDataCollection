#!/bin/bash

# Test data transfer size from 1 to 1000 
for i in {1..1000}
do
  ./CUPTI_receiver_covert 1 0 1 $i 588445102575399 >> nvlink_re/test_1.csv
done

echo "Processing completed."
