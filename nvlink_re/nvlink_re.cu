/*
* Author: Yicheng Zhang
* Association: UC Riverside
* Date: Sep 10th, 2023
*
* Description: 
* Description: Reverse engineering for nvlink packet/data flit
* Idea of program: send a small data packet from gpu A to gpu B. Use nvprof to profile nvlink transactions. 
* Candiate perf ctrs: nvlink_total_data_transmitted,nvlink_total_data_received,nvlink_overhead_data_transmitted,nvlink_overhead_data_received,nvlink_total_response_data_received,nvlink_user_response_data_received,nvlink_total_write_data_transmitted,nvlink_user_data_transmitted,nvlink_user_data_received,nvlink_user_write_data_transmitted
* Nvprof commands: nvprof  --profile-from-start off --devices 0 --aggregate-mode off --csv --log-file "file_name".csv --event-collection-mode continuous -m "ctr_name" ./nvlink_re size_to_transfer
*/

#include <vector>
#include <cuda_profiler_api.h> // For cudaProfilerStart() and cudaProfilerStop()
#include <cstdio>
#include <string>
#include <thrust/device_vector.h>
#include <fstream>
#include <cupti_profiler.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include "kernel.cu"
#include <iostream>
#include <chrono>
#include <thread>




 
int main(int argc, char **argv) {

    using namespace std;
    int local = 0; 
    int remote = 1;
    int sizeElement = 64;
    int *h_local, *h_remote;
    int *d_local, *d_remote;
    

    local = atoi(argv[1]);
    remote = atoi(argv[2]);
    sizeElement = atoi(argv[3]);

    // printf("%d\n", sizeElement);

    size_t size = sizeElement * sizeof(int);

    // Allocate input vectors in host memory
    h_local = (int*)malloc(size);
    h_remote = (int*)malloc(size);

    // Initialize input vectors, local sets to be 1, remote set to be 100
    initVec(h_local, sizeElement, 1);
    initVec(h_remote, sizeElement, 100);

    // local GPU contains d_local
    cudaSetDevice(local);
    cudaMalloc((void**)&d_local, size);  

    // remote GPU contains d_remote 
    cudaSetDevice(remote);
    cudaMalloc((void**)&d_remote, size);

    // make sure nvlink connection exists between src and det device.
    cudaSetDevice(remote); // Set local device to be used for GPU executions.
    cudaDeviceEnablePeerAccess(local, 0);  // Enables direct access to memory allocations on a peer device.
    cudaSetDevice(local); // Set local device to be used for GPU executions.
    cudaDeviceEnablePeerAccess(remote, 0);  // Enables direct access to memory allocations on a peer device.


    // Copy vector local from host memory to device memory
    cudaMemcpy(d_local, h_local, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Copy vector remote from host memory to device memory
    cudaMemcpy(d_remote, h_remote, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    
    int blockSize = 1;
    int gridSize = (sizeElement + blockSize - 1) / blockSize;



    cudaProfilerStart();
    // Use kernel to force nvlink transaction
    for(int i = 0; i < 100; i++){


        // kernel execution 
        test_nvlink <<<blockSize, gridSize>>>(d_remote, d_local, sizeElement); 
        // cudaDeviceSynchronize();


    }
    cudaProfilerStop();

    // // Use cudaMemcpyPeer api to launch nvlink transaction
    // for(int i = 0; i < 1000; i++){


    //     // kernel execution 
    //     cudaMemcpyPeer(d_local, local, d_remote, remote, size); // copy data from remote to local
    //     cudaDeviceSynchronize();


    // }


    
 


    // Copy back to host memory 
    cudaMemcpy(h_remote, d_remote, size, cudaMemcpyDeviceToHost); 
    cudaMemcpy(h_local, d_local, size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
   
    double mb = sizeElement * sizeof(int) / (double)1e6;
    printf("Size of data transfer (MB): %f\n", mb);
    printf("Vector V_local (original value = 1): %d\n",h_local[sizeElement-1]);
    printf("Vector V_remote (original value = 100): %d\n",h_remote[sizeElement-1]);




    cudaFree(d_local);
    cudaFree(d_remote);
    free(h_local);
    free(h_remote);
    

    // exit(EXIT_SUCCESS);
 }
 