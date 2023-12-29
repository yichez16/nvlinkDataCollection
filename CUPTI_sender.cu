/*
* Author: Yicheng Zhang
* Association: UC Riverside
* Date: Oct 3, 2023
*
* Description: 
* Description: Sender for covert-channel attack
* Idea of program: To convey bit "1", we use CUDA driver op (cudaMemcpyPeer) to force memcpy from remote gpu to local gpu.
* To convey bit "0", we use std::this_thread::sleep_for(std::chrono::microseconds(1000)) to block the execution of the current thread.
* The value of d_local, d_remote vector are set to be 1 and 100.
* Four input values for this program: local gpu ID, remote gpu ID, size of data transfer and sleep time (us).
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
    int time2sleep = 1000;
    int *h_local, *h_remote;
    int *d_local, *d_remote;
    
    struct timeval ts,te, te1, te2, te3 ;

    local = atoi(argv[1]);
    remote = atoi(argv[2]);
    sizeElement = atoi(argv[3]);
    time2sleep = atoi(argv[4]);

    // printf("%d\n", sizeElement);

    size_t size = sizeElement * sizeof(int);

    // set up profiler
    cudaSetDevice(local);
    CUdevice device;
    DRIVER_API_CALL(cuInit(0));  
	// Initialize the CUDA driver API Initializes the driver API and must be called before any other function from the driver API in the current process. Currently, the Flags parameter must be 0. If cuInit() has not been called, any function from the driver API will return CUDA_ERROR_NOT_INITIALIZED.
	DRIVER_API_CALL(cuDeviceGet(&device, local));
	// Returns a handle to a compute device.

    // define ctrs to profile
	vector<string> event_names {        
               
	};
	vector<string> metric_names {
    // ctr_num
	// "l2_read_transactions",// works
	//"nvlink_data_receive_efficiency",
	// "nvlink_data_transmission_efficiency",
	//"nvlink_overhead_data_received",
	//"nvlink_overhead_data_transmitted",
	//"nvlink_receive_throughput",
	"nvlink_total_data_received",// works
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
    CUcontext context;
	cuCtxCreate(&context, 0, local); // context is created on device # profile

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


    std::this_thread::sleep_for(std::chrono::seconds(2));   // wait for synchronization

    // synchronization: trojan send data through nvlink and let spy know ready to send

    // for(int j = 0; j < 1; j++){
    //     cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);

            
    //     p->start();
    //     gettimeofday(&ts,NULL);
    //     // cudaMemcpyPeer(d_local, local, d_remote, remote, size); // copy data from remote to local

    //     test_nvlink <<<gridSize, blockSize>>>(d_remote, d_local, 100000); // 56 SMs, 4*32 =  128 threads  (src, det, numElements)  force to transfer data from remote to local.
    //     p->stop();
    //     cudaDeviceSynchronize();

    //     gettimeofday(&te,NULL);
    //     // p->print_event_values(std::cout,ts,te);
    //     p->print_metric_values(std::cout,ts,te);

    //     printf("\n"); 
    //     free(p);

    // }
    

    
    gettimeofday(&ts, NULL);  
    std::cout   
    << ts.tv_sec*1000000 + ts.tv_usec
    // << ","
    // << te.tv_sec*1000000 + te.tv_usec
    ;
    // synchronization 1000 consecutive "1"
    printf("\n");

    cudaDeviceSynchronize();

    for(int i = 0; i < 10000; i++){

        // Start record time

        // kernel execution
        test_nvlink <<<blockSize, gridSize>>>(d_remote, d_local, sizeElement); 
        // std::this_thread::sleep_for(std::chrono::microseconds(time2sleep)); 
        cudaDeviceSynchronize();

        // test_nvlink <<<blockSize, gridSize>>>(d_remote, d_local, sizeElement); 
        // // std::this_thread::sleep_for(std::chrono::microseconds(10)); 
        // cudaDeviceSynchronize();

        // std::this_thread::sleep_for(std::chrono::microseconds(1));       
        // gettimeofday(&te, NULL);  

        
        // test_nvlink <<<gridSize, blockSize>>>(d_remote, d_local, sizeElement); 
        // std::this_thread::sleep_for(std::chrono::microseconds(1));       
        // gettimeofday(&te1, NULL);  
        // cudaDeviceSynchronize();



        std::this_thread::sleep_for(std::chrono::microseconds(time2sleep)); // Sleep for 1 millisecond (1000 microseconds)
        
        // std::this_thread::sleep_for(std::chrono::microseconds(time2sleep)); // Sleep for 1 millisecond (1000 microseconds)

        // gettimeofday(&te2, NULL);  

        // std::this_thread::sleep_for(std::chrono::microseconds(time2sleep)); // Sleep for 1 millisecond (1000 microseconds)

        // gettimeofday(&te3, NULL);  





        // // Print out start and stop time
        // std::cout   << size
        // << "," 
        // << (te.tv_sec - ts.tv_sec) * 1000000 + (te.tv_usec - ts.tv_usec)
        // << "," 
        // << (te1.tv_sec - te.tv_sec) * 1000000 + (te1.tv_usec - te.tv_usec)
        // << "," 
        // << (te2.tv_sec - te1.tv_sec) * 1000000 + (te2.tv_usec - te1.tv_usec)
        // << "," 
        // << (te3.tv_sec - te2.tv_sec) * 1000000 + (te3.tv_usec - te2.tv_usec)
        // ;
        // printf("\n"); 

        cudaDeviceSynchronize();


    }
    gettimeofday(&te, NULL);  
    std::cout   
    << te.tv_sec*1000000 + te.tv_usec
    // << ","
    // << te.tv_sec*1000000 + te.tv_usec
    ;
    printf("\n"); 


    // // Emd of communication
    // for(int j = 0; j < 1; j++){
    //     cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);

            
    //     p->start();
    //     gettimeofday(&ts,NULL);
    //     cudaMemcpyPeer(d_local, local, d_remote, remote, size); // copy data from remote to local

    //     test_nvlink <<<gridSize, blockSize>>>(d_remote, d_local, 100000); // 56 SMs, 4*32 =  128 threads  (src, det, numElements)  force to transfer data from remote to local.
    //     p->stop();
    //     cudaDeviceSynchronize();

    //     gettimeofday(&te,NULL);
    //     // p->print_event_values(std::cout,ts,te);
    //     p->print_metric_values(std::cout,ts,te);

    //     printf("\n"); 
    //     free(p);

    // }    


 


    // Copy back to host memory 
    cudaMemcpy(h_remote, d_remote, size, cudaMemcpyDeviceToHost); 
    cudaMemcpy(h_local, d_local, size, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
   
    double mb = sizeElement * sizeof(int) / (double)1e6;
    // printf("Size of data transfer (MB): %f\n", mb);
    // printf("Vector V_local (original value = 1): %d\n",h_local[sizeElement-1]);
    // printf("Vector V_remote (original value = 100): %d\n",h_remote[sizeElement-1]);




    cudaFree(d_local);
    cudaFree(d_remote);
    free(h_local);
    free(h_remote);
    

    // exit(EXIT_SUCCESS);
 }
 