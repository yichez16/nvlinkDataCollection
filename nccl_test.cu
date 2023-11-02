/*
* Author: Yicheng Zhang
* Association: UC Riverside
* Date: Sep 6, 2023
*
* Description: 
* Idea of nccl_test.cu: test basic nccl collective operations 
Also, add cupti lib to profile nvlink metrics
*/

#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <cuda_profiler_api.h> // For cudaProfilerStart() and cudaProfilerStop()
#include <cupti_profiler.h>


int main(int argc, char **argv)
{
    //managing nDev devices
    int nDev = 2;
    int size = 1000*1000;
    int recvcount, sendcount;
    int devs[nDev] = { 0, 1}; 
    ncclComm_t comms[nDev];

    // choice of nccl operations
    // "0" for ncclALLReduce
    // "1" for ncclBroadcast
    // "2" for ncclReduce
    // "3" for ncclAllGather
    // "4" for ncclReduceScatter
    int choice_nccl = atoi(argv[1]);

    // size of buffer
    int mbSize = atoi(argv[2]);
    size *= mbSize;
    
    recvcount = size/nDev;
    sendcount = size/nDev;



    //allocating and initializing device buffers
    int** sendbuff = (int**)malloc(nDev * sizeof(int*));
    int** recvbuff = (int**)malloc(nDev * sizeof(int*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

    // allocate a sendbuff (all 1) and a recvbuff (all 0) on each device
    // also create a cudaStream on each device
    for (int i = 0; i < nDev; ++i) {
        cudaSetDevice(i);
        cudaMalloc(sendbuff + i, size * sizeof(int));
        cudaMalloc(recvbuff + i, size * sizeof(int));
        cudaMemset(sendbuff[i], 1, size * sizeof(int));
        cudaMemset(recvbuff[i], 0, size * sizeof(int));
        cudaStreamCreate(s+i);
    }

    cudaSetDevice(0);
    // initializing NCCL
    // ncclCommInitAll is equivalent to calling a combination of ncclGetUniqueId and ncclCommInitRank.
    ncclCommInitAll(comms, nDev, devs); 


    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
    ncclGroupStart();

    // Start nccl opearions
    if (choice_nccl == 0){    
        // allReduce, op is sum
        for (int i = 0; i < nDev; ++i){
            ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclInt, ncclSum,
            comms[i], s[i]);
        }
        printf("NCCL op: allReduce on %d GPUs, Size is %d MB.\n", nDev, mbSize);
    }
    else if (choice_nccl == 1){    
        //  broadcast, root set to be 0
        for (int i = 0; i < nDev; ++i) {       
            ncclBcast(sendbuff[i], size, ncclInt, 0,
            comms[i], s[i]);
        }
        printf("NCCL op: broadcast on %d GPUs, Size is %d MB.\n", nDev, mbSize);
    }
    else if (choice_nccl == 2){
        //   reduce, root set to be 0
        for (int i = 0; i < nDev; ++i){
            ncclReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclInt, ncclSum, 0,
            comms[i], s[i]);
        }
        printf("NCCL op: reduce on %d GPUs, Size is %d MB.\n", nDev, mbSize);
    }
    else if (choice_nccl == 3){    
        for (int i = 0; i < nDev; ++i){
        // allGather 
            ncclAllGather((const void*)sendbuff[i], (void*)recvbuff[i], sendcount, ncclInt,
            comms[i], s[i]);
        }   
        printf("NCCL op: allGather on %d GPUs, Size is %d MB.\n", nDev, mbSize);
    }
    else if (choice_nccl == 4){    
        for (int i = 0; i < nDev; ++i){
        // reduceScatter, recvcount = size/numDev
            ncclReduceScatter((const void*)sendbuff[i], (void*)recvbuff[i], recvcount, ncclInt, ncclSum,
            comms[i], s[i]);
        }
        printf("NCCL op: reduceScatter on %d GPUs, Size is %d MB.\n", nDev, mbSize);
    }
    else {
        printf("Enter correct nccl operation");
    }
    // // Stop profiler
    // cudaProfilerStop(); 
        
    ncclGroupEnd();


    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(s[i]);
    }


    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        cudaSetDevice(i);
        cudaFree(sendbuff[i]);
        cudaFree(recvbuff[i]);
    }


    //finalizing NCCL
    for(int i = 0; i < nDev; ++i)
        ncclCommDestroy(comms[i]);


    
    return 0;
}