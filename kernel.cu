#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>
#define BLOCK_SIZE 512
#define BLOCK_NUM_LIMIT 16
#include <cupti_profiler.h>
#include <device_launch_parameters.h>
#include <iostream>

__global__ void 
copyKernel(int* det, int* src, int addressID1, int addressID2)
{
    int tid = threadIdx.x;
    if (tid == 0)
    {
    // clock_t startClock = clock();
    det[addressID1] = src[addressID1];
    // clock_t stopClock = clock();
    // clock_t elapsedTime = stopClock - startClock;
    // printf("ThreadID: %d\n", tid);
    // printf("ElapsedTime: %llu\n", elapsedTime);
    __syncthreads();

    //  startClock = clock();
    // det[addressID2] = src[addressID2];
    //  stopClock = clock();
    //  elapsedTime = stopClock - startClock;
    // printf("ThreadID: %d\n", threadIdx.x);
    // printf("ElapsedTime: %llu\n", elapsedTime);
    }
    else if (tid == 1)
    {
    // clock_t startClock = clock();
    det[addressID2] = src[addressID2];
    // clock_t stopClock = clock();
    // clock_t elapsedTime = stopClock - startClock;
    // printf("ThreadID: %d\n", tid);
    // printf("ElapsedTime: %llu\n", elapsedTime);
    __syncthreads();
    }
    
    // else if (tid == 2)
    // {
    // clock_t startClock = clock();
    // det[addressID2+1000] = src[addressID2+1000];
    // clock_t stopClock = clock();
    // clock_t elapsedTime = stopClock - startClock;
    // printf("ThreadID: %d\n", tid);
    // printf("ElapsedTime: %llu\n", elapsedTime);
    // __syncthreads();
    // }


}

__global__ void 
copyKernel_two(int* det, int* src, int* det1, int* src1, int addressID1, int addressID2)
{
    int tid = threadIdx.x;
    if (tid == 0)
    {
    // clock_t startClock = clock();
    det[addressID1] = src[addressID1];
    det1[addressID1] = src1[addressID1];
    // clock_t stopClock = clock();
    // clock_t elapsedTime = stopClock - startClock;
    // printf("ThreadID: %d\n", tid);
    // printf("ElapsedTime: %llu\n", elapsedTime);
    __syncthreads();

    //  startClock = clock();
    // det[addressID2] = src[addressID2];
    //  stopClock = clock();
    //  elapsedTime = stopClock - startClock;
    // printf("ThreadID: %d\n", threadIdx.x);
    // printf("ElapsedTime: %llu\n", elapsedTime);
    }
    else if (tid == 1)
    {
    // clock_t startClock = clock();
    det[addressID2] = src[addressID2];
    det1[addressID2] = src1[addressID2];
    // clock_t stopClock = clock();
    // clock_t elapsedTime = stopClock - startClock;
    // printf("ThreadID: %d\n", tid);
    // printf("ElapsedTime: %llu\n", elapsedTime);
    __syncthreads();
    }
    
    // else if (tid == 2)
    // {
    // clock_t startClock = clock();
    // det[addressID2+1000] = src[addressID2+1000];
    // clock_t stopClock = clock();
    // clock_t elapsedTime = stopClock - startClock;
    // printf("ThreadID: %d\n", tid);
    // printf("ElapsedTime: %llu\n", elapsedTime);
    // __syncthreads();
    // }


}

__global__ void 
copyKernel_single(int* det, int* src, int addressID1)
{
    int tid = threadIdx.x;
    if (tid == 0)
    {
    // clock_t startClock = clock();
    det[addressID1] = src[addressID1];
    // clock_t stopClock = clock();
    // clock_t elapsedTime = stopClock - startClock;
    // printf("ThreadID: %d\n", tid);
    // printf("ElapsedTime: %llu\n", elapsedTime);
    __syncthreads();

    //  startClock = clock();
    // det[addressID2] = src[addressID2];
    //  stopClock = clock();
    //  elapsedTime = stopClock - startClock;
    // printf("ThreadID: %d\n", threadIdx.x);
    // printf("ElapsedTime: %llu\n", elapsedTime);
    }
}




// Vector kernel with normal addition operations.
__global__ void
vecAdd_nvlink(int *A, int *B, int *C, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // int *cache1, *cache2;
    clock_t start, middle, end;
    double dramTime, nvlinkTime;

    start = clock(); // Start the clock
    for (int i = idx; i < numElements; i++) {
        {
            // C[idx] = A[idx] + B[idx];
            C[i] = A[i] + B[i];
            // cache2[idx] = B[idx];
        }
    }
    middle = clock();


    for (int i = idx; i < numElements; i++) {
        {
            C[i] = A[i] + B[i];
        }
    }
    end = clock(); // end clock

    dramTime = (double) (middle - start);   // dram + nvlink
    nvlinkTime = (double) (end - middle);   // cached time + nvlink
    // nvlinkTime = dramTime - cachedTime;

    printf("Dram acceess,%d,%f\n", idx, dramTime);
    printf("nvlink acceess,%d,%f\n", idx, nvlinkTime);


}

__global__ void
test_nvlink(int *src, int *dst, int numElements){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements){
        dst[idx] = src[idx] * 2;
    }
    unsigned long long delay_cycles = 60000000ULL;
    unsigned long long start = clock64();
    while (clock64() - start < delay_cycles);
    

}

__global__ void 
matMul(float* A, float* B, float* C, int numARows, int numACols, int numBCols) {
    // compute global thread coordinates
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // linearize coordinates for data access
    int offset = row * numBCols + col;

    if ((row < numARows) && (col < numBCols)) {
        float cumSum = 0;
        for (int k = 0; k < numACols; k++) {
            cumSum += A[row*numACols + k] * B[k*numBCols + col];
        }
        C[offset] = cumSum;
    }
}

__global__ void delay_kernel(unsigned long long delay_cycles)
{
    unsigned long long start = clock64();
    while (clock64() - start < delay_cycles);
}


__global__ void
vecAdd(int *A, int *B, int *C, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // for (int i = 0; i < 1000; i++) {
        if (idx < numElements)
        {
        int dummy = 0;

        // Dummy calculations to introduce additional computational workload
        for (int i = 0; i < numElements; i++) {
            // dummy += i;
            atomicAdd(&dummy, i);
        }
            C[idx] = A[idx] + B[idx];

        }
    // }

}

// VectorAdd with memory coalescing 
__global__ void vecAdd_coalescing(int *a, int *b, int *c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }

}

__global__ void vecAdd_coalescing_nvlink(int *a, int *b, int *c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // int *cache1, *cache2;
    clock_t start, middle, end;
    double dramTime, nvlinkTime;

    start = clock(); // Start the clock
    for (int i = tid; i < n; i += stride) {
        c[i] = a[i] + b[i];
        // cache1[i] = a[i] + c[i];
        // cache2[i] = b[i];
    }
    middle = clock();
    

    for (int i = tid; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
    end = clock(); // end clock

    dramTime = (double) (middle - start);   
    nvlinkTime = (double) (end - middle);
    // nvlinkTime = dramTime - cachedTime;

    printf("Dram acceess,%d,%f\n", tid, dramTime);
    printf("nvlink acceess,%d,%f\n", tid, nvlinkTime);

}

// Vector kernel with normal addition operations.
__global__ void atomic_op(int* input, int* bins, int num_elements, int num_bins)
{
	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
	__shared__ unsigned int histo_private[4096];

    for(int i = threadIdx.x; i < num_bins; i += BLOCK_SIZE){
        histo_private[i] = 0;
    } // Initialize all histo_private[]
    __syncthreads();
	
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while(i < num_elements){
        atomicAdd(&(histo_private[input[i]]), 1);
        i += stride;
    }
    __syncthreads();

    for(int i = threadIdx.x; i < num_bins; i += BLOCK_SIZE){
        atomicAdd(&(bins[i]), histo_private[i]);
    }
	  /*************************************************************************/
}

__global__ void
Atomic_cpy(int *B, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < numElements)
    {
        atomicAdd(&(B[idx]), 1);
    }
}


void
initVec(int *vec, int n, int value)
{
    for (int i = 0; i < n; i++)
        vec[i] = value;
}



