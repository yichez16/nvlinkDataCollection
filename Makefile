all:  CUPTI_profile_simple   nccl_test p2p_driver_op

CUPTI_PATH=/usr/local/cuda-9.1/extras/CUPTI
INCLUDES = -I ../include -I /usr/local/cuda-9.1/extras/CUPTI/include -I /raid/yzhan846/nccl/build/include
CXXARGS = -std=c++14 -g -arch=sm_60
CXXARGS += -Xcompiler -DNDEBUG

# CXXARGSPLUS = --default-stream per-thread 
LIBS = -lcuda -L$(CUPTI_PATH)/lib64 -lcupti -lcudart -lcusparse -lnccl 


# CXXARGS += -Xptxas="-flcm=cg"  # --def-load-cache  "cg" -> Cache at global level (cache in L2 and below, not L1).
# CXXARGS += -Xptxas="-fscm=cg"  # --def-store-cache  "cg" -> Cache at global level (cache in L2 and below, not L1).




p2p_driver_op : p2p_driver_op.cu	
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) p2p_driver_op.cu -o p2p_driver_op

p2p_nvlink_kernel : p2p_nvlink_kernel.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) p2p_nvlink_kernel.cu -o p2p_nvlink_kernel

p2p_atomic : p2p_atomic.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) p2p_atomic.cu -o p2p_atomic

p2p_coalescing : p2p_coalescing.cu	
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) p2p_coalescing.cu -o p2p_coalescing

CUPTI_profile_vecAdd : CUPTI_profile_vecAdd.cu	
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_profile_vecAdd.cu -o CUPTI_profile_vecAdd

CUPTI_profile_simple : CUPTI_profile_simple.cu	
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_profile_simple.cu -o CUPTI_profile_simple

CUPTI_profile_conv: CUPTI_profile_conv.cu	
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_profile_conv.cu -o CUPTI_profile_conv

CUPTI_profile_multiple: CUPTI_profile_multiple.cu	
	nvcc $(CXXARGS) $(CXXARGSPLUS) $(INCLUDES) $(LIBS) CUPTI_profile_multiple.cu -o CUPTI_profile_multiple

CUPTI_profile_matmul: CUPTI_profile_matmul.cu
	nvcc $(CXXARGS) $(CXXARGSPLUS) $(INCLUDES) $(LIBS) CUPTI_profile_matmul.cu -o CUPTI_profile_matmul

nccl_test: nccl_test.cu	
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) nccl_test.cu -o nccl_test

CUPTI_off_aggregate: CUPTI_off_aggregate.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_off_aggregate.cu -o CUPTI_off_aggregate


clean:  
	rm -f *.o  nccl_test CUPTI_off_aggregate CUPTI_profile_matmul p2p_driver_op CUPTI_profile_vecAdd CUPTI_profile_multiple CUPTI_profile_conv p2p_nvlink_kernel CUPTI_profile_simple

