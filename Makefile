all:  CUPTI_receiver CUPTI_sender 

CUPTI_PATH=/usr/local/cuda-9.1/extras/CUPTI
INCLUDES = -I cupti_profiler.h -I /usr/local/cuda-9.1/extras/CUPTI/include -I /raid/yzhan846/nccl/build/include
CXXARGS = -std=c++14 -g -arch=sm_60
CXXARGS += -Xcompiler -DNDEBUG

# CXXARGSPLUS = --default-stream per-thread 
LIBS = -lcuda -L$(CUPTI_PATH)/lib64 -lcupti -lcudart -lcusparse -lnccl 


# CXXARGS += -Xptxas="-flcm=cg"  # --def-load-cache  "cg" -> Cache at global level (cache in L2 and below, not L1).
# CXXARGS += -Xptxas="-fscm=cg"  # --def-store-cache  "cg" -> Cache at global level (cache in L2 and below, not L1).




CUPTI_receiver : CUPTI_receiver.cu	
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_receiver.cu -o CUPTI_receiver

CUPTI_sender : CUPTI_sender.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_sender.cu -o CUPTI_sender



clean:  
	rm -f *.o  CUPTI_receiver CUPTI_sender