all:  CUPTI_receiver CUPTI_sender  conv_100

CUPTI_PATH=/usr/local/cuda-9.1/extras/CUPTI
INCLUDES = -I ./ -I /usr/local/cuda-9.1/extras/CUPTI/include 
CXXARGS = -std=c++14 -g -arch=sm_60
CXXARGS += -Xcompiler -DNDEBUG

# CXXARGSPLUS = --default-stream per-thread 
LIBS = -lcuda -L$(CUPTI_PATH)/lib64  -lcupti 

# -lcudart -lcusparse  -lnccl 


# CXXARGS += -Xptxas="-flcm=cg"  # --def-load-cache  "cg" -> Cache at global level (cache in L2 and below, not L1).
# CXXARGS += -Xptxas="-fscm=cg"  # --def-store-cache  "cg" -> Cache at global level (cache in L2 and below, not L1).




CUPTI_receiver : CUPTI_receiver.cu	
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_receiver.cu -o CUPTI_receiver

CUPTI_sender : CUPTI_sender.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_sender.cu -o CUPTI_sender

nccl_test : nccl_test.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) -lnccl nccl_test.cu -o nccl_test

conv_100 : conv_100.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) conv_100.cu -o conv_100

clean:  
	rm -f *.o  CUPTI_receiver CUPTI_sender nccl_test conv_100