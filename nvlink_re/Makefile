all:  nvlink_re 

CUPTI_PATH=/usr/local/cuda-9.1/extras/CUPTI
INCLUDES = -I ./ -I /usr/local/cuda-9.1/extras/CUPTI/include 
CXXARGS = -std=c++14 -g 
CXXARGS += -Xcompiler -DNDEBUG

# CXXARGSPLUS = --default-stream per-thread 
LIBS = -lcuda -L$(CUPTI_PATH)/lib64  -lcupti 

# -lcudart -lcusparse  -lnccl 


# CXXARGS += -Xptxas="-flcm=cg"  # --def-load-cache  "cg" -> Cache at global level (cache in L2 and below, not L1).
# CXXARGS += -Xptxas="-fscm=cg"  # --def-store-cache  "cg" -> Cache at global level (cache in L2 and below, not L1).




nvlink_re : nvlink_re.cu	
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) nvlink_re.cu -o nvlink_re



clean:  
	rm -f *.o  nvlink_re