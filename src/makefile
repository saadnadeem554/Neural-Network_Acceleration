CC = gcc
NVCC = nvcc
CFLAGS = -Wall -O2
CUDAFLAGS = -O3 -arch=sm_80 --use_fast_math -Xptxas -O3,-v --threads 0 -Xcompiler=-march=native,-Ofast -maxrregcount=64

# Directories
V1_DIR = V1
V2_DIR = V2
V3_DIR = V3
V4_DIR = V4

# Executables
V1_EXE = $(V1_DIR)/nn.exe
V2_EXE = $(V2_DIR)/nn_cuda.exe
V3_EXE = $(V3_DIR)/nn_optimized.exe
V4_EXE = $(V4_DIR)/nn_tensor.exe

# Default target
all: v1

# Version 1: Sequential CPU Implementation
v1: $(V1_EXE)
	./$(V1_EXE)

$(V1_EXE): $(V1_DIR)/nn.c
	$(CC) $(CFLAGS) -o $(V1_EXE) $(V1_DIR)/nn.c -lm

# Version 2: Naive GPU Implementation
v2: $(V2_EXE)
	./$(V2_EXE)

$(V2_EXE): $(V2_DIR)/nn.cu
	$(NVCC) $(CUDAFLAGS) -o $(V2_EXE) $(V2_DIR)/nn.cu -lm

# Version 3: Optimized GPU Implementation
v3: $(V3_EXE)
	./$(V3_EXE)

$(V3_EXE): $(V3_DIR)/nn.cu
	$(NVCC) $(CUDAFLAGS) -o $(V3_EXE) $(V3_DIR)/nn.cu -lm

# Version 4: Tensor Core Implementation
v4: $(V4_EXE)
	./$(V4_EXE)

$(V4_EXE): $(V4_DIR)/nn.cu
	$(NVCC) $(CUDAFLAGS) -o $(V4_EXE) $(V4_DIR)/nn.cu -lm -lcublas

# Clean all executables
clean:
	rm -f $(V1_EXE) $(V2_EXE) $(V3_EXE) $(V4_EXE)
