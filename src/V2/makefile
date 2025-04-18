# Compiler
NVCC = nvcc

# Compiler Flags
CUDAFLAGS = -O3 -arch=sm_80 --use_fast_math -Xptxas -O3,-v --threads 0 -Xcompiler=-march=native,-Ofast -maxrregcount=64

# Executable
EXE = nn_cuda.exe

# Default target
all: $(EXE)

# Run target
run: $(EXE)
	./$(EXE)

# Compile the CUDA source
$(EXE): nn.cu
	$(NVCC) $(CUDAFLAGS) -o $(EXE) nn.cu -lm

# Clean the executable
clean:
	rm -f $(EXE)

