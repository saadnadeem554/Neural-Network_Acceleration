#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f
#define EPOCHS 3
#define BATCH_SIZE 4096
#define NUM_CLASSES 10
#define BLOCK_SIZE 256

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Neural network structure for GPU
typedef struct {
    float *d_W1, *d_W2;    // Device weights
    float *d_b1, *d_b2;    // Device biases
    float *h_W1, *h_W2;    // Host weights
    float *h_b1, *h_b2;    // Host biases
} NeuralNetwork;

// Fused kernel for matrix multiplication + ReLU activation
__global__ void batchFCReluKernel(float* weights, float* inputs, float* outputs, float* bias,
                               int output_size, int input_size, int batch_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;
    
    if (batch < batch_size && row < output_size) {
        float sum = bias[row];
        
        for (int i = 0; i < input_size; i++) {
            sum += weights[row * input_size + i] * inputs[batch * input_size + i];
        }
        
        // Apply ReLU directly
        outputs[batch * output_size + row] = fmaxf(0.0f, sum);
    }
}

// Kernel for matrix multiplication + linear (no activation)
__global__ void batchFCKernel(float* weights, float* inputs, float* outputs, float* bias,
                           int output_size, int input_size, int batch_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;
    
    if (batch < batch_size && row < output_size) {
        float sum = bias[row];
        
        for (int i = 0; i < input_size; i++) {
            sum += weights[row * input_size + i] * inputs[batch * input_size + i];
        }
        
        // No activation
        outputs[batch * output_size + row] = sum;
    }
}



// Optimized softmax kernel with register caching
__global__ void batchSoftmaxSmallKernel(float* x, int size, int batchSize) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch < batchSize) {
        // Cache batch offset in register
        float* batch_data = x + batch * size;
        
        // Use shared memory for this small array
        __shared__ float data[32];
        __shared__ float max_val;
        __shared__ float sum_val;
        
        // Load data into shared memory (and cache our value in register)
        float my_val = 0.0f;
        if (tid < size) {
            my_val = batch_data[tid];
            data[tid] = my_val;
        }
        __syncthreads();
        
        // Find maximum with thread 0
        if (tid == 0) {
            max_val = data[0];
            for (int i = 1; i < size; i++) {
                max_val = fmaxf(max_val, data[i]);
            }
        }
        __syncthreads();
        
        // Cache max value in register
        float max_val_reg = max_val;
        
        // Compute exp(x - max) and prepare for sum
        if (tid < size) {
            float exp_val = expf(my_val - max_val_reg);
            data[tid] = exp_val;
        }
        __syncthreads();
        
        // Compute sum with reduction
        if (tid == 0) {
            sum_val = 0.0f;
            for (int i = 0; i < size; i++) {
                sum_val += data[i];
            }
        }
        __syncthreads();
        
        // Cache sum in register
        float sum_val_reg = sum_val;
        
        // Normalize and write back
        if (tid < size) {
            batch_data[tid] = data[tid] / sum_val_reg;
        }
    }
}

// Add these GPU kernels for calculating loss and accuracy on device
__global__ void calculateBatchLossAccuracy(float* d_batch_output, float* d_batch_target, 
                                         float* d_loss, int* d_correct, int batchSize) {
    __shared__ float batch_loss[BLOCK_SIZE];
    __shared__ int batch_correct[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    
    batch_loss[tid] = 0.0f;
    batch_correct[tid] = 0;
    
    if (batch_idx < batchSize) {
        // Each thread handles one sample in the batch
        if (tid == 0) {
            float* output = d_batch_output + batch_idx * OUTPUT_SIZE;
            float* target = d_batch_target + batch_idx * OUTPUT_SIZE;
            
            // Find predicted class
            int pred = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) {
                    pred = j;
                }
            }
            
            // Find actual class
            int actual = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (target[j] > target[actual]) {
                    actual = j;
                }
            }
            
            // Compute loss
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (target[j] > 0.5f) {
                    batch_loss[tid] -= logf(fmaxf(output[j], 1e-7f));
                }
            }
            
            // Check if prediction was correct
            if (pred == actual) {
                batch_correct[tid] = 1;
            }
        }
    }
    
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            batch_loss[tid] += batch_loss[tid + stride];
            batch_correct[tid] += batch_correct[tid + stride];
        }
        __syncthreads();
    }
    
    // Write results back
    if (tid == 0) {
        atomicAdd(d_loss, batch_loss[0]);
        atomicAdd(d_correct, batch_correct[0]);
    }
}

// Allocate memory for a matrix
float** allocateMatrix(int rows, int cols) {
    float** mat = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (float*)malloc(cols * sizeof(float));
    }
    return mat;
}

// Enhanced createNetwork with CPU-optimized initialization
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Allocate host memory
    net->h_W1 = (float*)aligned_alloc(32, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    net->h_W2 = (float*)aligned_alloc(32, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    net->h_b1 = (float*)aligned_alloc(32, HIDDEN_SIZE * sizeof(float));
    net->h_b2 = (float*)aligned_alloc(32, OUTPUT_SIZE * sizeof(float));

    // Initialize weights with Xavier/Glorot on CPU (more efficient than GPU for this)
    float w1_scale = sqrtf(6.0f / (INPUT_SIZE + HIDDEN_SIZE));
    float w2_scale = sqrtf(6.0f / (HIDDEN_SIZE + OUTPUT_SIZE));

    // Use OpenMP for parallel initialization
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++)
        net->h_W1[i] = ((2.0f * (float)rand() / RAND_MAX) - 1.0f) * w1_scale;
        
    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++)
        net->h_W2[i] = ((2.0f * (float)rand() / RAND_MAX) - 1.0f) * w2_scale;
        
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->h_b1[i] = 0.0f;
        
    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->h_b2[i] = 0.0f;

    // Allocate device memory and copy from host
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(float)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W1, net->h_W1, 
        HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W2, net->h_W2,
        OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b1, net->h_b1,
        HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b2, net->h_b2,
        OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    return net;
}

// Optimized gradient computation kernel
__global__ void batchComputeGradientsOptimized(float* d_batch_output, float* d_batch_target, 
                                           float* d_batch_hidden, float* d_batch_input,
                                           float* d_W2_grad, float* d_b2_grad,
                                           float* d_W1_grad, float* d_b1_grad,
                                           float* d_W2, float* d_W1,
                                           int batchSize) {
    int tid = threadIdx.x;
    int batch = blockIdx.x;
    
    if (batch >= batchSize) return;
    
    // Get pointers to this batch's data
    float* output = d_batch_output + batch * OUTPUT_SIZE;
    float* target = d_batch_target + batch * OUTPUT_SIZE;
    float* hidden = d_batch_hidden + batch * HIDDEN_SIZE;
    float* input = d_batch_input + batch * INPUT_SIZE;
    
    // Use shared memory to compute and store output errors
    __shared__ float output_errors[32];  // Enough for OUTPUT_SIZE=10
    
    // Calculate output errors in parallel
    if (tid < OUTPUT_SIZE) {
        output_errors[tid] = output[tid] - target[tid];
        
        // Update bias gradient
        atomicAdd(&d_b2_grad[tid], output_errors[tid]);
    }
    __syncthreads();
    
    // Each thread updates a subset of W2 gradients
    for (int i = tid; i < OUTPUT_SIZE * HIDDEN_SIZE; i += blockDim.x) {
        int out_idx = i / HIDDEN_SIZE;
        int hid_idx = i % HIDDEN_SIZE;
        atomicAdd(&d_W2_grad[i], output_errors[out_idx] * hidden[hid_idx]);
    }
    
    // Compute hidden layer errors
    __shared__ float hidden_errors[128];  // Maximum HIDDEN_SIZE we'd expect
    
    // First initialize all to zero
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) {
        hidden_errors[i] = 0.0f;
    }
    __syncthreads();
    
    // Compute hidden errors in parallel across threads
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = tid; j < HIDDEN_SIZE; j += blockDim.x) {
            hidden_errors[j] += output_errors[i] * d_W2[i * HIDDEN_SIZE + j];
        }
    }
    __syncthreads();
    
    // Apply ReLU derivative and update bias gradients
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) {
        hidden_errors[i] *= (hidden[i] > 0.0f);
        atomicAdd(&d_b1_grad[i], hidden_errors[i]);
    }
    __syncthreads();
    
    // Each thread updates a subset of W1 gradients
    for (int i = tid; i < HIDDEN_SIZE * INPUT_SIZE; i += blockDim.x) {
        int hid_idx = i / INPUT_SIZE;
        int in_idx = i % INPUT_SIZE;
        atomicAdd(&d_W1_grad[i], hidden_errors[hid_idx] * input[in_idx]);
    }
}

// Optimized parameter update kernel
__global__ void batchUpdateParametersOptimized(float* d_W, float* d_b, float* d_W_grad, 
                                           float* d_b_grad, int rows, int cols, 
                                           float learning_rate, int batchSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Each thread updates multiple weights using grid-stride loop
    for (int i = tid; i < rows * cols; i += blockDim.x * gridDim.x) {
        d_W[i] -= (learning_rate / batchSize) * d_W_grad[i];
        d_W_grad[i] = 0.0f;  // Reset gradient for next batch
    }
    
    // Update biases with efficient access pattern
    if (tid < rows) {
        d_b[tid] -= (learning_rate / batchSize) * d_b_grad[tid];
        d_b_grad[tid] = 0.0f;  // Reset gradient for next batch
    }
}

// Optimized backward pass using more efficient kernels
void batchBackward(NeuralNetwork* net, float* d_batch_input, float* d_batch_hidden, 
                  float* d_batch_output, float* d_batch_target, int batchSize) {
    // Allocate gradient matrices
    static float *d_W1_grad, *d_W2_grad, *d_b1_grad, *d_b2_grad;
    static bool gradients_initialized = false;
    
    // Initialize gradient buffers if first call
    if (!gradients_initialized) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_W1_grad, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_W2_grad, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b1_grad, HIDDEN_SIZE * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b2_grad, OUTPUT_SIZE * sizeof(float)));
        
        gradients_initialized = true;
    }
    
    // Clear gradients
    cudaMemset(d_W1_grad, 0, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMemset(d_W2_grad, 0, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMemset(d_b1_grad, 0, HIDDEN_SIZE * sizeof(float));
    cudaMemset(d_b2_grad, 0, OUTPUT_SIZE * sizeof(float));
    
    // Compute gradients with optimized kernel - one block per batch item
    dim3 blockDim(256);
    dim3 gridDim(batchSize);
    
    batchComputeGradientsOptimized<<<gridDim, blockDim>>>(
        d_batch_output, d_batch_target, d_batch_hidden, d_batch_input,
        d_W2_grad, d_b2_grad, d_W1_grad, d_b1_grad,
        net->d_W2, net->d_W1, batchSize
    );
    
    // Update parameters with optimized kernel
    batchUpdateParametersOptimized<<<32, 256>>>(
        net->d_W1, net->d_b1, d_W1_grad, d_b1_grad,
        HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE, batchSize
    );
    
    batchUpdateParametersOptimized<<<32, 256>>>(
        net->d_W2, net->d_b2, d_W2_grad, d_b2_grad,
        OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE, batchSize
    );
}

// Optimized forward pass using kernel fusion
void forwardBatch(NeuralNetwork* net, float* d_batch_input, float* d_batch_hidden, 
                 float* d_batch_output, int batchSize, cudaStream_t stream = 0) {
    // Set block dimensions for FC layers
    dim3 blockDim(1, 256);  // Using y dimension for row parallelism
    
    // First layer: FC + ReLU fused
    dim3 gridDim1(1, (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y, batchSize);
    batchFCReluKernel<<<gridDim1, blockDim, 0, stream>>>(
        net->d_W1, d_batch_input, d_batch_hidden, 
        net->d_b1, HIDDEN_SIZE, INPUT_SIZE, batchSize);
    
    // Second layer: FC (no activation, will apply softmax after)
    dim3 gridDim2(1, (OUTPUT_SIZE + blockDim.y - 1) / blockDim.y, batchSize);
    batchFCKernel<<<gridDim2, blockDim, 0, stream>>>(
        net->d_W2, d_batch_hidden, d_batch_output,
        net->d_b2, OUTPUT_SIZE, HIDDEN_SIZE, batchSize);
    
    // Apply optimized softmax for small output vectors
    batchSoftmaxSmallKernel<<<batchSize, 32, 0, stream>>>(
        d_batch_output, OUTPUT_SIZE, batchSize);
}

// Modified train function with prefetching
void train(NeuralNetwork* net, float** h_images, float** h_labels, int numImages) {
    const int batchSize = BATCH_SIZE;
    const int numBatches = (numImages + batchSize - 1) / batchSize;
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // Double buffering for prefetching
    float *d_batch_input[2], *d_batch_hidden[2], *d_batch_output[2], *d_batch_target[2];
    
    // Allocate two sets of device memory for double buffering
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_batch_input[i], batchSize * INPUT_SIZE * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_batch_hidden[i], batchSize * HIDDEN_SIZE * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_batch_output[i], batchSize * OUTPUT_SIZE * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_batch_target[i], batchSize * OUTPUT_SIZE * sizeof(float)));
    }
    
    // Allocate metrics memory
    float *d_loss;
    int *d_correct;
    CHECK_CUDA_ERROR(cudaMalloc(&d_loss, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_correct, sizeof(int)));
    
    // Use page-locked memory for faster transfers (also double-buffered)
    float *h_batch_data[2];
    CHECK_CUDA_ERROR(cudaMallocHost(&h_batch_data[0], batchSize * (INPUT_SIZE + OUTPUT_SIZE) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_batch_data[1], batchSize * (INPUT_SIZE + OUTPUT_SIZE) * sizeof(float)));
    
    float *h_batch_input[2], *h_batch_target[2];
    for (int i = 0; i < 2; i++) {
        h_batch_input[i] = h_batch_data[i];
        h_batch_target[i] = h_batch_data[i] + batchSize * INPUT_SIZE;
    }
    
    // Create index array for shuffling
    int* indices = (int*)malloc(numImages * sizeof(int));
    for (int i = 0; i < numImages; i++) {
        indices[i] = i;
    }
    
    // Create two CUDA streams for overlapping operations
    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Reset metrics
        float h_loss = 0.0f;
        int h_correct = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_correct, &h_correct, sizeof(int), cudaMemcpyHostToDevice));
        
        clock_t epoch_start = clock();
        float transferTime = 0, forwardTime = 0, backwardTime = 0;
        
        // Shuffle data indices
        for (int i = numImages - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        // Prepare first batch before the main loop
        int current_batch_size = min(batchSize, numImages);
        int idx_buf = 0;  // Start with buffer 0
        
        // Prepare the first batch data
        for (int i = 0; i < current_batch_size; i++) {
            int idx = indices[i];
            
            memcpy(h_batch_input[idx_buf] + i * INPUT_SIZE, 
                   h_images[idx], 
                   INPUT_SIZE * sizeof(float));
                   
            memcpy(h_batch_target[idx_buf] + i * OUTPUT_SIZE, 
                   h_labels[idx], 
                   OUTPUT_SIZE * sizeof(float));
        }
        
        // Start first batch transfer
        cudaEventRecord(start, stream[idx_buf]);
        cudaMemcpyAsync(d_batch_input[idx_buf], h_batch_input[idx_buf],
            current_batch_size * INPUT_SIZE * sizeof(float), 
            cudaMemcpyHostToDevice, stream[idx_buf]);
            
        cudaMemcpyAsync(d_batch_target[idx_buf], h_batch_target[idx_buf],
            current_batch_size * OUTPUT_SIZE * sizeof(float), 
            cudaMemcpyHostToDevice, stream[idx_buf]);
        cudaEventRecord(stop, stream[idx_buf]);
        
        // Train in batches with prefetching
        for (int batch = 0; batch < numBatches; batch++) {
            int start_idx = batch * batchSize;
            current_batch_size = min(batchSize, numImages - start_idx);
            
            // Current buffer index
            int curr_buf = idx_buf;
            
            // Next buffer index (for prefetching)
            idx_buf = 1 - idx_buf;
            
            // Wait for current transfer to complete
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            transferTime += milliseconds;
            
            // Start preparing next batch data (if not the last batch)
            if (batch + 1 < numBatches) {
                int next_start_idx = (batch + 1) * batchSize;
                int next_batch_size = min(batchSize, numImages - next_start_idx);
                
                // Prepare next batch in the other buffer
                for (int i = 0; i < next_batch_size; i++) {
                    int idx = indices[next_start_idx + i];
                    
                    memcpy(h_batch_input[idx_buf] + i * INPUT_SIZE, 
                           h_images[idx], 
                           INPUT_SIZE * sizeof(float));
                           
                    memcpy(h_batch_target[idx_buf] + i * OUTPUT_SIZE, 
                           h_labels[idx], 
                           OUTPUT_SIZE * sizeof(float));
                }
                
                // Start next batch transfer (overlapped with current computation)
                cudaEventRecord(start, stream[idx_buf]);
                cudaMemcpyAsync(d_batch_input[idx_buf], h_batch_input[idx_buf],
                    next_batch_size * INPUT_SIZE * sizeof(float), 
                    cudaMemcpyHostToDevice, stream[idx_buf]);
                    
                cudaMemcpyAsync(d_batch_target[idx_buf], h_batch_target[idx_buf],
                    next_batch_size * OUTPUT_SIZE * sizeof(float), 
                    cudaMemcpyHostToDevice, stream[idx_buf]);
                cudaEventRecord(stop, stream[idx_buf]);
            }
            
            // Process current batch
            cudaEventRecord(start, stream[curr_buf]);
            forwardBatch(net, d_batch_input[curr_buf], d_batch_hidden[curr_buf], 
                        d_batch_output[curr_buf], current_batch_size, stream[curr_buf]);
                        
            calculateBatchLossAccuracy<<<current_batch_size, BLOCK_SIZE, 0, stream[curr_buf]>>>(
                d_batch_output[curr_buf], d_batch_target[curr_buf], d_loss, d_correct, current_batch_size);
            cudaEventRecord(stop, stream[curr_buf]);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            forwardTime += milliseconds;
            
            // Time backward pass
            cudaEventRecord(start, stream[curr_buf]);
            batchBackward(net, d_batch_input[curr_buf], d_batch_hidden[curr_buf], 
                         d_batch_output[curr_buf], d_batch_target[curr_buf], current_batch_size);
            cudaEventRecord(stop, stream[curr_buf]);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            backwardTime += milliseconds;
        }
        
        // Get final metrics
        CHECK_CUDA_ERROR(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
        
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, h_loss / numImages, (h_correct / (float)numImages) * 100,
               (float)(clock() - epoch_start) / CLOCKS_PER_SEC);
               
        printf("  Transfer: %.2f ms, Forward: %.2f ms, Backward: %.2f ms\n",
               transferTime, forwardTime, backwardTime);
    }
    
    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Clean up streams
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    
    // Free double-buffered memory
    for (int i = 0; i < 2; i++) {
        cudaFree(d_batch_input[i]);
        cudaFree(d_batch_hidden[i]);
        cudaFree(d_batch_output[i]);
        cudaFree(d_batch_target[i]);
        cudaFreeHost(h_batch_data[i]);
    }
    
    cudaFree(d_loss);
    cudaFree(d_correct);
    free(indices);
}

// Free network memory
void freeNetwork(NeuralNetwork* net) {
    // Free host memory
    free(net->h_W1);
    free(net->h_W2);
    free(net->h_b1);
    free(net->h_b2);

    // Free device memory
    cudaFree(net->d_W1);
    cudaFree(net->d_W2);
    cudaFree(net->d_b1);
    cudaFree(net->d_b2);

    free(net);
}

// Updated evaluate function
void evaluate(NeuralNetwork* net, float** h_images, float** h_labels, int numImages) {
    const int batchSize = BATCH_SIZE;
    
    // Allocate device memory for batches
    float *d_batch_input, *d_batch_hidden, *d_batch_output, *d_batch_target;
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_input, batchSize * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_hidden, batchSize * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_output, batchSize * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_target, batchSize * OUTPUT_SIZE * sizeof(float)));
    
    // For loss and accuracy tracking
    float *d_loss;
    int *d_correct;
    CHECK_CUDA_ERROR(cudaMalloc(&d_loss, sizeof(float)));  // Allocate dummy loss buffer
    CHECK_CUDA_ERROR(cudaMalloc(&d_correct, sizeof(int)));
    
    // Initialize counters
    float h_loss = 0.0f;
    int h_correct = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_correct, &h_correct, sizeof(int), cudaMemcpyHostToDevice));
    
    // Use pinned memory for faster transfers
    float *h_batch_input, *h_batch_target;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_batch_input, batchSize * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_batch_target, batchSize * OUTPUT_SIZE * sizeof(float)));
    
    // Process test data in batches
    for (int batch_start = 0; batch_start < numImages; batch_start += batchSize) {
        int current_batch_size = min(batchSize, numImages - batch_start);
        
        // Pack batch data 
        for (int i = 0; i < current_batch_size; i++) {
            memcpy(h_batch_input + i * INPUT_SIZE, 
                   h_images[batch_start + i],
                   INPUT_SIZE * sizeof(float));
            
            memcpy(h_batch_target + i * OUTPUT_SIZE,
                   h_labels[batch_start + i],
                   OUTPUT_SIZE * sizeof(float));
        }
        
        // Copy to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_batch_input, h_batch_input,
            current_batch_size * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_batch_target, h_batch_target,
            current_batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        
        // Forward pass
        forwardBatch(net, d_batch_input, d_batch_hidden, d_batch_output, current_batch_size);
        
        // Use d_loss instead of nullptr - this was causing the error
        calculateBatchLossAccuracy<<<current_batch_size, BLOCK_SIZE>>>(
            d_batch_output, d_batch_target, d_loss, d_correct, current_batch_size);
    }
    
    // Get final accuracy
    CHECK_CUDA_ERROR(cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Clean up resources
    cudaFree(d_batch_input);
    cudaFree(d_batch_hidden);
    cudaFree(d_batch_output);
    cudaFree(d_batch_target);
    cudaFree(d_loss);
    cudaFree(d_correct);
    cudaFreeHost(h_batch_input);
    cudaFreeHost(h_batch_target);
    
    printf("Test Accuracy: %.2f%%\n", (h_correct / (float)numImages) * 100);
}
// Memory-mapped file loading for improved I/O performance
float** loadMNISTImagesOptimized(const char* filename, int numImages) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    
    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) < 0) {
        printf("Error getting file size\n");
        close(fd);
        exit(1);
    }
    
    // Map file into memory
    unsigned char* file_data = (unsigned char*)mmap(
        NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_data == MAP_FAILED) {
        printf("Error mapping file\n");
        close(fd);
        exit(1);
    }
    
    // Allocate memory for images
    float** images = allocateMatrix(numImages, INPUT_SIZE);
    
    // Skip header (16 bytes)
    unsigned char* data_ptr = file_data + 16;
    
    // Process images with OpenMP
    #pragma omp parallel for
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            images[i][j] = data_ptr[i * INPUT_SIZE + j] / 255.0f;
        }
    }
    
    // Unmap and close file
    munmap(file_data, sb.st_size);
    close(fd);
    
    return images;
}

// Memory-mapped label loading
float** loadMNISTLabelsOptimized(const char* filename, int numLabels) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    
    struct stat sb;
    fstat(fd, &sb);
    
    unsigned char* file_data = (unsigned char*)mmap(
        NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_data == MAP_FAILED) {
        printf("Error mapping file\n");
        close(fd);
        exit(1);
    }
    
    float** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    
    // Skip header (8 bytes)
    unsigned char* data_ptr = file_data + 8;
    
    #pragma omp parallel for
    for (int i = 0; i < numLabels; i++) {
        unsigned char label = data_ptr[i];
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0f : 0.0f;
        }
    }
    
    munmap(file_data, sb.st_size);
    close(fd);
    
    return labels;
}
// Main function remains similar but with float arrays instead of double
int main() {
    printf("MNIST Neural Network\n\n");
    clock_t total_start = clock();

    // Measure time for loading data
    clock_t start = clock();
    
    // Load data using memory-mapped I/O (faster than thread-based loading)
    float **train_images = loadMNISTImagesOptimized("../data/train-images.idx3-ubyte", 60000);
    float **train_labels = loadMNISTLabelsOptimized("../data/train-labels.idx1-ubyte", 60000);
    float **test_images = loadMNISTImagesOptimized("../data/t10k-images.idx3-ubyte", 10000);
    float **test_labels = loadMNISTLabelsOptimized("../data/t10k-labels.idx1-ubyte", 10000);
    
    clock_t end = clock();
    printf("Time to load data: %.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);
    
    // Measure time for training
    start = clock();
    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    end = clock();
    printf("Time to train: %.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Measure time for evaluation
    start = clock();
    evaluate(net, test_images, test_labels, 10000);
    end = clock();
    printf("Time to evaluate: %.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    // End measuring total execution time
    clock_t total_end = clock();
    printf("Total execution time: %.3fs\n", (double)(total_end - total_start) / CLOCKS_PER_SEC);

    // Cleanup - CPU-based as it's more efficient than GPU for memory management
    freeNetwork(net);
    
    // Free training and test data
    for (int i = 0; i < 60000; i++) {
        free(train_images[i]);
        free(train_labels[i]);
    }
    for (int i = 0; i < 10000; i++) {
        free(test_images[i]);
        free(test_labels[i]);
    }
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);
    
    return 0;
}