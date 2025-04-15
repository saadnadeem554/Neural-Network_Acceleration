#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f
#define EPOCHS 3
#define BATCH_SIZE 64
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

// Neural network structure with mixed precision support
typedef struct {
    // Master weights (FP32)
    float *d_W1, *d_W2;    // Device weights
    float *d_b1, *d_b2;    // Device biases
    
    // Half precision copies for computation (FP16)
    half *d_W1_half, *d_W2_half;  // Half precision weights
    half *d_b1_half, *d_b2_half;  // Half precision biases
    
    // Host weights (only needed for initialization and final results)
    float *h_W1, *h_W2;    
    float *h_b1, *h_b2;    
} NeuralNetwork;


// Kernel to convert float array to half precision
__global__ void convertFloatToHalfArray(float* src, half* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __float2half(src[idx]);
    }
}

// Kernel to convert half array to float precision
__global__ void convertHalfToFloatArray(half* src, float* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __half2float(src[idx]);
    }
}

// Fused kernel for matrix multiplication + ReLU activation with mixed precision
__global__ void batchFCReluKernelMixedPrecision(half* weights, half* inputs, half* outputs, half* bias,
                                            int output_size, int input_size, int batch_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;
    
    if (batch < batch_size && row < output_size) {
        // Use float for accumulation (higher precision)
        float sum = __half2float(bias[row]);
        
        // Calculate matrix multiplication with half precision inputs
        // but float accumulation for better numerical stability
        for (int i = 0; i < input_size; i++) {
            sum += __half2float(weights[row * input_size + i]) * 
                   __half2float(inputs[batch * input_size + i]);
        }
        
        // Apply ReLU and convert back to half precision
        outputs[batch * output_size + row] = __float2half(fmaxf(0.0f, sum));
    }
}

// Kernel for matrix multiplication + linear (no activation) with mixed precision
__global__ void batchFCKernelMixedPrecision(half* weights, half* inputs, half* outputs, half* bias,
                                        int output_size, int input_size, int batch_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;
    
    if (batch < batch_size && row < output_size) {
        // Use float for accumulation
        float sum = __half2float(bias[row]);
        
        // Matrix multiplication with FP32 accumulation
        for (int i = 0; i < input_size; i++) {
            sum += __half2float(weights[row * input_size + i]) * 
                   __half2float(inputs[batch * input_size + i]);
        }
        
        // Store result in half precision
        outputs[batch * output_size + row] = __float2half(sum);
    }
}

__global__ void calculateBatchLossAccuracyMixedPrecision(half* d_batch_output, half* d_batch_target, 
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
            half* output = d_batch_output + batch_idx * OUTPUT_SIZE;
            half* target = d_batch_target + batch_idx * OUTPUT_SIZE;
            
            // Find predicted class
            int pred = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (__half2float(output[j]) > __half2float(output[pred])) {
                    pred = j;
                }
            }
            
            // Find actual class
            int actual = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (__half2float(target[j]) > __half2float(target[actual])) {
                    actual = j;
                }
            }
            
            // Compute loss
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (__half2float(target[j]) > 0.5f) {
                    batch_loss[tid] -= logf(fmaxf(__half2float(output[j]), 1e-7f));
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

NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Allocate host memory (FP32 only)
    net->h_W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    net->h_W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    net->h_b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    net->h_b2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    // Xavier/Glorot initialization
    float w1_scale = sqrtf(6.0f / (INPUT_SIZE + HIDDEN_SIZE));
    float w2_scale = sqrtf(6.0f / (HIDDEN_SIZE + OUTPUT_SIZE));

    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++)
        net->h_W1[i] = ((2.0f * (float)rand() / RAND_MAX) - 1.0f) * w1_scale;
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++)
        net->h_W2[i] = ((2.0f * (float)rand() / RAND_MAX) - 1.0f) * w2_scale;
    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->h_b1[i] = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->h_b2[i] = 0.0f;

    // Allocate device memory for FP32 (master) weights
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(float)));
    
    // Allocate device memory for FP16 (computation) weights
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W1_half, HIDDEN_SIZE * INPUT_SIZE * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W2_half, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b1_half, HIDDEN_SIZE * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b2_half, OUTPUT_SIZE * sizeof(half)));

    // Copy FP32 data to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W1, net->h_W1, 
        HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W2, net->h_W2,
        OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b1, net->h_b1,
        HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b2, net->h_b2,
        OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        
    // Convert and copy to half precision versions
    convertFloatToHalfArray<<<(HIDDEN_SIZE * INPUT_SIZE + 255) / 256, 256>>>(
        net->d_W1, net->d_W1_half, HIDDEN_SIZE * INPUT_SIZE);
    convertFloatToHalfArray<<<(OUTPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(
        net->d_W2, net->d_W2_half, OUTPUT_SIZE * HIDDEN_SIZE);
    convertFloatToHalfArray<<<(HIDDEN_SIZE + 255) / 256, 256>>>(
        net->d_b1, net->d_b1_half, HIDDEN_SIZE);
    convertFloatToHalfArray<<<(OUTPUT_SIZE + 255) / 256, 256>>>(
        net->d_b2, net->d_b2_half, OUTPUT_SIZE);

    return net;
}
__global__ void batchComputeGradientsMixedPrecision(half* d_batch_output, half* d_batch_target, 
                                                   half* d_batch_hidden, half* d_batch_input,
                                                   float* d_W2_grad, float* d_b2_grad,
                                                   float* d_W1_grad, float* d_b1_grad,
                                                   half* d_W2_half, half* d_W1_half,
                                                   int batchSize) {
    int tid = threadIdx.x;
    int batch = blockIdx.x;
    
    if (batch >= batchSize) return;
    
    // Get pointers to this batch's data
    half* output = d_batch_output + batch * OUTPUT_SIZE;
    half* target = d_batch_target + batch * OUTPUT_SIZE;
    half* hidden = d_batch_hidden + batch * HIDDEN_SIZE;
    half* input = d_batch_input + batch * INPUT_SIZE;
    
    // Use shared memory to compute and store output errors (as float)
    __shared__ float output_errors[32];  // Enough for OUTPUT_SIZE=10
    
    // Calculate output errors in parallel, converting to float for better precision
    if (tid < OUTPUT_SIZE) {
        output_errors[tid] = __half2float(output[tid]) - __half2float(target[tid]);
        
        // Update bias gradient
        atomicAdd(&d_b2_grad[tid], output_errors[tid]);
    }
    __syncthreads();
    
    // Each thread updates a subset of W2 gradients
    for (int i = tid; i < OUTPUT_SIZE * HIDDEN_SIZE; i += blockDim.x) {
        int out_idx = i / HIDDEN_SIZE;
        int hid_idx = i % HIDDEN_SIZE;
        atomicAdd(&d_W2_grad[i], output_errors[out_idx] * __half2float(hidden[hid_idx]));
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
            hidden_errors[j] += output_errors[i] * __half2float(d_W2_half[i * HIDDEN_SIZE + j]);
        }
    }
    __syncthreads();
    
    // Apply ReLU derivative and update bias gradients
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) {
        // Apply ReLU derivative - output is positive if hidden value is positive
        hidden_errors[i] *= (__half2float(hidden[i]) > 0.0f);
        atomicAdd(&d_b1_grad[i], hidden_errors[i]);
    }
    __syncthreads();
    
    // Each thread updates a subset of W1 gradients
    for (int i = tid; i < HIDDEN_SIZE * INPUT_SIZE; i += blockDim.x) {
        int hid_idx = i / INPUT_SIZE;
        int in_idx = i % INPUT_SIZE;
        atomicAdd(&d_W1_grad[i], hidden_errors[hid_idx] * __half2float(input[in_idx]));
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

// Backward pass with mixed precision
void batchBackwardMixedPrecision(NeuralNetwork* net, half* d_batch_input, half* d_batch_hidden, 
                             half* d_batch_output, half* d_batch_target, int batchSize) {
    // We'll still maintain gradient buffers in FP32 for better stability
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
    
    // Compute gradients with mixed precision kernel
    dim3 blockDim(256);
    dim3 gridDim(batchSize);
    
    // This kernel will compute gradients using half precision inputs
    // but accumulate in float precision for the weight updates
    batchComputeGradientsMixedPrecision<<<gridDim, blockDim>>>(
        d_batch_output, d_batch_target, d_batch_hidden, d_batch_input,
        d_W2_grad, d_b2_grad, d_W1_grad, d_b1_grad,
        net->d_W2_half, net->d_W1_half, batchSize
    );
    
    // Update parameters using FP32 master weights
    batchUpdateParametersOptimized<<<32, 256>>>(
        net->d_W1, net->d_b1, d_W1_grad, d_b1_grad,
        HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE, batchSize
    );
    
    batchUpdateParametersOptimized<<<32, 256>>>(
        net->d_W2, net->d_b2, d_W2_grad, d_b2_grad,
        OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE, batchSize
    );
    
    // After updating the FP32 master weights, update the FP16 copies
    convertFloatToHalfArray<<<(HIDDEN_SIZE * INPUT_SIZE + 255) / 256, 256>>>(
        net->d_W1, net->d_W1_half, HIDDEN_SIZE * INPUT_SIZE);
    convertFloatToHalfArray<<<(OUTPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(
        net->d_W2, net->d_W2_half, OUTPUT_SIZE * HIDDEN_SIZE);
    convertFloatToHalfArray<<<(HIDDEN_SIZE + 255) / 256, 256>>>(
        net->d_b1, net->d_b1_half, HIDDEN_SIZE);
    convertFloatToHalfArray<<<(OUTPUT_SIZE + 255) / 256, 256>>>(
        net->d_b2, net->d_b2_half, OUTPUT_SIZE);
}

// Optimized softmax for half precision
__global__ void batchSoftmaxSmallKernelMixedPrecision(half* x, int size, int batchSize) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch < batchSize) {
        // Get pointer to this batch's data
        half* batch_data = x + batch * size;
        
        // Use shared memory for this small array
        __shared__ float data[32];  // Convert to float for computation
        __shared__ float max_val;
        __shared__ float sum_val;
        
        // Load and convert data to float for better precision
        if (tid < size) {
            data[tid] = __half2float(batch_data[tid]);
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
        
        // Compute exp(x - max) in float precision
        if (tid < size) {
            data[tid] = expf(data[tid] - max_val);
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
        
        // Normalize and write back to half precision
        if (tid < size) {
            batch_data[tid] = __float2half(data[tid] / sum_val);
        }
    }
}
// Forward pass using mixed precision
void forwardBatchMixedPrecision(NeuralNetwork* net, half* d_batch_input, half* d_batch_hidden, 
    half* d_batch_output, int batchSize, cudaStream_t stream = 0) {
// Set block dimensions for FC layers
dim3 blockDim(1, 256);  // Using y dimension for row parallelism

// First layer: FC + ReLU fused (using half precision)
dim3 gridDim1(1, (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y, batchSize);
batchFCReluKernelMixedPrecision<<<gridDim1, blockDim, 0, stream>>>(
net->d_W1_half, d_batch_input, d_batch_hidden, 
net->d_b1_half, HIDDEN_SIZE, INPUT_SIZE, batchSize);

// Second layer: FC (using half precision)
dim3 gridDim2(1, (OUTPUT_SIZE + blockDim.y - 1) / blockDim.y, batchSize);
batchFCKernelMixedPrecision<<<gridDim2, blockDim, 0, stream>>>(
net->d_W2_half, d_batch_hidden, d_batch_output,
net->d_b2_half, OUTPUT_SIZE, HIDDEN_SIZE, batchSize);

// Apply optimized softmax for small output vectors
// The softmax kernel will need to be updated to work with half precision
batchSoftmaxSmallKernelMixedPrecision<<<batchSize, 32, 0, stream>>>(
d_batch_output, OUTPUT_SIZE, batchSize);
}
// New train function with mixed precision
void trainMixedPrecision(NeuralNetwork* net, float** h_images, float** h_labels, int numImages) {
    const int batchSize = BATCH_SIZE;
    const int numBatches = (numImages + batchSize - 1) / batchSize;
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // Allocate device memory in half precision for activations
    half *d_batch_input_half, *d_batch_hidden_half, *d_batch_output_half, *d_batch_target_half;
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_input_half, batchSize * INPUT_SIZE * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_hidden_half, batchSize * HIDDEN_SIZE * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_output_half, batchSize * OUTPUT_SIZE * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_target_half, batchSize * OUTPUT_SIZE * sizeof(half)));
    
    // We need float buffers for initial data loading
    float *d_batch_input_float, *d_batch_target_float;
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_input_float, batchSize * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_target_float, batchSize * OUTPUT_SIZE * sizeof(float)));
    
    // Metrics are still in float precision
    float *d_loss;
    int *d_correct;
    CHECK_CUDA_ERROR(cudaMalloc(&d_loss, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_correct, sizeof(int)));
    
    // Use page-locked memory for faster transfers
    float *h_batch_data;  // Combined buffer for input and target
    size_t input_bytes = batchSize * INPUT_SIZE * sizeof(float);
    size_t target_bytes = batchSize * OUTPUT_SIZE * sizeof(float);
    CHECK_CUDA_ERROR(cudaMallocHost(&h_batch_data, input_bytes + target_bytes));
    
    float *h_batch_input = h_batch_data;
    float *h_batch_target = h_batch_data + batchSize * INPUT_SIZE;
    
    // Create index array for shuffling
    int* indices = (int*)malloc(numImages * sizeof(int));
    for (int i = 0; i < numImages; i++) {
        indices[i] = i;
    }
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
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
        
        // Train in batches
        for (int batch = 0; batch < numBatches; batch++) {
            int start_idx = batch * batchSize;
            int current_batch_size = min(batchSize, numImages - start_idx);
            
            // Prepare batch data in coalesced order
            for (int i = 0; i < current_batch_size; i++) {
                int idx = indices[start_idx + i];
                
                memcpy(h_batch_input + i * INPUT_SIZE, 
                       h_images[idx], 
                       INPUT_SIZE * sizeof(float));
                       
                memcpy(h_batch_target + i * OUTPUT_SIZE, 
                       h_labels[idx], 
                       OUTPUT_SIZE * sizeof(float));
            }
            
            // Time transfer
            cudaEventRecord(start, stream);
            
            // Transfer data to FP32 buffers first
            cudaMemcpyAsync(d_batch_input_float, h_batch_input,
                current_batch_size * INPUT_SIZE * sizeof(float), 
                cudaMemcpyHostToDevice, stream);
                
            cudaMemcpyAsync(d_batch_target_float, h_batch_target,
                current_batch_size * OUTPUT_SIZE * sizeof(float), 
                cudaMemcpyHostToDevice, stream);
                
            // Convert to half precision
            convertFloatToHalfArray<<<(current_batch_size * INPUT_SIZE + 255) / 256, 256, 0, stream>>>(
                d_batch_input_float, d_batch_input_half, current_batch_size * INPUT_SIZE);
                
            convertFloatToHalfArray<<<(current_batch_size * OUTPUT_SIZE + 255) / 256, 256, 0, stream>>>(
                d_batch_target_float, d_batch_target_half, current_batch_size * OUTPUT_SIZE);
            
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            transferTime += milliseconds;
            
            // Time forward pass
            cudaEventRecord(start, stream);
            
            forwardBatchMixedPrecision(net, d_batch_input_half, d_batch_hidden_half, 
                                    d_batch_output_half, current_batch_size, stream);
                                    
            calculateBatchLossAccuracyMixedPrecision<<<current_batch_size, BLOCK_SIZE, 0, stream>>>(
                d_batch_output_half, d_batch_target_half, d_loss, d_correct, current_batch_size);
                
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            forwardTime += milliseconds;
            
            // Time backward pass
            cudaEventRecord(start, stream);
            
            batchBackwardMixedPrecision(net, d_batch_input_half, d_batch_hidden_half, 
                                    d_batch_output_half, d_batch_target_half, current_batch_size);
                                    
            cudaEventRecord(stop, stream);
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
    cudaStreamDestroy(stream);
    
    // Free half precision buffers
    cudaFree(d_batch_input_half);
    cudaFree(d_batch_hidden_half);
    cudaFree(d_batch_output_half);
    cudaFree(d_batch_target_half);
    
    // Free float buffers
    cudaFree(d_batch_input_float);
    cudaFree(d_batch_target_float);
    
    cudaFree(d_loss);
    cudaFree(d_correct);
    cudaFreeHost(h_batch_data);
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

// Mixed precision evaluation function
void evaluateMixedPrecision(NeuralNetwork* net, float** h_images, float** h_labels, int numImages) {
    const int batchSize = BATCH_SIZE;
    
    // Allocate device memory for batches - both float and half precision
    float *d_batch_input_float, *d_batch_target_float;
    half *d_batch_input_half, *d_batch_hidden_half, *d_batch_output_half, *d_batch_target_half;
    
    // Allocate float precision buffers for data loading
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_input_float, batchSize * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_target_float, batchSize * OUTPUT_SIZE * sizeof(float)));
    
    // Allocate half precision buffers for computation
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_input_half, batchSize * INPUT_SIZE * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_hidden_half, batchSize * HIDDEN_SIZE * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_output_half, batchSize * OUTPUT_SIZE * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_target_half, batchSize * OUTPUT_SIZE * sizeof(half)));
    
    // For metrics tracking
    float *d_loss;
    int *d_correct;
    CHECK_CUDA_ERROR(cudaMalloc(&d_loss, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_correct, sizeof(int)));
    
    // Initialize counters
    float h_loss = 0.0f;
    int h_correct = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_correct, &h_correct, sizeof(int), cudaMemcpyHostToDevice));
    
    // Use pinned memory for faster transfers
    float *h_batch_data;
    size_t total_bytes = batchSize * (INPUT_SIZE + OUTPUT_SIZE) * sizeof(float);
    CHECK_CUDA_ERROR(cudaMallocHost(&h_batch_data, total_bytes));
    
    float *h_batch_input = h_batch_data;
    float *h_batch_target = h_batch_data + batchSize * INPUT_SIZE;
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Process test data in batches
    for (int batch_start = 0; batch_start < numImages; batch_start += batchSize) {
        int current_batch_size = min(batchSize, numImages - batch_start);
        
        // Pack batch data in coalesced order
        for (int i = 0; i < current_batch_size; i++) {
            memcpy(h_batch_input + i * INPUT_SIZE, 
                   h_images[batch_start + i],
                   INPUT_SIZE * sizeof(float));
            
            memcpy(h_batch_target + i * OUTPUT_SIZE,
                   h_labels[batch_start + i],
                   OUTPUT_SIZE * sizeof(float));
        }
        
        // Copy to device (float precision first)
        cudaMemcpyAsync(d_batch_input_float, h_batch_input,
            current_batch_size * INPUT_SIZE * sizeof(float), 
            cudaMemcpyHostToDevice, stream);
            
        cudaMemcpyAsync(d_batch_target_float, h_batch_target,
            current_batch_size * OUTPUT_SIZE * sizeof(float), 
            cudaMemcpyHostToDevice, stream);
        
        // Convert to half precision
        convertFloatToHalfArray<<<(current_batch_size * INPUT_SIZE + 255) / 256, 256, 0, stream>>>(
            d_batch_input_float, d_batch_input_half, current_batch_size * INPUT_SIZE);
            
        convertFloatToHalfArray<<<(current_batch_size * OUTPUT_SIZE + 255) / 256, 256, 0, stream>>>(
            d_batch_target_float, d_batch_target_half, current_batch_size * OUTPUT_SIZE);
        
        // Forward pass with half precision
        forwardBatchMixedPrecision(net, d_batch_input_half, d_batch_hidden_half, 
                                d_batch_output_half, current_batch_size, stream);
        
        // Calculate accuracy with half precision
        calculateBatchLossAccuracyMixedPrecision<<<current_batch_size, BLOCK_SIZE, 0, stream>>>(
            d_batch_output_half, d_batch_target_half, d_loss, d_correct, current_batch_size);
    }
    
    cudaStreamSynchronize(stream);
    
    // Get final metrics
    CHECK_CUDA_ERROR(cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Clean up resources
    cudaFree(d_batch_input_float);
    cudaFree(d_batch_target_float);
    cudaFree(d_batch_input_half);
    cudaFree(d_batch_hidden_half);
    cudaFree(d_batch_output_half);
    cudaFree(d_batch_target_half);
    cudaFree(d_loss);
    cudaFree(d_correct);
    cudaFreeHost(h_batch_data);
    cudaStreamDestroy(stream);
    
    printf("Test Accuracy: %.2f%% - Loss: %.4f\n", 
           (h_correct / (float)numImages) * 100, h_loss / numImages);
}

// Read MNIST dataset
float** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    float** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;

            // fread(&pixel, sizeof(unsigned char), 1, file);
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }

            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}


float** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    float** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        // fread(&label, sizeof(unsigned char), 1, file);
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}


// Main function remains similar but with float arrays instead of double
int main() {
    printf("MNIST Neural Network\n\n");
    // Start measuring total execution time
    clock_t total_start = clock();

    // Measure time for loading data
    clock_t start = clock();
    float** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    float** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    float** test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    float** test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);
    clock_t end = clock();
    printf("Time to load data: %.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Measure time for training
    start = clock();
    NeuralNetwork* net = createNetwork();
    trainMixedPrecision(net, train_images, train_labels, 60000);
    end = clock();
    printf("Time to train: %.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Measure time for evaluation
    start = clock();
    evaluateMixedPrecision(net, test_images, test_labels, 10000);

    end = clock();
    printf("Time to evaluate: %.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    // End measuring total execution time
    clock_t total_end = clock();
    printf("Total execution time: %.3fs\n", (double)(total_end - total_start) / CLOCKS_PER_SEC);

    freeNetwork(net);
    return 0;
}
