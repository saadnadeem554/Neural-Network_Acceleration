#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

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

// Neural network structure for GPU
typedef struct {
    float *d_W1, *d_W2;    // Device weights
    float *d_b1, *d_b2;    // Device biases
    float *h_W1, *h_W2;    // Host weights
    float *h_b1, *h_b2;    // Host biases
} NeuralNetwork;

// Fix 1: Correct batch matrix multiplication kernel
__global__ void batchMatrixMulKernel(float* A, float* B, float* C, float* bias,
                                   int M, int N, int K, int batchSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;
    
    if (batch < batchSize && row < M && col < K) {
        // For each output element, compute dot product of row of A and column of B
        float sum = bias[row];
        
        // B is now treated as a batch of matrices
        // Each batch element B[batch] is a matrix of shape NxK
        for (int i = 0; i < N; i++) {
            // A[row, i] * B[batch, i, col]
            sum += A[row * N + i] * B[batch * (N * K) + i * K + col];
        }
        
        C[batch * M * K + row * K + col] = sum;
    }
}

// CUDA kernel for ReLU activation for batches
__global__ void batchReluKernel(float* x, int size, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    
    if (batch < batchSize && idx < size) {
        x[batch * size + idx] = fmaxf(0.0f, x[batch * size + idx]);
    }
}

// CUDA kernel for applying softmax to each sample in a batch
__global__ void batchSoftmaxKernel(float* x, int size, int batchSize) {
    int batch = blockIdx.x;
    
    if (batch < batchSize) {
        // Get pointer to this batch's data
        float* batch_data = x + batch * size;
        
        // Find maximum (reduce operation)
        float max_val = -INFINITY;
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            max_val = fmaxf(max_val, batch_data[i]);
        }
        
        // Reduce within block to find maximum
        __shared__ float temp_max[BLOCK_SIZE];
        temp_max[threadIdx.x] = max_val;
        __syncthreads();
        
        for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                temp_max[threadIdx.x] = fmaxf(temp_max[threadIdx.x], temp_max[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        
        float shared_max = temp_max[0];
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            batch_data[i] = expf(batch_data[i] - shared_max);
            sum += batch_data[i];
        }
        
        // Reduce within block to find sum
        __shared__ float temp_sum[BLOCK_SIZE];
        temp_sum[threadIdx.x] = sum;
        __syncthreads();
        
        for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                temp_sum[threadIdx.x] += temp_sum[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        float shared_sum = temp_sum[0];
        
        // Normalize by sum
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            batch_data[i] /= shared_sum;
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

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Allocate host memory
    net->h_W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    net->h_W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    net->h_b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    net->h_b2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    // Update in createNetwork function
    // Initialize weights with Xavier/Glorot initialization
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

    // Allocate device memory
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

// Batch version of computeGradients kernel
__global__ void batchComputeGradients(float* d_batch_output, float* d_batch_target, 
                                    float* d_batch_hidden, float* d_batch_input,
                                    float* d_W2_grad, float* d_b2_grad,
                                    float* d_W1_grad, float* d_b1_grad,
                                    float* d_W2, float* d_W1,
                                    int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    
    if (batch >= batchSize) return;
    
    // Get pointers to this batch's data
    float* output = d_batch_output + batch * OUTPUT_SIZE;
    float* target = d_batch_target + batch * OUTPUT_SIZE;
    float* hidden = d_batch_hidden + batch * HIDDEN_SIZE;
    float* input = d_batch_input + batch * INPUT_SIZE;
    
    // Output layer gradients
    if (idx < OUTPUT_SIZE) {
        float output_error = output[idx] - target[idx];
        
        // Accumulate bias gradients (one per batch item)
        atomicAdd(&d_b2_grad[idx], output_error);
        
        // Accumulate weight gradients
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            atomicAdd(&d_W2_grad[idx * HIDDEN_SIZE + j], output_error * hidden[j]);
        }
    }
    
    // Hidden layer gradients
    if (idx < HIDDEN_SIZE) {
        float hidden_error = 0.0f;
        
        // Calculate error at hidden layer
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_error += (output[j] - target[j]) * d_W2[j * HIDDEN_SIZE + idx];
        }
        
        // Apply ReLU derivative
        hidden_error *= (hidden[idx] > 0.0f);
        
        // Accumulate bias gradients
        atomicAdd(&d_b1_grad[idx], hidden_error);
        
        // Accumulate weight gradients
        for (int j = 0; j < INPUT_SIZE; j++) {
            atomicAdd(&d_W1_grad[idx * INPUT_SIZE + j], hidden_error * input[j]);
        }
    }
}

// Batch version of parameter updates
__global__ void batchUpdateParameters(float* d_W, float* d_b, float* d_W_grad, 
                                    float* d_b_grad, int rows, int cols, 
                                    float learning_rate, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Update weights
    if (idx < rows * cols) {
        // Normalize by batch size and apply learning rate
        d_W[idx] -= (learning_rate / batchSize) * d_W_grad[idx];
        // Reset gradient for next batch
        d_W_grad[idx] = 0.0f;
    }
    
    // Update biases
    if (idx < rows) {
        // Normalize by batch size and apply learning rate
        d_b[idx] -= (learning_rate / batchSize) * d_b_grad[idx];
        // Reset gradient for next batch
        d_b_grad[idx] = 0.0f;
    }
}

// New batched backward function
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
    
    // Clear gradients before accumulating new ones - this is critical!
    cudaMemset(d_W1_grad, 0, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMemset(d_W2_grad, 0, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMemset(d_b1_grad, 0, HIDDEN_SIZE * sizeof(float));
    cudaMemset(d_b2_grad, 0, OUTPUT_SIZE * sizeof(float));
    
    // Compute gradients for batch
    dim3 blockDim(256);
    dim3 gridDim(max(1, (max(OUTPUT_SIZE, HIDDEN_SIZE) + blockDim.x - 1) / blockDim.x), batchSize);
    
    batchComputeGradients<<<gridDim, blockDim>>>(
        d_batch_output, d_batch_target, d_batch_hidden, d_batch_input,
        d_W2_grad, d_b2_grad, d_W1_grad, d_b1_grad,
        net->d_W2, net->d_W1, batchSize
    );
    
    // Update parameters
    dim3 updateBlockDim(256);
    dim3 updateGridDim((max(HIDDEN_SIZE * INPUT_SIZE, OUTPUT_SIZE * HIDDEN_SIZE) + 
                      updateBlockDim.x - 1) / updateBlockDim.x);
                      
    batchUpdateParameters<<<updateGridDim, updateBlockDim>>>(
        net->d_W1, net->d_b1, d_W1_grad, d_b1_grad,
        HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE, batchSize
    );
    
    batchUpdateParameters<<<updateGridDim, updateBlockDim>>>(
        net->d_W2, net->d_b2, d_W2_grad, d_b2_grad,
        OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE, batchSize
    );
}

// Remove debug messages and error checks that aren't needed anymore
void forwardBatch(NeuralNetwork* net, float* d_batch_input, float* d_batch_hidden, 
                 float* d_batch_output, int batchSize, cudaStream_t stream = 0) {
    // First layer: Hidden = W1 * Input + b1
    dim3 blockDim(16, 16);
    
    // For hidden layer, output is of size (batchSize, HIDDEN_SIZE)
    dim3 gridDim;
    
    // K=1 for matrix-vector multiplication (1 column per result)
    gridDim.x = 1;  // K=1 so only 1 column
    gridDim.y = (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y;  // Rows
    gridDim.z = batchSize;  // Batch dimension
    
    // First layer: FC + ReLU
    batchMatrixMulKernel<<<gridDim, blockDim, 0, stream>>>(
        net->d_W1, d_batch_input, d_batch_hidden, 
        net->d_b1, HIDDEN_SIZE, INPUT_SIZE, 1, batchSize);
    
    // Apply ReLU to the hidden layer outputs
    dim3 reluGrid((HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, batchSize);
    batchReluKernel<<<reluGrid, BLOCK_SIZE, 0, stream>>>(
        d_batch_hidden, HIDDEN_SIZE, batchSize);
    
    // Output layer: Output = W2 * Hidden + b2
    gridDim.x = 1;  // K=1 (single column output)
    gridDim.y = (OUTPUT_SIZE + blockDim.y - 1) / blockDim.y;  // Rows
    gridDim.z = batchSize;  // Batch dimension
    
    batchMatrixMulKernel<<<gridDim, blockDim, 0, stream>>>(
        net->d_W2, d_batch_hidden, d_batch_output,
        net->d_b2, OUTPUT_SIZE, HIDDEN_SIZE, 1, batchSize);
    
    // Apply softmax to each sample in the batch
    batchSoftmaxKernel<<<batchSize, BLOCK_SIZE, 0, stream>>>(
        d_batch_output, OUTPUT_SIZE, batchSize);
}

// Modified train function with full batch processing
void train(NeuralNetwork* net, float** h_images, float** h_labels, int numImages) {
    const int batchSize = BATCH_SIZE;
    const int numBatches = (numImages + batchSize - 1) / batchSize;
    
    // Allocate device memory for batches
    float *d_batch_input, *d_batch_hidden, *d_batch_output, *d_batch_target;
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_input, batchSize * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_hidden, batchSize * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_output, batchSize * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_target, batchSize * OUTPUT_SIZE * sizeof(float)));
    
    // Allocate device memory for metrics
    float *d_loss;
    int *d_correct;
    CHECK_CUDA_ERROR(cudaMalloc(&d_loss, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_correct, sizeof(int)));
    
    // Use page-locked memory for faster transfers
    float *h_batch_data;  // Combined buffer for both input and target data
    size_t input_bytes = batchSize * INPUT_SIZE * sizeof(float);
    size_t target_bytes = batchSize * OUTPUT_SIZE * sizeof(float);
    CHECK_CUDA_ERROR(cudaMallocHost(&h_batch_data, input_bytes + target_bytes));
    
    // Create pointers to the appropriate sections of the combined buffer
    float *h_batch_input = h_batch_data;
    float *h_batch_target = h_batch_data + batchSize * INPUT_SIZE;
    
    // Create and initialize index array for shuffling
    int* indices = (int*)malloc(numImages * sizeof(int));
    for (int i = 0; i < numImages; i++) {
        indices[i] = i;
    }
    
    // Set up CUDA streams for overlapping operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Reset metrics
        float h_loss = 0.0f;
        int h_correct = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_correct, &h_correct, sizeof(int), cudaMemcpyHostToDevice));
        
        clock_t epoch_start = clock();
        
        // Shuffle data indices for this epoch
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
            
            // Pack all batch data at once (input and target together)
            for (int i = 0; i < current_batch_size; i++) {
                int idx = indices[start_idx + i];
                
                // Copy input data
                memcpy(h_batch_input + i * INPUT_SIZE, 
                       h_images[idx], 
                       INPUT_SIZE * sizeof(float));
                       
                // Copy target data
                memcpy(h_batch_target + i * OUTPUT_SIZE, 
                       h_labels[idx], 
                       OUTPUT_SIZE * sizeof(float));
            }
            
            // Combine transfers using a single stream for pipelining
            cudaMemcpyAsync(d_batch_input, h_batch_input,
                current_batch_size * INPUT_SIZE * sizeof(float), 
                cudaMemcpyHostToDevice, stream);
                
            cudaMemcpyAsync(d_batch_target, h_batch_target,
                current_batch_size * OUTPUT_SIZE * sizeof(float), 
                cudaMemcpyHostToDevice, stream);
                
            // Ensure transfers complete before proceeding
            cudaStreamSynchronize(stream);
                
            // Forward pass
            forwardBatch(net, d_batch_input, d_batch_hidden, d_batch_output, current_batch_size, stream);
            
            // Calculate metrics on GPU
            calculateBatchLossAccuracy<<<current_batch_size, BLOCK_SIZE, 0, stream>>>(
                d_batch_output, d_batch_target, d_loss, d_correct, current_batch_size);
            
            // Backward pass
            batchBackward(net, d_batch_input, d_batch_hidden, d_batch_output, d_batch_target, current_batch_size);
        }
        
        // Get final metrics
        CHECK_CUDA_ERROR(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
        
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, h_loss / numImages, (h_correct / (float)numImages) * 100,
               (float)(clock() - epoch_start) / CLOCKS_PER_SEC);
    }
    
    // Clean up
    cudaStreamDestroy(stream);
    cudaFree(d_batch_input);
    cudaFree(d_batch_hidden);
    cudaFree(d_batch_output);
    cudaFree(d_batch_target);
    cudaFree(d_loss);
    cudaFree(d_correct);
    cudaFreeHost(h_batch_data);  // Free the combined buffer
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

    freeNetwork(net);
    return 0;
}