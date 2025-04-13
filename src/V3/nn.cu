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

// CUDA kernel for ReLU activation
__global__ void reluKernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

// CUDA kernel for Softmax activation
__global__ void softmaxKernel(float* x, int size) {
    // Fixed size shared memory allocations
    __shared__ float shared_x[OUTPUT_SIZE];  // Assuming size <= OUTPUT_SIZE
    __shared__ float temp_max[BLOCK_SIZE];
    __shared__ float temp_sum[BLOCK_SIZE];
    __shared__ float shared_max;
    __shared__ float shared_sum;

    // Load input data into shared memory
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        shared_x[i] = x[i];
    }
    __syncthreads();

    // Find maximum value using shared memory
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        local_max = fmaxf(local_max, shared_x[i]);
    }

    // Reduce to find maximum
    temp_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            temp_max[threadIdx.x] = fmaxf(temp_max[threadIdx.x], temp_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        shared_max = temp_max[0];
    }
    __syncthreads();

    // Compute exp and sum using shared memory
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        shared_x[i] = expf(shared_x[i] - shared_max);
        local_sum += shared_x[i];
    }

    // Reduce to find sum
    temp_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            temp_sum[threadIdx.x] += temp_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        shared_sum = temp_sum[0];
    }
    __syncthreads();

    // Normalize using shared memory and write back to global memory
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        x[i] = shared_x[i] / shared_sum;
    }
}

// CUDA kernel for matrix multiplication (for forward pass)
__global__ void optimizedMatrixMulKernel(float* A, float* B, float* C, float* bias,int M, int N, int K) {
    // Block size (TILE_SIZE x TILE_SIZE)
    const int TILE_SIZE = 16;

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A into shared memory
        int A_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && A_col < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + A_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B into shared memory
        int B_row = t * TILE_SIZE + threadIdx.y;
        if (B_row < N && col < K) {
            Bs[threadIdx.y][threadIdx.x] = B[B_row * K + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded their data
        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result to C with bias
    if (row < M && col < K) {
        C[row * K + col] = sum + bias[row];
    }
}

// CUDA kernel for backward pass output layer
__global__ void backwardOutputKernel(float* d_output, float* d_target,
                                   float* d_output_error, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output_error[idx] = d_output[idx] - d_target[idx];
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
// Helper function to check if file exists and has correct size
bool verifyMNISTFile(const char* filename, long expectedSize) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return false;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fclose(file);
    
    if (fileSize != expectedSize) {
        fprintf(stderr, "Error: File %s has incorrect size. Expected %ld bytes, got %ld bytes\n",
                filename, expectedSize, fileSize);
        return false;
    }
    return true;
}

// Helper function to verify all MNIST files
void verifyMNISTFiles() {
    const char* files[] = {
        "train-images.idx3-ubyte",
        "train-labels.idx1-ubyte",
        "t10k-images.idx3-ubyte",
        "t10k-labels.idx1-ubyte"
    };
    long sizes[] = {
        47040016,  // 60000 * 784 + 16 (header)
        60008,     // 60000 + 8 (header)
        7840016,   // 10000 * 784 + 16 (header)
        10008      // 10000 + 8 (header)
    };
    
    for (int i = 0; i < 4; i++) {
        if (!verifyMNISTFile(files[i], sizes[i])) {
            fprintf(stderr, "\nPlease ensure the MNIST dataset files are in the 'data' directory:\n");
            fprintf(stderr, "1. Create a 'data' directory in your project root\n");
            fprintf(stderr, "2. Download the MNIST dataset files:\n");
            fprintf(stderr, "   wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\n");
            fprintf(stderr, "   wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz\n");
            fprintf(stderr, "   wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz\n");
            fprintf(stderr, "   wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz\n");
            fprintf(stderr, "3. Extract the files:\n");
            fprintf(stderr, "   gunzip *.gz\n");
            fprintf(stderr, "4. Move the files to the data directory:\n");
            fprintf(stderr, "   mv *ubyte data/\n");
            exit(EXIT_FAILURE);
        }
    }
}
// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Allocate host memory
    net->h_W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    net->h_W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    net->h_b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    net->h_b2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    // Initialize weights with random values
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++)
        net->h_W1[i] = ((float)rand() / RAND_MAX) * 0.01f;
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++)
        net->h_W2[i] = ((float)rand() / RAND_MAX) * 0.01f;
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
// Add these kernel functions for gradient computation
__global__ void computeGradients(float* d_output, float* d_target, float* d_hidden,
    float* d_input, float* d_W2_grad, float* d_b2_grad,float* d_W1_grad, float* d_b1_grad,float* d_W2, float* d_W1,int M, int N, int K) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < OUTPUT_SIZE) {
        // Output layer gradients
        float output_error = d_output[idx] - d_target[idx];
        d_b2_grad[idx] = output_error;

        for (int j = 0; j < HIDDEN_SIZE; j++) {
            atomicAdd(&d_W2_grad[idx * HIDDEN_SIZE + j], output_error * d_hidden[j]);
        }
    }

    if (idx < HIDDEN_SIZE) {
        // Hidden layer gradients
        float hidden_error = 0.0f;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_error += (d_output[j] - d_target[j]) * d_W2[j * HIDDEN_SIZE + idx];
        }
        hidden_error *= (d_hidden[idx] > 0.0f); // ReLU derivative

        d_b1_grad[idx] = hidden_error;
        for (int j = 0; j < INPUT_SIZE; j++) {
            atomicAdd(&d_W1_grad[idx * INPUT_SIZE + j], hidden_error * d_input[j]);
        }
    }
}

__global__ void updateParameters(float* d_W, float* d_b, float* d_W_grad, float* d_b_grad, int rows, int cols, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows * cols) {
    d_W[idx] -= learning_rate * d_W_grad[idx];
    }

    if (idx < rows) {
    d_b[idx] -= learning_rate * d_b_grad[idx];
    }
}

// Update the backward function
void backward(NeuralNetwork* net, float* d_input, float* d_hidden, 
    float* d_output, float* d_target) {
        // Allocate gradient matrices
    float *d_W1_grad, *d_W2_grad, *d_b1_grad, *d_b2_grad;
    CHECK_CUDA_ERROR(cudaMalloc(&d_W1_grad, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_W2_grad, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b1_grad, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b2_grad, OUTPUT_SIZE * sizeof(float)));
    
    // Initialize gradients to zero
    cudaMemset(d_W1_grad, 0, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMemset(d_W2_grad, 0, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMemset(d_b1_grad, 0, HIDDEN_SIZE * sizeof(float));
    cudaMemset(d_b2_grad, 0, OUTPUT_SIZE * sizeof(float));
    
    // Compute gradients
    int blockSize = 256;
    int numBlocks = (max(OUTPUT_SIZE, HIDDEN_SIZE) + blockSize - 1) / blockSize;
    
    computeGradients<<<numBlocks, blockSize>>>(
        d_output, d_target, d_hidden, d_input, 
        d_W2_grad, d_b2_grad, d_W1_grad, d_b1_grad,
        net->d_W2, net->d_W1, HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE);

    updateParameters<<<(HIDDEN_SIZE * INPUT_SIZE + blockSize - 1) / blockSize, blockSize>>>(
            net->d_W1, net->d_b1, d_W1_grad, d_b1_grad, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
    
    updateParameters<<<(OUTPUT_SIZE * HIDDEN_SIZE + blockSize - 1) / blockSize, blockSize>>>(
            net->d_W2, net->d_b2, d_W2_grad, d_b2_grad, OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    
    
    // Free gradient memory
    cudaFree(d_W1_grad);
    cudaFree(d_W2_grad);
    cudaFree(d_b1_grad);
    cudaFree(d_b2_grad);
    }
// Forward pass
void forward(NeuralNetwork* net, float* d_input, float* d_hidden, float* d_output) {
    // Configure block and grid dimensions
    const int TILE_SIZE = 16;
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    
    // Hidden layer
    dim3 gridDim((1 + blockDim.x - 1) / blockDim.x,
                 (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y);
    optimizedMatrixMulKernel<<<gridDim, blockDim>>>(net->d_W1, d_input, d_hidden, 
                                                  net->d_b1, HIDDEN_SIZE, INPUT_SIZE, 1);
    reluKernel<<<(HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_hidden, HIDDEN_SIZE);

    // Output layer
    gridDim.x = (1 + blockDim.x - 1) / blockDim.x;
    gridDim.y = (OUTPUT_SIZE + blockDim.y - 1) / blockDim.y;
    optimizedMatrixMulKernel<<<gridDim, blockDim>>>(net->d_W2, d_hidden, d_output,
                                                  net->d_b2, OUTPUT_SIZE, HIDDEN_SIZE, 1);
    softmaxKernel<<<1, BLOCK_SIZE>>>(d_output, OUTPUT_SIZE);
}
// Update the train function to include loss calculation
void train(NeuralNetwork* net, float** h_images, float** h_labels, int numImages) {
float *d_input, *d_hidden, *d_output, *d_target;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_target, OUTPUT_SIZE * sizeof(float)));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float loss = 0.0f;
        int correct = 0;
        clock_t epoch_start = clock();

        for (int i = 0; i < numImages; i++) {
            // Copy current training example to device
            CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_images[i],
            INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_target, h_labels[i],
            OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

            // Forward and backward passes
            forward(net, d_input, d_hidden, d_output);
            backward(net, d_input, d_hidden, d_output, d_target);

            // Compute accuracy and loss
            float h_output[OUTPUT_SIZE];
            float h_target[OUTPUT_SIZE];
            CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output,
            OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(h_target, d_target,
            OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            // Calculate cross-entropy loss
            for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_target[j] > 0.5f) {  // This is the correct class
            loss -= log(fmax(h_output[j], 1e-7f));
            }
            }

            // Calculate accuracy
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[pred]) pred = j;
            if (h_target[j] > h_target[actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        // Print epoch statistics
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
        epoch + 1, loss / numImages, (correct / (float)numImages) * 100,
        (float)(clock() - epoch_start) / CLOCKS_PER_SEC);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_target);
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
// Evaluate network performance
void evaluate(NeuralNetwork* net, float** h_images, float** h_labels, int numImages) {
    float *d_input, *d_hidden, *d_output;
    int correct = 0;

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float)));

    for (int i = 0; i < numImages; i++) {
        // Copy input image to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_images[i], 
            INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        // Forward pass
        forward(net, d_input, d_hidden, d_output);

        // Copy output back to host
        float h_output[OUTPUT_SIZE];
        CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, 
            OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        // Find predicted and actual class
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[pred]) pred = j;
            if (h_labels[i][j] > h_labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);

    printf("Test Accuracy: %.2f%%\n", (correct / (float)numImages) * 100);
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
    // Verify MNIST files exist and have correct sizes
    verifyMNISTFiles();
    // Start measuring total execution time
    clock_t total_start = clock();

    // Measure time for loading data
    clock_t start = clock();
    float** train_images = loadMNISTImages("train-images.idx3-ubyte", 60000);
    float** train_labels = loadMNISTLabels("train-labels.idx1-ubyte", 60000);
    float** test_images = loadMNISTImages("t10k-images.idx3-ubyte", 10000);
    float** test_labels = loadMNISTLabels("t10k-labels.idx1-ubyte", 10000);
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
