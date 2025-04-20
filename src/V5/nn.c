#include <openacc.h>   
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Activation functions
void relu(double* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax(double* x, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Neural network structure
typedef struct {
    double* W1;      // [HIDDEN_SIZE * INPUT_SIZE]
    double* b1;      // [HIDDEN_SIZE]
    double* W2;      // [OUTPUT_SIZE * HIDDEN_SIZE]
    double* b2;      // [OUTPUT_SIZE]

    // Device copies
    double* d_W1;
    double* d_b1;
    double* d_W2;
    double* d_b2;
} NeuralNetwork;

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));

    // Allocate and initialize host memory
    net->W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    net->b1 = (double*)malloc(HIDDEN_SIZE * sizeof(double));
    net->W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    net->b2 = (double*)malloc(OUTPUT_SIZE * sizeof(double));

    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++)
        net->W1[i] = ((double)rand() / RAND_MAX - 0.5);

    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->b1[i] = 0.0;

    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++)
        net->W2[i] = ((double)rand() / RAND_MAX - 0.5);

    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->b2[i] = 0.0;

    // Allocate device memory
    net->d_W1 = (double*)acc_malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    net->d_b1 = (double*)acc_malloc(HIDDEN_SIZE * sizeof(double));
    net->d_W2 = (double*)acc_malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    net->d_b2 = (double*)acc_malloc(OUTPUT_SIZE * sizeof(double));

    // Copy host data to device
    acc_memcpy_to_device(net->d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    acc_memcpy_to_device(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double));
    acc_memcpy_to_device(net->d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    acc_memcpy_to_device(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double));

    return net;
}


// Forward pass
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    double* d_input = (double*)acc_malloc(INPUT_SIZE * sizeof(double));
    double* d_hidden = (double*)acc_malloc(HIDDEN_SIZE * sizeof(double));
    double* d_output = (double*)acc_malloc(OUTPUT_SIZE * sizeof(double));

    acc_memcpy_to_device(d_input, input, INPUT_SIZE * sizeof(double));

    // Hidden layer: W1 * input + b1
    #pragma acc parallel loop deviceptr(net->d_W1, net->d_b1, d_input, d_hidden)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = net->d_b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += net->d_W1[i * INPUT_SIZE + j] * d_input[j];
        }
        d_hidden[i] = (sum > 0) ? sum : 0; // ReLU
    }

    relu(hidden, HIDDEN_SIZE);
    // Output layer: W2 * hidden + b2
    #pragma acc parallel loop deviceptr(net->d_W2, net->d_b2, d_hidden, d_output)
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double sum = net->d_b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += net->d_W2[i * HIDDEN_SIZE + j] * d_hidden[j];
        }
        d_output[i] = sum;
    }

    softmax(output, OUTPUT_SIZE);
 
    // Softmax (in-place)
    double sum = 0.0;
    #pragma acc parallel loop deviceptr(d_output) reduction(+:sum)
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        d_output[i] = exp(d_output[i]);
        sum += d_output[i];
    }

    #pragma acc parallel loop deviceptr(d_output)
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        d_output[i] /= sum;
    }

    // Copy results back to host
    acc_memcpy_from_device(hidden, d_hidden, HIDDEN_SIZE * sizeof(double));
    acc_memcpy_from_device(output, d_output, OUTPUT_SIZE * sizeof(double));

    // Free device buffers
    acc_free(d_input);
    acc_free(d_hidden);
    acc_free(d_output);
}

// Backpropagation
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

    // Compute output layer gradient: d_output = output - target
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        d_output[i] = output[i] - target[i];
    }

    // Compute hidden layer gradient
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_hidden[i] = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            d_hidden[i] += net->W2[j * HIDDEN_SIZE + i] * d_output[j];
        }
        // Derivative of ReLU
        d_hidden[i] *= (hidden[i] > 0);
    }

    // Update W2 weights and b2 biases
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_output[i] * hidden[j];
        }
        net->b2[i] -= LEARNING_RATE * d_output[i];
    }

    // Update W1 weights and b1 biases
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_hidden[i] * input[j];
        }
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
    }
    // Copy updated weights back to device
	acc_memcpy_to_device(net->d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
	acc_memcpy_to_device(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double));
	acc_memcpy_to_device(net->d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
	acc_memcpy_to_device(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double));

    
}

// Train network
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            forward(net, images[i], hidden, output);
            backward(net, images[i], hidden, output, labels[i]);

            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward(net, images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Read MNIST dataset
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
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


double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
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


// Free network memory
void freeNetwork(NeuralNetwork* net) {
    free(net->W1);
    free(net->b1);
    free(net->W2);
    free(net->b2);

    acc_free(net->d_W1);
    acc_free(net->d_b1);
    acc_free(net->d_W2);
    acc_free(net->d_b2);

    free(net);
}


// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    // Start measuring total execution time
    clock_t total_start = clock();

    // Measure time for loading data
    clock_t start = clock();
    double** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);
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

