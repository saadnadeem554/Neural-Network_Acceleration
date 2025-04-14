# Neural Network Acceleration on GPUs

This project focuses on accelerating a neural network for MNIST digit classification using CUDA on GPUs. We implement several versions with increasing optimization to demonstrate performance improvements.

## Project Overview

We start with a baseline CPU implementation of a simple neural network for classifying handwritten digits (MNIST dataset) and progressively optimize it for GPU execution using CUDA. The project demonstrates various optimization techniques including naive parallelization, memory optimizations, and utilization of tensor cores.

## Project Structure

```
Neural-Network_Acceleration/
├── src/
│   ├── V1/        # Sequential Implementation
│   ├── V2/        # TO BE IMPLEMENTED
│   ├── V3/        # TO BE IMPLEMENTED
│   ├── V4/        # TO BE IMPLEMENTED
│   
├── report/        # Project report
├── slides/        # Presentation slides
└── data/      # MNIST dataset files
└── README.md      # This file
```

## Dataset

The MNIST dataset consists of:
- 60,000 training images
- 10,000 test images
- Each image is 28x28 grayscale (784 pixels)
- 10 classes (digits 0-9)

## Installation and Setup

### Prerequisites
- CUDA toolkit (v11.0 or higher recommended)
- GCC compiler
- Make

## Compilation and Execution

Navigate to the src directory to run all commands:

```bash
cd src
```

### V1: Sequential Implementation
```bash
make v1
```

### V2: Naive GPU Implementation
```bash
make v2
```

### V3: Optimized GPU Implementation
```bash
make v3
```

### V4: Tensor Core Implementation
```bash
make v4
```

### Additional Make Commands
- Clean all executables:
  ```bash
  make clean
  ```

## Implementation Details
# MNIST Neural Network - V2 (Naive GPU Implementation)

## Overview
This is a naive GPU implementation of a neural network for MNIST digit classification using CUDA. This version represents the first step in transitioning from CPU to GPU computation, serving as a baseline for further optimizations.


### Architecture
- Input Layer: 784 neurons (28x28 pixels)
- Hidden Layer: 128 neurons with ReLU activation
- Output Layer: 10 neurons with Softmax activation
- Learning Rate: 0.01
- Epochs: 3
- Batch Size: 64

### Key Components
1. **Neural Network Structure**
   - Device (GPU) weights and biases
   - Host (CPU) copies of weights and biases
   - Basic CUDA kernels for matrix operations

2. **CUDA Kernels**
   - `reluKernel`: ReLU activation function
   - `softmaxKernel`: Softmax activation function
   - `matrixMulKernel`: Basic matrix multiplication
   - `backwardOutputKernel`: Gradient computation

3. **Memory Management**
   - Basic GPU memory allocation and deallocation
   - Simple host-to-device and device-to-host transfers

## Performance Metrics
- Training Time: ~36.47 seconds
- Test Accuracy: 96.80%
- Training Accuracy: 97.92% (after 3 epochs)
- Data Loading Time: ~1.67 seconds
- Total Execution Time: ~38.73 seconds

### Epoch-wise Performance
- Epoch 1: 91.93% accuracy (12.239s)
- Epoch 2: 96.99% accuracy (12.020s)
- Epoch 3: 97.92% accuracy (12.004s)

## Known Limitations
- Basic memory transfer pattern without optimization
- No use of shared memory
- Limited parallelization in backward pass
- Single-stream execution
- No batching optimization
- Simple kernel configurations

## Implementation Details
# MNIST Neural Network - V3 (Optimised GPU Implementation)

## Overview
This version builds upon V2 by introducing shared memory and other optimizations

## Performance Metrics
- Training Time: ~35.82 seconds
- Test Accuracy: 96.97%
- Training Accuracy: 97.77% (after 3 epochs)
- Data Loading Time: ~1.67 seconds
- Total Execution Time: ~37.83 seconds

### Epoch-wise Performance
- Epoch 1: 91.91% accuracy (11.970s)
- Epoch 2: 96.86% accuracy (11.828s)
- Epoch 3: 97.77% accuracy (11.821s)

# Team Members

## Saad Nadeem
**Github**: [Saadnadeem554](https://github.com/saadnadeem554)
## Hassaan Anwar
**Github**: [Hassaan-Anwar](https://github.com/Hassaan-Anwar)
## Mustafa Iqbal
**Github**: [Mustafaiqbal2](https://github.com/mustafaiqbal2)
