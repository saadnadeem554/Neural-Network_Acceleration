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
TO BE DECIDED

# Team Members

## Saad Nadeem
**Github**: [Saadnadeem554](https://github.com/saadnadeem554)
## Hassaan Anwar
**Github**: [Hassaan-Anwar](https://github.com/Hassaan-Anwar)
## Mustafa Iqbal
**Github**: [Mustafaiqbal2](https://github.com/mustafaiqbal2)
