# ⚡ Neural Network Acceleration on GPUs

This project focuses on **accelerating a neural network for MNIST digit classification using CUDA**. Starting from a baseline sequential implementation, we progressively optimize the architecture using GPU computing techniques — from naive parallelization to Tensor Core acceleration with cuBLAS and OpenACC.

---

## 🧠 Project Overview

The MNIST dataset (handwritten digits 0–9) is used as a benchmark for evaluating various neural network implementations. The goal is to **compare multiple versions** of a feedforward neural network to assess improvements in:

- Execution time (training + evaluation)
- Model accuracy
- GPU utilization and profiling metrics

We implement 6 versions:
- **V1**: Sequential CPU
- **V2**: Naive CUDA
- **V3**: Optimized CUDA (with batching, kernel fusion)
- **V4**: Tensor Core using WMMA
- **V5**: OpenACC-based acceleration
- **V6**: cuBLAS + Tensor Core (TF32) acceleration

---

## 📁 Project Structure

```bash
Neural-Network_Acceleration/
├── src/
│   ├── V1/            # Sequential C implementation
│   ├── V2/            # Naive CUDA version
│   ├── V3/            # Optimized CUDA with batching and kernel fusion
│   ├── V4/            # Tensor Core using WMMA (FP16)
│   ├── V5/            # OpenACC accelerated version
│   ├── V6/            # cuBLAS + TF32 Tensor Core acceleration
│
├── data/              # Raw MNIST dataset (.ubyte files)
├── report/            # Detailed report (PDF)
├── slides/            # Final presentation slides (PPTX)
└── README.md          # This file
```

## 📦 Dataset

We use the official [MNIST dataset](http://yann.lecun.com/exdb/mnist/) containing:

- **60,000** training images
- **10,000** test images
- **784 features per image** (28×28 grayscale)
- **10 output classes** (digits 0–9)

Ensure the extracted `.ubyte` files are placed in the `data/` directory.

## ⚙️ Setup & Installation

### 🖥 Prerequisites

- NVIDIA GPU with CUDA support (Compute Capability ≥ 7.0 recommended)
- CUDA Toolkit (v11 or newer)
- `make`, `gcc`, `nvcc`, and OpenACC (e.g., PGI or NVIDIA HPC compiler)
- cuBLAS (included with CUDA)

### 🔧 Compile & Run

From the root directory:

```bash
cd src
```

To build and run a specific version:

```bash
make v1      # Sequential
make v2      # Naive CUDA
make v3      # Optimized CUDA
make v4      # Tensor Core (WMMA)
make v5      # OpenACC
make v6      # cuBLAS + Tensor Core
```

To clean all builds:

```bash
make clean
```

## 🚀 Implementations & Highlights

### ✅ V1: Sequential (Baseline)
- Pure C implementation with no parallelism
- Single hidden layer (128 neurons, ReLU)
- Softmax output, SGD
- Execution Time: 23.48s
- Test Accuracy: ~97%

### ✅ V2: Naive CUDA
- GPU offload of forward & backward passes
- Kernels: matrixMul, relu, softmax, updateParameters
- Operates per-sample (no batching)
- Execution Time: 37.63s (slower than CPU 😬)
- Test Accuracy: 96.66%
- Speedup: 0.62×

🔍 Bottlenecks:
- Excessive memory transfers
- atomicAdd overhead
- Kernel launch latency

### ✅ V3: Optimized CUDA
- Introduces batching, double buffering, and stream-based updates
- Fused kernels, shared memory, warp-level reductions
- Execution Time: 0.675s
- Test Accuracy: 91.25%
- Speedup: 34.79×

📊 Batch Size Tuning:

| Batch Size | Time  | Accuracy |
|------------|-------|----------|
| 4          | 5.56s | 96.43%   |
| 32         | 0.87s | 92.52%   |
| 64         | 0.59s | 91.25%   |
| 1024       | 0.35s | 75.62%   |

### ✅ V4: Tensor Core via WMMA
- Uses FP16 matrix multiplication via wmma::mma_sync
- Manual kernel for __half inputs
- Very fast but accuracy drops due to quantization
- Result: Not suitable without mixed precision/loss scaling

### ✅ V5: OpenACC
- Forward pass parallelized with #pragma acc
- Softmax and memory managed via OpenACC
- Backward pass remains on CPU
- Easy to implement but limited acceleration

### ✅ V6: cuBLAS + Tensor Cores (TF32)
- Uses cublasGemmEx with CUBLAS_TF32_TENSOR_OP_MATH
- Combined bias + ReLU kernels
- Shared memory for loss/accuracy aggregation
- Execution Time: 1.033s
- Test Accuracy: 91.30%
- Speedup: 22.73×

## 📈 Version Comparison

| Version | Time (s) | Speedup | Accuracy |
|---------|----------|---------|----------|
| V1      | 23.48    | 1.00×   | ~97%     |
| V2      | 37.63    | 0.62×   | 96.66%   |
| V3      | 0.675    | 34.79×  | 91.25%   |
| V6      | 1.033    | 22.73×  | 91.30%   |

## 💡 Learnings & Takeaways

- Batching is essential for GPU throughput
- cuBLAS outperforms manual kernels for larger matrices
- Tensor Cores require precision trade-offs (FP16/TF32)
- OpenACC simplifies code but offers limited control
- Memory layout, stream management, and kernel fusion = key performance levers

## 🧑‍💻 Team

| Name           | GitHub           |
|----------------|------------------|
| Saad Nadeem   | @saadnadeem554  |
| Mustafa Iqbal | @Mustafaiqbal2  |
| Hassaan Anwar | @Hassaan-Anwar  |

## 📎 References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/index.html)
- [OpenACC Standard](https://www.openacc.org/)

## 📄 Report and Slides

- [📘 Project Report (PDF)](./report/report.pdf)
- [📊 Presentation Slides (PPTX)](./slides/presentation.pptx)

## github repo link: https://github.com/Mustafaiqbal2/Neural-Network_Acceleration