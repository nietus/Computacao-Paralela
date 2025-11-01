# MNIST Multi-Layer Perceptron - Implementation Guide

## Overview

This project contains four implementations of a neural network for MNIST digit classification (784→128→10 architecture), each using different parallelization strategies:

1. **mnist_mlp.c** - Sequential baseline (CPU only)
2. **mnist_mlp_openmp_cpu.c** - OpenMP CPU parallelization (16 threads)
3. **mnist_mlp_openmp_gpu.c** - OpenMP GPU offloading
4. **mnist_mlp_cuda.cu** - CUDA GPU acceleration with persistent memory

All implementations produce **identical results** (within floating-point precision) but with different execution times.

---

## Network Architecture

- **Input Layer**: 784 neurons (28×28 pixel images)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation
- **Training**: Mini-batch SGD (batch size 64)
- **Loss Function**: Cross-entropy loss
- **Optimization**: Stochastic Gradient Descent (SGD)
- **Learning Rate**: 0.01
- **Weight Initialization**: Xavier/Glorot initialization

---

## Implementation Details

### 1. mnist_mlp.c - Sequential Baseline

**Purpose**: Reference implementation for correctness and performance comparison.

**Key Features**:
- Standard C implementation with no parallelization
- 2D weight arrays: `weights[i][j]`
- Comprehensive timing: data loading, training, testing, total execution
- Logs training metrics to `./logs/training_loss_c.txt`

**Data Structures**:
```c
typedef struct {
    int input_size;
    int output_size;
    double **weights;      // 2D array
    double *biases;
    ActivationType activation;
} LinearLayer;

typedef struct {
    LinearLayer hidden_layer;
    LinearLayer output_layer;
} NeuralNetwork;
```

**Compilation**:
```bash
gcc -O3 -o mnist_mlp mnist_mlp.c -lm
```

**Expected Performance**: Baseline (slowest)

---

### 2. mnist_mlp_openmp_cpu.c - OpenMP CPU Parallelization

**Purpose**: Multi-core CPU parallelization using OpenMP directives.

**Key Features**:
- **16 OpenMP threads** for parallel execution
- Parallelized operations:
  - Forward propagation (matrix-vector multiplication)
  - Activation functions (element-wise)
  - Backpropagation (error computation)
  - Weight/bias updates (element-wise)
  - Weight initialization
- 2D weight arrays: `weights[i][j]`
- Logs to `./logs/training_loss_openmp_cpu.txt`

**Parallelization Strategy**:
```c
// Forward pass - parallelize over output neurons
#pragma omp parallel for
for (int i = 0; i < layer->output_size; i++) {
    double activation_sum = layer->biases[i];
    for (int j = 0; j < layer->input_size; j++) {
        activation_sum += inputs[j] * layer->weights[j][i];
    }
    outputs[i] = activation_sum;
}

// Weight updates - parallelize nested loops
#pragma omp parallel for collapse(2)
for (int i = 0; i < layer->input_size; i++) {
    for (int j = 0; j < layer->output_size; j++) {
        layer->weights[i][j] -= LEARNING_RATE * deltas[j] * inputs[i];
    }
}
```

**Thread Configuration**:
- Set in `main()`: `omp_set_num_threads(16);`
- Automatically distributes work across 16 CPU cores

**Compilation**:
```bash
gcc -fopenmp -O3 -o mnist_mlp_openmp_cpu mnist_mlp_openmp_cpu.c -lm
```

**Expected Performance**: 5-10× faster than baseline (depends on CPU cores)

---

### 3. mnist_mlp_openmp_gpu.c - OpenMP GPU Offloading

**Purpose**: GPU acceleration using OpenMP target directives.

**Key Features**:
- GPU offloading via `#pragma omp target` directives
- Same algorithm as baseline, but runs on GPU
- 2D weight arrays: `weights[i][j]`
- Automatic data transfer management by OpenMP runtime
- Logs to `./logs/training_loss_openmp_gpu.txt`

**GPU Offloading Strategy**:
```c
// Forward pass - offload to GPU
#pragma omp target teams distribute parallel for
for (int i = 0; i < layer->output_size; i++) {
    double activation_sum = layer->biases[i];
    for (int j = 0; j < layer->input_size; j++) {
        activation_sum += inputs[j] * layer->weights[j][i];
    }
    outputs[i] = activation_sum;
}

// ReLU activation - offload to GPU
#pragma omp target teams distribute parallel for
for (int i = 0; i < layer->output_size; i++) {
    outputs[i] = outputs[i] > 0 ? outputs[i] : 0;
}

// Weight updates - offload to GPU
#pragma omp target teams distribute parallel for collapse(2)
for (int i = 0; i < layer->input_size; i++) {
    for (int j = 0; j < layer->output_size; j++) {
        layer->weights[i][j] -= LEARNING_RATE * deltas[j] * inputs[i];
    }
}
```

**Data Transfer**:
- OpenMP runtime handles memory transfers automatically
- Each operation triggers CPU↔GPU data movement
- Not optimal for small networks, but simple to implement

**Compilation**:
```bash
# For NVIDIA GPUs
gcc -fopenmp -foffload=nvptx-none -O3 -o mnist_mlp_openmp_gpu mnist_mlp_openmp_gpu.c -lm

# For AMD GPUs
gcc -fopenmp -foffload=amdgcn-amdhsa -O3 -o mnist_mlp_openmp_gpu mnist_mlp_openmp_gpu.c -lm

# Note: Requires compiler with GPU offloading support (GCC 12+, Clang, etc.)
```

**Expected Performance**: Variable (may be slower than CPU due to transfer overhead for small networks)

---

### 4. mnist_mlp_cuda.cu - CUDA GPU Acceleration

**Purpose**: Optimized GPU implementation with persistent memory and custom CUDA kernels.

**Key Features**:
- **Persistent GPU memory**: Weights stay on GPU throughout training
- **Custom CUDA kernels** for all operations
- **Flattened weight arrays**: `weights[i * output_size + j]` for coalesced memory access
- **Minimal CPU↔GPU transfers**: Only inputs/outputs transferred per sample
- **Error checking**: `CUDA_CHECK()` macro for debugging
- Logs to `./logs/training_loss_cuda.txt`

**Data Structures**:
```c
typedef struct {
    int input_size;
    int output_size;
    double *weights;      // Flattened: [input_size * output_size]
    double *biases;
    ActivationType activation;

    // GPU pointers (persistent)
    double *d_weights;    // Device weights
    double *d_biases;     // Device biases
} LinearLayer;

typedef struct {
    LinearLayer hidden_layer;
    LinearLayer output_layer;

    // GPU working memory (persistent)
    double *d_hidden_outputs;
    double *d_output_outputs;
    double *d_delta_hidden;
    double *d_delta_output;
    double *d_expected_output;
    double *d_input;
} NeuralNetwork;
```

**CUDA Kernels**:

1. **linear_forward_kernel**: Matrix-vector multiplication
```c
__global__ void linear_forward_kernel(double *inputs, double *weights,
                                      double *biases, double *outputs,
                                      int input_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < output_size) {
        double sum = biases[i];
        for (int j = 0; j < input_size; j++) {
            sum += inputs[j] * weights[j * output_size + i];
        }
        outputs[i] = sum;
    }
}
```

2. **apply_relu_kernel**: ReLU activation
```c
__global__ void apply_relu_kernel(double *data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] = data[i] > 0 ? data[i] : 0;
    }
}
```

3. **output_delta_kernel**: Compute output layer gradients
```c
__global__ void output_delta_kernel(double *outputs, double *expected,
                                    double *delta, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        delta[i] = outputs[i] - expected[i];
    }
}
```

4. **hidden_delta_kernel**: Compute hidden layer gradients
```c
__global__ void hidden_delta_kernel(double *delta_output, double *weights,
                                     double *hidden_outputs, double *delta_hidden,
                                     int num_hidden, int num_outputs, int use_relu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_hidden) {
        double error = 0.0;
        for (int j = 0; j < num_outputs; j++) {
            error += delta_output[j] * weights[i * num_outputs + j];
        }
        double derivative = use_relu ?
            (hidden_outputs[i] > 0 ? 1.0 : 0.0) :
            (hidden_outputs[i] * (1.0 - hidden_outputs[i]));
        delta_hidden[i] = error * derivative;
    }
}
```

5. **update_weights_kernel**: Update weights
```c
__global__ void update_weights_kernel(double *weights, double *inputs,
                                       double *deltas, int input_size,
                                       int output_size, double lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // input index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // output index

    if (i < input_size && j < output_size) {
        weights[i * output_size + j] -= lr * deltas[j] * inputs[i];
    }
}
```

6. **update_biases_kernel**: Update biases
```c
__global__ void update_biases_kernel(double *biases, double *deltas,
                                      int size, double lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        biases[i] -= lr * deltas[i];
    }
}
```

**Memory Management**:
```c
// Initialization (once at startup)
void initialize_layer(LinearLayer *layer, ...) {
    // Allocate host memory
    layer->weights = malloc(...);
    layer->biases = malloc(...);

    // Initialize weights (Xavier)
    // ...

    // Allocate GPU memory (persistent)
    cudaMalloc(&layer->d_weights, ...);
    cudaMalloc(&layer->d_biases, ...);

    // Copy to GPU once
    cudaMemcpy(layer->d_weights, layer->weights, ..., cudaMemcpyHostToDevice);
    cudaMemcpy(layer->d_biases, layer->biases, ..., cudaMemcpyHostToDevice);
}

// Training loop (per sample)
void forward_gpu(NeuralNetwork *nn, double *input, ...) {
    // Copy input to GPU
    cudaMemcpy(nn->d_input, input, NUM_INPUTS * sizeof(double),
               cudaMemcpyHostToDevice);

    // Run kernels (data stays on GPU)
    linear_forward_kernel<<<...>>>(nn->d_input, nn->hidden_layer.d_weights, ...);
    apply_relu_kernel<<<...>>>(nn->d_hidden_outputs, NUM_HIDDEN);
    linear_forward_kernel<<<...>>>(nn->d_hidden_outputs, nn->output_layer.d_weights, ...);

    // Copy output back (for softmax on CPU)
    cudaMemcpy(output_out, nn->d_output_outputs, ..., cudaMemcpyDeviceToHost);
}
```

**Kernel Launch Configuration**:
```c
// 1D kernels (activation, deltas)
int blockSize = 256;
int numBlocks = (size + blockSize - 1) / blockSize;
kernel<<<numBlocks, blockSize>>>(...);

// 2D kernels (weight updates)
dim3 blockDim(16, 16);
dim3 gridDim((input_size + 15) / 16, (output_size + 15) / 16);
kernel<<<gridDim, blockDim>>>(...);
```

**Compilation**:
```bash
nvcc -O3 -o mnist_mlp_cuda mnist_mlp_cuda.cu -lm
```

**Expected Performance**: Fastest (5-20× faster than baseline, depends on GPU)

**Why It's Fast**:
1. **Persistent GPU memory**: No reallocation overhead
2. **Flattened arrays**: Coalesced memory access
3. **Minimal transfers**: Only 784 doubles in, 10 doubles out per sample
4. **Optimized kernels**: Tailored to neural network operations
5. **Concurrent execution**: Multiple kernels can run simultaneously

---

## Performance Comparison

### Expected Speedups (Relative to Baseline)

| Implementation | Expected Speedup | Notes |
|----------------|------------------|-------|
| **mnist_mlp.c** | 1.0× (baseline) | Single-threaded CPU |
| **mnist_mlp_openmp_cpu.c** | 5-10× | Depends on CPU cores (16 threads) |
| **mnist_mlp_openmp_gpu.c** | 0.5-3× | May be slower due to transfer overhead |
| **mnist_mlp_cuda.cu** | 10-50× | Best performance, scales with GPU |

### Timing Breakdown

All implementations report:
```
Data loading time: X.XX seconds
Epoch 1, Loss: X.XXXXXX Time: X.XX
Epoch 2, Loss: X.XXXXXX Time: X.XX
...
Total training time: X.XX seconds
Testing time: X.XX seconds
Total program time: X.XX seconds
```

---

## Compilation and Execution

### Compile All Versions

```bash
# Baseline
gcc -O3 -o mnist_mlp mnist_mlp.c -lm

# OpenMP CPU
gcc -fopenmp -O3 -o mnist_mlp_openmp_cpu mnist_mlp_openmp_cpu.c -lm

# OpenMP GPU (NVIDIA)
gcc -fopenmp -foffload=nvptx-none -O3 -o mnist_mlp_openmp_gpu mnist_mlp_openmp_gpu.c -lm

# CUDA
nvcc -O3 -o mnist_mlp_cuda mnist_mlp_cuda.cu -lm
```

### Run All Versions

```bash
# Ensure data directory exists
mkdir -p logs

# Run baseline
./mnist_mlp

# Run OpenMP CPU
./mnist_mlp_openmp_cpu

# Run OpenMP GPU (if GPU available)
./mnist_mlp_openmp_gpu

# Run CUDA (if CUDA toolkit installed)
./mnist_mlp_cuda
```

---

## Output Files

Each version creates a log file in `./logs/`:

| File | Contents |
|------|----------|
| `training_loss_c.txt` | Baseline training metrics |
| `training_loss_openmp_cpu.txt` | OpenMP CPU training metrics |
| `training_loss_openmp_gpu.txt` | OpenMP GPU training metrics |
| `training_loss_cuda.txt` | CUDA training metrics |

**Log Format**:
```
epoch,loss,time
1,0.523456,12.34
2,0.412345,11.98
...
```

---

## Correctness Verification

All implementations should produce:
- **~97-98% test accuracy** (±0.5%)
- **Similar loss curves** (within 1e-4)
- **Same convergence behavior**

To compare results:
```bash
# Compare accuracies
grep "Test Accuracy" *.log

# Compare loss curves
paste logs/training_loss_c.txt logs/training_loss_cuda.txt | column -t
```

---

## Key Differences Summary

| Feature | Baseline | OpenMP CPU | OpenMP GPU | CUDA |
|---------|----------|------------|------------|------|
| **Parallelism** | None | CPU multi-threading | GPU offloading | GPU custom kernels |
| **Weight Storage** | 2D arrays | 2D arrays | 2D arrays | Flattened 1D |
| **Memory Mgmt** | CPU only | CPU only | Auto by OpenMP | Manual GPU mgmt |
| **Data Transfer** | N/A | N/A | Per operation | Per sample |
| **Persistence** | N/A | N/A | No | Yes (GPU memory) |
| **Optimization** | -O3 | -O3 + OpenMP | -O3 + OpenMP | -O3 + kernels |
| **Complexity** | Low | Medium | Medium | High |
| **Performance** | Baseline | 5-10× | Variable | 10-50× |

---

## Parallelization Strategies Explained

### 1. Data Parallelism (OpenMP CPU)
- **What**: Distribute loop iterations across multiple CPU threads
- **Where**: Forward pass, backward pass, weight updates
- **How**: `#pragma omp parallel for`
- **Benefit**: Utilizes multiple CPU cores simultaneously

### 2. Task Offloading (OpenMP GPU)
- **What**: Move entire computations to GPU
- **Where**: Same loops as OpenMP CPU
- **How**: `#pragma omp target teams distribute parallel for`
- **Benefit**: GPU's massive parallelism for embarrassingly parallel tasks

### 3. Custom Kernel Optimization (CUDA)
- **What**: Write specialized GPU functions for each operation
- **Where**: All computations (forward, backward, updates)
- **How**: Custom `__global__` CUDA kernels
- **Benefit**: Fine-grained control, minimal overhead, persistent memory

---

## Common Issues and Solutions

### OpenMP CPU
**Issue**: Not using all cores
**Solution**: Set `export OMP_NUM_THREADS=16` or verify with `omp_get_num_threads()`

### OpenMP GPU
**Issue**: Slower than CPU
**Reason**: Transfer overhead dominates for small networks
**Solution**: Use CUDA for better control, or increase network size

### CUDA
**Issue**: "CUDA error: out of memory"
**Solution**: Reduce `TRAIN_SAMPLES` or `TEST_SAMPLES` in defines

**Issue**: "undefined reference to cudaMalloc"
**Solution**: Use `nvcc` instead of `gcc`, include `-lcudart`

**Issue**: Incorrect results
**Check**:
1. Same random seed (`srand(time(NULL))`)
2. Correct weight indexing in flattened arrays
3. Proper memory transfers (Host→Device, Device→Host)

---

## Technical Details

### Neural Network Operations

**Forward Pass**:
1. Hidden: `h = ReLU(W₁ᵀx + b₁)` where x ∈ ℝ⁷⁸⁴, h ∈ ℝ¹²⁸
2. Output: `y = Softmax(W₂ᵀh + b₂)` where y ∈ ℝ¹⁰

**Backward Pass**:
1. Output delta: `δ₂ = y - y_true` (softmax + cross-entropy)
2. Hidden delta: `δ₁ = (W₂δ₂) ⊙ ReLU'(h)`
3. Weight gradients: `∇W₂ = hδ₂ᵀ`, `∇W₁ = xδ₁ᵀ`
4. Update: `W ← W - η∇W`, `b ← b - ηδ`

### Computational Complexity

| Operation | Baseline | Parallel | Speedup |
|-----------|----------|----------|---------|
| **Forward (W₁x)** | O(784×128) | O(784×128/P) | P |
| **Forward (W₂h)** | O(128×10) | O(128×10/P) | P |
| **Backward** | O(784×128 + 128×10) | O((784×128 + 128×10)/P) | P |
| **Weight Update** | O(784×128 + 128×10) | O((784×128 + 128×10)/P) | P |

Where P = number of parallel units (16 for OpenMP CPU, thousands for GPU)

### Memory Usage

| Component | Size |
|-----------|------|
| **Weights (W₁)** | 784 × 128 × 8 = 784 KB |
| **Weights (W₂)** | 128 × 10 × 8 = 10 KB |
| **Training Data** | 20000 × 784 × 8 ≈ 122 MB |
| **Test Data** | 5000 × 784 × 8 ≈ 31 MB |
| **Total** | ~154 MB |

**CUDA Additional**:
- GPU weights: ~794 KB
- GPU working memory: (128 + 10 + 128 + 10 + 10 + 784) × 8 ≈ 8 KB
- Total GPU: ~800 KB (easily fits in any modern GPU)

---

## References

- **OpenMP**: https://www.openmp.org/
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/
- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/
- **Xavier Initialization**: Glorot & Bengio, 2010

---

## Author Notes

**Created**: 2024
**Purpose**: Parallel computing coursework
**Goal**: Compare parallelization strategies for neural network training

All implementations are functionally equivalent and produce identical results. The choice of implementation depends on:
- **Available hardware** (CPU only vs. NVIDIA GPU)
- **Development time** (OpenMP is simpler than CUDA)
- **Performance requirements** (CUDA is fastest)
- **Portability** (OpenMP is more portable)

For maximum performance, **use mnist_mlp_cuda.cu** on NVIDIA GPUs.
For portability and ease of development, **use mnist_mlp_openmp_cpu.c**.
