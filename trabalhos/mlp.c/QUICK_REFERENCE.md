# Quick Reference Guide - MNIST MLP Implementations

## Files Overview

```
mnist_mlp.c              - Sequential baseline (CPU)
mnist_mlp_openmp_cpu.c   - OpenMP multicore (16 threads)
mnist_mlp_openmp_gpu.c   - OpenMP GPU offloading
mnist_mlp_cuda.cu        - CUDA optimized GPU
```

## Compilation Commands

```bash
# Baseline
gcc -O3 -o mnist_mlp mnist_mlp.c -lm

# OpenMP CPU (16 threads)
gcc -fopenmp -O3 -o mnist_mlp_openmp_cpu mnist_mlp_openmp_cpu.c -lm

# OpenMP GPU
gcc -fopenmp -foffload=nvptx-none -O3 -o mnist_mlp_openmp_gpu mnist_mlp_openmp_gpu.c -lm

# CUDA (fastest)
nvcc -O3 -o mnist_mlp_cuda mnist_mlp_cuda.cu -lm
```

## Execution

```bash
./mnist_mlp              # Baseline
./mnist_mlp_openmp_cpu   # OpenMP CPU
./mnist_mlp_openmp_gpu   # OpenMP GPU
./mnist_mlp_cuda         # CUDA
```

## Expected Results

| Implementation | Speedup | Accuracy | Log File |
|----------------|---------|----------|----------|
| **mnist_mlp.c** | 1× (baseline) | ~97-98% | `logs/training_loss_c.txt` |
| **mnist_mlp_openmp_cpu.c** | 5-10× | ~97-98% | `logs/training_loss_openmp_cpu.txt` |
| **mnist_mlp_openmp_gpu.c** | 0.5-3× | ~97-98% | `logs/training_loss_openmp_gpu.txt` |
| **mnist_mlp_cuda.cu** | 10-50× | ~97-98% | `logs/training_loss_cuda.txt` |

## Key Architectural Differences

### Weight Storage

```c
// mnist_mlp.c, openmp_cpu, openmp_gpu
double **weights;  // 2D array: weights[i][j]

// mnist_mlp_cuda.cu
double *weights;   // Flattened: weights[i * output_size + j]
```

### Memory Management

```c
// mnist_mlp.c, openmp_cpu, openmp_gpu
// All memory on CPU

// mnist_mlp_cuda.cu
typedef struct {
    double *weights;     // Host (CPU)
    double *biases;      // Host (CPU)
    double *d_weights;   // Device (GPU) - persistent
    double *d_biases;    // Device (GPU) - persistent
} LinearLayer;
```

### Parallelization Patterns

```c
// OpenMP CPU
#pragma omp parallel for
for (int i = 0; i < size; i++) {
    // computation
}

// OpenMP GPU
#pragma omp target teams distribute parallel for
for (int i = 0; i < size; i++) {
    // computation
}

// CUDA
__global__ void kernel(double *data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // computation
    }
}
// Launch: kernel<<<numBlocks, blockSize>>>(data, size);
```

## CUDA-Specific Features

### Custom Kernels

1. `linear_forward_kernel` - Matrix-vector multiplication
2. `apply_relu_kernel` - ReLU activation
3. `apply_sigmoid_kernel` - Sigmoid activation
4. `output_delta_kernel` - Output gradients
5. `hidden_delta_kernel` - Hidden gradients
6. `update_weights_kernel` - Weight updates (2D grid)
7. `update_biases_kernel` - Bias updates

### Memory Lifecycle

```c
// Initialization (once)
cudaMalloc(&d_weights, size);
cudaMemcpy(d_weights, weights, size, cudaMemcpyHostToDevice);

// Training loop (many times)
cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);  // In
kernel<<<...>>>(d_input, d_weights, ...);                  // Compute
cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost); // Out

// Cleanup (once)
cudaFree(d_weights);
```

### Error Checking

```c
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

CUDA_CHECK(cudaMalloc(&d_data, size));
```

## Performance Tips

### OpenMP CPU
- Ensure 16 cores available: `lscpu | grep "^CPU(s):"`
- Set threads: `export OMP_NUM_THREADS=16`
- Use `schedule(static)` for load balancing

### OpenMP GPU
- May be slower than CPU for small networks
- Best for larger networks (not 784→128→10)
- Requires compiler with GPU offload support

### CUDA
- **Always fastest** for this implementation
- Check GPU availability: `nvidia-smi`
- Monitor GPU usage: `watch -n 1 nvidia-smi`
- Persistent memory minimizes overhead

## Troubleshooting

### OpenMP CPU
```bash
# Problem: Not using all cores
lscpu | grep "CPU(s)"
export OMP_NUM_THREADS=16

# Verify parallelization
./mnist_mlp_openmp_cpu | grep "Epoch 1"
# Should be ~10× faster than baseline
```

### OpenMP GPU
```bash
# Problem: Compilation fails
# Solution: Use newer compiler
gcc --version  # Need GCC 12+ or Clang 14+

# Problem: Slower than CPU
# Expected for small networks due to transfer overhead
```

### CUDA
```bash
# Problem: "nvcc: command not found"
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Problem: "CUDA error: out of memory"
# Reduce TRAIN_SAMPLES in #define

# Problem: Different results
# Check: Same random seed? Correct indexing in flattened arrays?
```

## Performance Measurement

### Compare All Versions

```bash
# Run all and save output
./mnist_mlp 2>&1 | tee baseline.log
./mnist_mlp_openmp_cpu 2>&1 | tee openmp_cpu.log
./mnist_mlp_cuda 2>&1 | tee cuda.log

# Extract training times
grep "Total training time" *.log
```

### Expected Output

```
baseline.log:     Total training time: 120.00 seconds
openmp_cpu.log:   Total training time: 15.00 seconds (8× faster)
cuda.log:         Total training time: 3.00 seconds (40× faster)
```

## Verification

### Check Correctness

```bash
# All should have similar accuracy
grep "Test Accuracy" baseline.log openmp_cpu.log cuda.log

# Compare loss curves
paste logs/training_loss_c.txt logs/training_loss_cuda.txt
```

### Expected Values

```
Test Accuracy: 97.50% (±0.5%)
Final Loss: ~0.12 (±0.02)
```

## Network Configuration

```c
#define NUM_INPUTS 784       // 28×28 pixels
#define NUM_HIDDEN 128       // Hidden layer size
#define NUM_OUTPUTS 10       // 10 digits
#define TRAIN_SAMPLES 20000  // Training set size
#define TEST_SAMPLES 5000    // Test set size
#define LEARNING_RATE 0.01   // Learning rate
#define EPOCHS 10            // Training epochs
#define BATCH_SIZE 64        // Mini-batch size
```

## Data Requirements

```bash
# Directory structure
./data/train-images.idx3-ubyte  # Training images
./data/train-labels.idx1-ubyte  # Training labels
./data/t10k-images.idx3-ubyte   # Test images
./data/t10k-labels.idx1-ubyte   # Test labels
./logs/                         # Output directory (auto-created)
```

## Summary Decision Tree

```
Do you have NVIDIA GPU with CUDA?
├─ YES → Use mnist_mlp_cuda.cu (fastest, 10-50× speedup)
└─ NO
   └─ Do you have multi-core CPU?
      ├─ YES → Use mnist_mlp_openmp_cpu.c (5-10× speedup)
      └─ NO → Use mnist_mlp.c (baseline)
```

## Code Size Comparison

| File | Lines | Complexity |
|------|-------|------------|
| mnist_mlp.c | ~471 | Low |
| mnist_mlp_openmp_cpu.c | ~479 | Medium |
| mnist_mlp_openmp_gpu.c | ~479 | Medium |
| mnist_mlp_cuda.cu | ~478 | High |

**Note**: All implementations have similar code size, but CUDA requires understanding GPU programming concepts.

---

**For detailed explanations, see `IMPLEMENTATION_GUIDE.md`**
