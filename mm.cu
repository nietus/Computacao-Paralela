/*
nvcc -O3 mm.cu -o mm
ncu --metrics smsp__warps_launched.sum,smsp__thread_inst_executed_per_inst_executed.ratio ./mm

Running sequential version...
Sequential time: 32.533685 seconds
Running multicore parallel version...
Multicore parallel time: 31.661228 seconds
Running GPU (distribute) version...
==PROF== Connected to process 20472 (C:\Users\anton\Downloads\computacaoparalela\mm.exe)
==PROF== Profiling "mm_gpu_kernel_basic" - 0: 0%....50%
==WARNING== Launching the workload is taking more time than expected. If this continues to hang, terminate the profile and re-try by profiling the range of all related launches using '--replay-mode range'. See https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay for more details.
....100% - 1 pass
GPU (distribute) time: 7.439555 seconds
Running GPU (distribute parallel for) version...
GPU (distribute parallel for) time: 0.014122 seconds
Running GPU (distribute parallel for simd) version...
==PROF== Profiling "mm_gpu_kernel_2d" - 1: 0%....50%....100% - 1 pass
GPU (distribute parallel for simd) time: 0.157809 seconds
==PROF== Disconnected from process 20472
[20472] mm.exe@127.0.0.1
  mm_gpu_kernel_basic(double *, double *, double *, int) (2000, 1, 1)x(1, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: Command line profiler metrics
    -------------------------------------------------- ----------- ------------
    Metric Name                                        Metric Unit Metric Value
    -------------------------------------------------- ----------- ------------
    smsp__thread_inst_executed_per_inst_executed.ratio                        1
    smsp__warps_launched.sum                                  warp        2,000
    -------------------------------------------------- ----------- ------------

  mm_gpu_kernel_2d(double *, double *, double *, int) (125, 125, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: Command line profiler metrics
    -------------------------------------------------- ----------- ------------
    Metric Name                                        Metric Unit Metric Value
    -------------------------------------------------- ----------- ------------
    smsp__thread_inst_executed_per_inst_executed.ratio                       32
    smsp__warps_launched.sum                                  warp      125,000
    -------------------------------------------------- ----------- ------------
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

// Sequential version
void mm_seq(double* a, double* b, double* c, int width)
{
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            double sum = 0;
            for (int k = 0; k < width; k++) {
                double x = a[i * width + k];
                double y = b[k * width + j];
                sum += x * y;
            }
            c[i * width + j] = sum;
        }
    }
}

// Multicore parallel version
void mm_multicore(double* a, double* b, double* c, int width)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            double sum = 0;
            for (int k = 0; k < width; k++) {
                double x = a[i * width + k];
                double y = b[k * width + j];
                sum += x * y;
            }
            c[i * width + j] = sum;
        }
    }
}

// CUDA kernel - basic version (like distribute)
__global__ void mm_gpu_kernel_basic(double* a, double* b, double* c, int width)
{
    int i = blockIdx.x;
    if (i < width) {
        for (int j = 0; j < width; j++) {
            double sum = 0;
            for (int k = 0; k < width; k++) {
                double x = a[i * width + k];
                double y = b[k * width + j];
                sum += x * y;
            }
            c[i * width + j] = sum;
        }
    }
}

// CUDA kernel - parallel version (like distribute parallel for)
__global__ void mm_gpu_kernel_parallel(double* a, double* b, double* c, int width)
{
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (i < width && j < width) {
        double sum = 0;
        for (int k = 0; k < width; k++) {
            double x = a[i * width + k];
            double y = b[k * width + j];
            sum += x * y;
        }
        c[i * width + j] = sum;
    }
}

// CUDA kernel - 2D grid version (most parallel)
__global__ void mm_gpu_kernel_2d(double* a, double* b, double* c, int width)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width && j < width) {
        double sum = 0;
        for (int k = 0; k < width; k++) {
            double x = a[i * width + k];
            double y = b[k * width + j];
            sum += x * y;
        }
        c[i * width + j] = sum;
    }
}

// GPU wrapper - basic (distribute)
void mm_gpu_distribute(double* a, double* b, double* c, int width)
{
    double *d_a, *d_b, *d_c;
    size_t size = width * width * sizeof(double);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    mm_gpu_kernel_basic<<<width, 1>>>(d_a, d_b, d_c, width);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// GPU wrapper - distribute parallel for
void mm_gpu_distribute_parallel(double* a, double* b, double* c, int width)
{
    double *d_a, *d_b, *d_c;
    size_t size = width * width * sizeof(double);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    mm_gpu_kernel_parallel<<<width, width>>>(d_a, d_b, d_c, width);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// GPU wrapper - distribute parallel for simd (2D grid)
void mm_gpu_distribute_parallel_simd(double* a, double* b, double* c, int width)
{
    double *d_a, *d_b, *d_c;
    size_t size = width * width * sizeof(double);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 15) / 16, (width + 15) / 16);

    mm_gpu_kernel_2d<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, width);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main()
{
    int width = 2000;
    double *a = (double*) malloc (width * width * sizeof(double));
    double *b = (double*) malloc (width * width * sizeof(double));
    double *c = (double*) malloc (width * width * sizeof(double));

    // Initialize matrices
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < width; j++) {
            a[i*width+j] = i;
            b[i*width+j] = j;
            c[i*width+j] = 0;
        }
    }

    double start, end;

    // Sequential version
    printf("Running sequential version...\n");
    start = omp_get_wtime();
    mm_seq(a, b, c, width);
    end = omp_get_wtime();
    printf("Sequential time: %f seconds\n", end - start);

    // Reset c
    for(int i = 0; i < width * width; i++) c[i] = 0;

    // Multicore version
    printf("Running multicore parallel version...\n");
    start = omp_get_wtime();
    mm_multicore(a, b, c, width);
    end = omp_get_wtime();
    printf("Multicore parallel time: %f seconds\n", end - start);

    // Reset c
    for(int i = 0; i < width * width; i++) c[i] = 0;

    // GPU distribute version
    printf("Running GPU (distribute) version...\n");
    start = omp_get_wtime();
    mm_gpu_distribute(a, b, c, width);
    end = omp_get_wtime();
    printf("GPU (distribute) time: %f seconds\n", end - start);

    // Reset c
    for(int i = 0; i < width * width; i++) c[i] = 0;

    // GPU distribute parallel for version
    printf("Running GPU (distribute parallel for) version...\n");
    start = omp_get_wtime();
    mm_gpu_distribute_parallel(a, b, c, width);
    end = omp_get_wtime();
    printf("GPU (distribute parallel for) time: %f seconds\n", end - start);

    // Reset c
    for(int i = 0; i < width * width; i++) c[i] = 0;

    // GPU distribute parallel for simd version
    printf("Running GPU (distribute parallel for simd) version...\n");
    start = omp_get_wtime();
    mm_gpu_distribute_parallel_simd(a, b, c, width);
    end = omp_get_wtime();
    printf("GPU (distribute parallel for simd) time: %f seconds\n", end - start);

    // Cleanup
    free(a);
    free(b);
    free(c);

    return 0;
}
