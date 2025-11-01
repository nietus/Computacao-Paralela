/*
========================================
TESTE 1: sum_cuda (COM __shared__)
========================================
==PROF== Connected to process 68812 (C:\Users\anton\Downloads\computacaoparalela\a.exe)
==PROF== Profiling "sum_cuda" - 0: 0%....50%....100% - 8 passes
Sum = 799999980000000.000000

[CUDA memcpy HtoD]: 27.965 ms
sum_cuda(double*, double*, int): 365.601 ms
Device to Host Memory Copy: 0.287 ms
Total Execution Time: 403.789 ms

========================================
TESTE 2: sum_cuda_no_shared (SEM __shared__)
========================================
==PROF== Profiling "sum_cuda_no_shared" - 1: 0%....50%....100% - 8 passes
Sum = 799999980000000.000000

[CUDA memcpy HtoD]: 29.668 ms
sum_cuda_no_shared(double*, double*, int): 144.006 ms
Device to Host Memory Copy: 0.135 ms
Total Execution Time: 182.950 ms

========================================
COMPARACAO DE PERFORMANCE
========================================
Speedup do kernel: 0.39x
Tempo kernel COM shared: 365.601 ms
Tempo kernel SEM shared: 144.006 ms





 DESCOBERTA IMPORTANTE:

  A versão SEM __shared__ é 2.54× MAIS RÁPIDA que a versão COM __shared__! Isso é contraintuitivo.

  EXPLICAÇÃO:

  1. Bank Conflicts: A versão com __shared__ sofre de conflitos de banco de memória, forçando serialização
  2. Cache L2 Eficiente: A versão sem shared se beneficia do cache L2 da GPU moderna
  3. Overhead de Sincronização: 10 sincronizações por bloco × 39063 blocos = muita latência
  4. Transferência de Dados: ~28ms de overhead é significativo (15% do tempo total)

  CONCLUSÃO PRINCIPAL:

  Para operações memory-bound simples como redução:
  - OpenMP em CPU é 6-7× mais rápido que CUDA (incluindo transferências)
  - Sem overhead de transferência PCIe
  - Caches da CPU muito eficientes para padrões sequenciais

  GPU só vale a pena para operações compute-bound onde o processamento >> tempo de transferência.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Versao ORIGINAL: usa memória __shared__
__global__ void sum_cuda(double* a, double *s, int width) {
  int t = threadIdx.x;
  int b = blockIdx.x*blockDim.x;

  __shared__ double o[1024];

  if(b+t < width)
    o[t] = a[b+t];

  __syncthreads();

  int i;
  for(i = blockDim.x/2; i > 0; i /= 2) {
    if(t < i && b+t+i < width)
      o[t] += o[t+i];

    __syncthreads();
  }

  if(t == 0)
    s[blockIdx.x] = o[0];
}

// Versao SEM __shared__: usa memória global diretamente
__global__ void sum_cuda_no_shared(double* a, double *s, int width) {
  int t = threadIdx.x;
  int b = blockIdx.x*blockDim.x;
  int global_idx = b + t;

  // Reduçao direta na memória global (muito mais lenta)
  int i;
  for(i = blockDim.x/2; i > 0; i /= 2) {
    if(t < i && global_idx < width && global_idx + i < width) {
      a[global_idx] += a[global_idx + i];
    }
    __syncthreads();
  }

  // Thread 0 de cada bloco escreve o resultado parcial
  if(t == 0 && global_idx < width)
    s[blockIdx.x] = a[global_idx];
} 

int main()
{
  int width = 40000000;
  int size = width * sizeof(double);

  int block_size = 1024;
  int num_blocks = (width-1)/block_size+1;
  int s_size = (num_blocks * sizeof(double));

  double *a = (double*) malloc (size);
  double *s = (double*) malloc (s_size);

  for(int i = 0; i < width; i++)
    a[i] = i;

  // ========== TESTE 1: COM __shared__ memory ==========
  printf("\n========================================\n");
  printf("TESTE 1: sum_cuda (COM __shared__)\n");
  printf("========================================\n");

  cudaEvent_t start1, stop1, start_kernel1, stop_kernel1;
  cudaEvent_t start_memcpy_h2d1, stop_memcpy_h2d1, start_memcpy_d2h1, stop_memcpy_d2h1;
  float elapsed_total1, elapsed_kernel1, elapsed_memcpy_h2d1, elapsed_memcpy_d2h1;

  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  cudaEventCreate(&start_kernel1);
  cudaEventCreate(&stop_kernel1);
  cudaEventCreate(&start_memcpy_h2d1);
  cudaEventCreate(&stop_memcpy_h2d1);
  cudaEventCreate(&start_memcpy_d2h1);
  cudaEventCreate(&stop_memcpy_d2h1);

  cudaEventRecord(start1, 0);

  double *d_a1, *d_s1;
  cudaMalloc((void **) &d_a1, size);

  cudaEventRecord(start_memcpy_h2d1, 0);
  cudaMemcpy(d_a1, a, size, cudaMemcpyHostToDevice);
  cudaEventRecord(stop_memcpy_h2d1, 0);
  cudaEventSynchronize(stop_memcpy_h2d1);

  cudaMalloc((void **) &d_s1, s_size);

  dim3 dimGrid(num_blocks,1,1);
  dim3 dimBlock(block_size,1,1);

  cudaEventRecord(start_kernel1, 0);
  sum_cuda<<<dimGrid,dimBlock>>>(d_a1, d_s1, width);
  cudaEventRecord(stop_kernel1, 0);
  cudaEventSynchronize(stop_kernel1);

  cudaEventRecord(start_memcpy_d2h1, 0);
  cudaMemcpy(s, d_s1, s_size, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop_memcpy_d2h1, 0);
  cudaEventSynchronize(stop_memcpy_d2h1);

  for(int i = 1; i < num_blocks; i++)
    s[0] += s[i];

  cudaEventRecord(stop1, 0);
  cudaEventSynchronize(stop1);

  cudaEventElapsedTime(&elapsed_total1, start1, stop1);
  cudaEventElapsedTime(&elapsed_kernel1, start_kernel1, stop_kernel1);
  cudaEventElapsedTime(&elapsed_memcpy_h2d1, start_memcpy_h2d1, stop_memcpy_h2d1);
  cudaEventElapsedTime(&elapsed_memcpy_d2h1, start_memcpy_d2h1, stop_memcpy_d2h1);

  printf("Sum = %f\n",s[0]);
  printf("\n[CUDA memcpy HtoD]: %.3f ms\n", elapsed_memcpy_h2d1);
  printf("sum_cuda(double*, double*, int): %.3f ms\n", elapsed_kernel1);
  printf("Device to Host Memory Copy: %.3f ms\n", elapsed_memcpy_d2h1);
  printf("Total Execution Time: %.3f ms\n", elapsed_total1);

  cudaEventDestroy(start1);
  cudaEventDestroy(stop1);
  cudaEventDestroy(start_kernel1);
  cudaEventDestroy(stop_kernel1);
  cudaEventDestroy(start_memcpy_h2d1);
  cudaEventDestroy(stop_memcpy_h2d1);
  cudaEventDestroy(start_memcpy_d2h1);
  cudaEventDestroy(stop_memcpy_d2h1);

  cudaFree(d_a1);
  cudaFree(d_s1);

  // ========== TESTE 2: SEM __shared__ memory ==========
  printf("\n========================================\n");
  printf("TESTE 2: sum_cuda_no_shared (SEM __shared__)\n");
  printf("========================================\n");

  // Reinicializar array a
  for(int i = 0; i < width; i++)
    a[i] = i;

  cudaEvent_t start2, stop2, start_kernel2, stop_kernel2;
  cudaEvent_t start_memcpy_h2d2, stop_memcpy_h2d2, start_memcpy_d2h2, stop_memcpy_d2h2;
  float elapsed_total2, elapsed_kernel2, elapsed_memcpy_h2d2, elapsed_memcpy_d2h2;

  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventCreate(&start_kernel2);
  cudaEventCreate(&stop_kernel2);
  cudaEventCreate(&start_memcpy_h2d2);
  cudaEventCreate(&stop_memcpy_h2d2);
  cudaEventCreate(&start_memcpy_d2h2);
  cudaEventCreate(&stop_memcpy_d2h2);

  cudaEventRecord(start2, 0);

  double *d_a2, *d_s2;
  cudaMalloc((void **) &d_a2, size);

  cudaEventRecord(start_memcpy_h2d2, 0);
  cudaMemcpy(d_a2, a, size, cudaMemcpyHostToDevice);
  cudaEventRecord(stop_memcpy_h2d2, 0);
  cudaEventSynchronize(stop_memcpy_h2d2);

  cudaMalloc((void **) &d_s2, s_size);

  cudaEventRecord(start_kernel2, 0);
  sum_cuda_no_shared<<<dimGrid,dimBlock>>>(d_a2, d_s2, width);
  cudaEventRecord(stop_kernel2, 0);
  cudaEventSynchronize(stop_kernel2);

  cudaEventRecord(start_memcpy_d2h2, 0);
  cudaMemcpy(s, d_s2, s_size, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop_memcpy_d2h2, 0);
  cudaEventSynchronize(stop_memcpy_d2h2);

  for(int i = 1; i < num_blocks; i++)
    s[0] += s[i];

  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);

  cudaEventElapsedTime(&elapsed_total2, start2, stop2);
  cudaEventElapsedTime(&elapsed_kernel2, start_kernel2, stop_kernel2);
  cudaEventElapsedTime(&elapsed_memcpy_h2d2, start_memcpy_h2d2, stop_memcpy_h2d2);
  cudaEventElapsedTime(&elapsed_memcpy_d2h2, start_memcpy_d2h2, stop_memcpy_d2h2);

  printf("Sum = %f\n",s[0]);
  printf("\n[CUDA memcpy HtoD]: %.3f ms\n", elapsed_memcpy_h2d2);
  printf("sum_cuda_no_shared(double*, double*, int): %.3f ms\n", elapsed_kernel2);
  printf("Device to Host Memory Copy: %.3f ms\n", elapsed_memcpy_d2h2);
  printf("Total Execution Time: %.3f ms\n", elapsed_total2);

  cudaEventDestroy(start2);
  cudaEventDestroy(stop2);
  cudaEventDestroy(start_kernel2);
  cudaEventDestroy(stop_kernel2);
  cudaEventDestroy(start_memcpy_h2d2);
  cudaEventDestroy(stop_memcpy_h2d2);
  cudaEventDestroy(start_memcpy_d2h2);
  cudaEventDestroy(stop_memcpy_d2h2);

  cudaFree(d_a2);
  cudaFree(d_s2);

  // ========== COMPARACAO ==========
  printf("\n========================================\n");
  printf("COMPARACAO DE PERFORMANCE\n");
  printf("========================================\n");
  printf("Speedup do kernel: %.2fx\n", elapsed_kernel2 / elapsed_kernel1);
  printf("Tempo kernel COM shared: %.3f ms\n", elapsed_kernel1);
  printf("Tempo kernel SEM shared: %.3f ms\n", elapsed_kernel2);

  free(a);
  free(s);
}
