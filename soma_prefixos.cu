/*
nvcc -O3 soma_prefixos.cu -o soma_prefixos
ncu ./soma_prefixos

=== SOMA DE PREFIXOS EM CUDA ===
Tamanho do array: 1048576 elementos

Executando versao sequencial (CPU)...
==PROF== Connected to process 74020 (C:\Users\anton\Downloads\computacaoparalela\soma_prefixos.exe)
Tempo CPU: 0.722944 ms

Executando versao paralela (GPU)...
Configura├ºao: 4096 blocos x 256 threads
==PROF== Profiling "somaPrefixosBloco" - 0: 0%....50%....100% - 8 passes
==PROF== Profiling "adicionarOffsets" - 1: 0%....50%....100% - 8 passes
Tempo GPU: 429.223 ms

Verificando corretude...

 SOMA DE PREFIXOS OCORREU COM SUCESSO!
 Resultados CPU e GPU sao idanticos!

Performance:
  - Tempo CPU: 0.722944 ms
  - Tempo GPU: 429.223 ms
  - Speedup: 0.00168431x
==PROF== Disconnected from process 74020
[74020] soma_prefixos.exe@127.0.0.1
  somaPrefixosBloco(int *, int *, int *, int) (4096, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.98
    SM Frequency                    Ghz         1.51
    Elapsed Cycles                cycle      118,929
    Memory Throughput                 %        83.67
    DRAM Throughput                   %        63.05
    Duration                         us        78.53
    L1/TEX Cache Throughput           %        85.21
    L2 Cache Throughput               %        19.79
    SM Active Cycles              cycle   116,325.15
    Compute (SM) Throughput           %        83.67
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing workloads in the Compute Workload Analysis section.

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                  4,096
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           16.38
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      Kbyte/block            1.02
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              20
    Stack Size                                                 1,024
    Threads                                   thread       1,048,576
    # TPCs                                                        10
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                               34.13
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        92.54
    Achieved Active Warps Per SM           warp        44.42
    ------------------------------- ----------- ------------

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle   345,754.67
    Total DRAM Elapsed Cycles        cycle    1,645,056
    Average L1 Active Cycles         cycle   116,325.15
    Total L1 Elapsed Cycles          cycle    2,369,480
    Average L2 Active Cycles         cycle   106,973.67
    Total L2 Elapsed Cycles          cycle    1,348,980
    Average SM Active Cycles         cycle   116,325.15
    Total SM Elapsed Cycles          cycle    2,369,480
    Average SMSP Active Cycles       cycle   116,428.90
    Total SMSP Elapsed Cycles        cycle    9,477,920
    -------------------------- ----------- ------------

  adicionarOffsets(int *, int *, int) (4096, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.98
    SM Frequency                    Ghz         1.51
    Elapsed Cycles                cycle       76,427
    Memory Throughput                 %        91.71
    DRAM Throughput                   %        91.71
    Duration                         us        50.53
    L1/TEX Cache Throughput           %        20.41
    L2 Cache Throughput               %        30.76
    SM Active Cycles              cycle       71,437
    Compute (SM) Throughput           %        17.48
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing DRAM in the Memory Workload Analysis section.

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                  4,096
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              20
    Stack Size                                                 1,024
    Threads                                   thread       1,048,576
    # TPCs                                                        10
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                               34.13
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        85.71
    Achieved Active Warps Per SM           warp        41.14
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 14.29%
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (85.7%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle      323,520
    Total DRAM Elapsed Cycles        cycle    1,058,304
    Average L1 Active Cycles         cycle       71,437
    Total L1 Elapsed Cycles          cycle    1,499,330
    Average L2 Active Cycles         cycle    68,529.67
    Total L2 Elapsed Cycles          cycle      866,124
    Average SM Active Cycles         cycle       71,437
    Total SM Elapsed Cycles          cycle    1,499,330
    Average SMSP Active Cycles       cycle    71,859.27
    Total SMSP Elapsed Cycles        cycle    5,997,320
    -------------------------- ----------- ------------
*/

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <cuda_runtime.h>

using std::generate;
using std::cout;
using std::vector;

#define TAMANHO_MEM_COMPARTILHADA 256

// Macro para verificar erros CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// ## PONTO 2: SOMA DE PREFIXOS EM GPU ##
// Kernel CUDA para computar soma de prefixos em paralelo DENTRO de cada bloco
// Implementaçao usando algoritmo de Hillis-Steele com memória compartilhada
// Complexidade: O(log n) iterações com O(n) threads
__global__ void somaPrefixosBloco(int *v, int *v_somas, int *somas_blocos, int N) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Carrega dados para memória compartilhada
    if (gid < N) {
        temp[tid] = v[gid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    // Algoritmo de Hillis-Steele dentro do bloco
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int val = 0;
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        __syncthreads();

        if (tid >= offset) {
            temp[tid] += val;
        }
        __syncthreads();
    }

    // Escreve resultado de volta para memória global
    if (gid < N) {
        v_somas[gid] = temp[tid];
    }

    // A última thread de cada bloco salva o total do bloco
    if (tid == blockDim.x - 1 && somas_blocos != nullptr) {
        somas_blocos[blockIdx.x] = temp[tid];
    }
}

// Kernel para adicionar os offsets dos blocos anteriores
__global__ void adicionarOffsets(int *v_somas, int *offsets, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < N) {
        v_somas[gid] += offsets[blockIdx.x];
    }
}

// ## PONTO 1: SOMA DE PREFIXOS EM CPU ##
// Implementaçao sequencial da soma de prefixos (prefix sum inclusiva)
// Exemplo: [1, 2, 3, 4] -> [1, 3, 6, 10]
void somaPrefixosCPU(int *arr, int *somas, int tamanho) {
    if (tamanho == 0) return;

    somas[0] = arr[0];
    for (int i = 1; i < tamanho; i++) {
        somas[i] = somas[i - 1] + arr[i];
    }
}

// ## PONTO 3: VERIFICAÇaO ##
// Funçao para verificar se dois vetores sao iguais
bool verificarResultados(const vector<int>& cpu_result, const vector<int>& gpu_result) {
    if (cpu_result.size() != gpu_result.size()) {
        cout << "ERRO: Tamanhos diferentes!\n";
        return false;
    }

    for (size_t i = 0; i < cpu_result.size(); i++) {
        if (cpu_result[i] != gpu_result[i]) {
            cout << "ERRO na posiçao " << i << ": CPU=" << cpu_result[i]
                 << " GPU=" << gpu_result[i] << "\n";
            return false;
        }
    }
    return true;
}

// Funçao para imprimir vetores (útil para debug)
void imprimirVetor(const char* nome, const vector<int>& vec, int max_elementos = 20) {
    cout << nome << ": [";
    int n = std::min((int)vec.size(), max_elementos);
    for (int i = 0; i < n; i++) {
        cout << vec[i];
        if (i < n - 1) cout << ", ";
    }
    if (vec.size() > max_elementos) cout << ", ...";
    cout << "]\n";
}

int main() {
    // Tamanho do array: 2^20 = 1048576
    int N = 1 << 20;
    size_t bytes = N * sizeof(int);

    cout << "=== SOMA DE PREFIXOS EM CUDA ===\n";
    cout << "Tamanho do array: " << N << " elementos\n\n";

    // Vetores na máquina host
    vector<int> host_arr(N);
    vector<int> host_somas_cpu(N);     // Resultado da CPU
    vector<int> host_somas_gpu(N);     // Resultado da GPU

    // Inicializa o vetor com valores aleatórios (0-9)
    generate(begin(host_arr), end(host_arr), [](){ return rand() % 10; });

    // Se N for pequeno, imprime o vetor original
    if (N <= 20) {
        imprimirVetor("Vetor original", host_arr);
    }

    // ## PONTO 1: Executa soma de prefixos em CPU ##
    cout << "Executando versao sequencial (CPU)...\n";
    cudaEvent_t start_cpu, stop_cpu;
    CUDA_CHECK(cudaEventCreate(&start_cpu));
    CUDA_CHECK(cudaEventCreate(&stop_cpu));

    CUDA_CHECK(cudaEventRecord(start_cpu));
    somaPrefixosCPU(host_arr.data(), host_somas_cpu.data(), N);
    CUDA_CHECK(cudaEventRecord(stop_cpu));
    CUDA_CHECK(cudaEventSynchronize(stop_cpu));

    float tempo_cpu = 0;
    CUDA_CHECK(cudaEventElapsedTime(&tempo_cpu, start_cpu, stop_cpu));
    cout << "Tempo CPU: " << tempo_cpu << " ms\n\n";

    if (N <= 20) {
        imprimirVetor("Resultado CPU", host_somas_cpu);
    }

    // ## PONTO 2: Executa soma de prefixos em GPU ##
    cout << "Executando versao paralela (GPU)...\n";

    // Aloca memória no dispositivo (device)
    int *device_arr, *device_somas, *device_somas_blocos, *device_offsets;
    CUDA_CHECK(cudaMalloc(&device_arr, bytes));
    CUDA_CHECK(cudaMalloc(&device_somas, bytes));

    // Configuraçao do grid e blocos
    const int TAMANHO_BLOCO = 256;
    int TAMANHO_GRID = (N + TAMANHO_BLOCO - 1) / TAMANHO_BLOCO;

    CUDA_CHECK(cudaMalloc(&device_somas_blocos, TAMANHO_GRID * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_offsets, TAMANHO_GRID * sizeof(int)));

    cout << "Configuraçao: " << TAMANHO_GRID << " blocos x "
         << TAMANHO_BLOCO << " threads\n";

    // Cria eventos para medir tempo
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    // Copia da máquina hospedeira (host) para o dispositivo (device)
    CUDA_CHECK(cudaMemcpy(device_arr, host_arr.data(), bytes, cudaMemcpyHostToDevice));

    // Registra tempo de início
    CUDA_CHECK(cudaEventRecord(start_gpu));

    // FASE 1: Calcula soma de prefixos dentro de cada bloco
    size_t shared_mem_size = TAMANHO_BLOCO * sizeof(int);
    somaPrefixosBloco<<<TAMANHO_GRID, TAMANHO_BLOCO, shared_mem_size>>>(
        device_arr, device_somas, device_somas_blocos, N);
    CUDA_CHECK(cudaGetLastError());

    // FASE 2: Calcula soma de prefixos das somas dos blocos (na CPU para simplificar)
    if (TAMANHO_GRID > 1) {
        vector<int> host_somas_blocos(TAMANHO_GRID);
        CUDA_CHECK(cudaMemcpy(host_somas_blocos.data(), device_somas_blocos,
                              TAMANHO_GRID * sizeof(int), cudaMemcpyDeviceToHost));

        // Calcula soma de prefixos EXCLUSIVA dos totais dos blocos
        vector<int> host_offsets(TAMANHO_GRID);
        host_offsets[0] = 0;
        for (int i = 1; i < TAMANHO_GRID; i++) {
            host_offsets[i] = host_offsets[i-1] + host_somas_blocos[i-1];
        }

        CUDA_CHECK(cudaMemcpy(device_offsets, host_offsets.data(),
                              TAMANHO_GRID * sizeof(int), cudaMemcpyHostToDevice));

        // FASE 3: Adiciona os offsets aos resultados de cada bloco
        adicionarOffsets<<<TAMANHO_GRID, TAMANHO_BLOCO>>>(device_somas, device_offsets, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // Registra tempo de fim
    CUDA_CHECK(cudaEventRecord(stop_gpu));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));

    // Calcula tempo decorrido
    float tempo_gpu = 0;
    CUDA_CHECK(cudaEventElapsedTime(&tempo_gpu, start_gpu, stop_gpu));
    cout << "Tempo GPU: " << tempo_gpu << " ms\n\n";

    // Copia do dispositivo (device) para a máquina hospedeira (host)
    CUDA_CHECK(cudaMemcpy(host_somas_gpu.data(), device_somas, bytes, cudaMemcpyDeviceToHost));

    if (N <= 20) {
        imprimirVetor("Resultado GPU", host_somas_gpu);
    }

    // ## PONTO 3: VERIFICAÇaO ##
    cout << "Verificando corretude...\n";
    bool resultados_iguais = verificarResultados(host_somas_cpu, host_somas_gpu);

    if (resultados_iguais) {
        cout << "\n SOMA DE PREFIXOS OCORREU COM SUCESSO!\n";
        cout << " Resultados CPU e GPU sao idanticos!\n\n";

        // Calcula speedup
        float speedup = tempo_cpu / tempo_gpu;
        cout << "Performance:\n";
        cout << "  - Tempo CPU: " << tempo_cpu << " ms\n";
        cout << "  - Tempo GPU: " << tempo_gpu << " ms\n";
        cout << "  - Speedup: " << speedup << "x\n";
    } else {
        cout << "\n✗ ERRO: Resultados CPU e GPU sao diferentes!\n";
        return 1;
    }

    // Limpeza
    CUDA_CHECK(cudaFree(device_arr));
    CUDA_CHECK(cudaFree(device_somas));
    CUDA_CHECK(cudaFree(device_somas_blocos));
    CUDA_CHECK(cudaFree(device_offsets));
    CUDA_CHECK(cudaEventDestroy(start_cpu));
    CUDA_CHECK(cudaEventDestroy(stop_cpu));
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));

    return 0;
}
