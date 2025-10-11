/*
Neste exercício, você deve implementar uma soma de prefixos em CUDA.

Você deve completar o código em três pontos:

## SOMA DE PREFIXOS EM CPU ##

Implemente a soma de prefixos em CPU e execute sobre o vetor original.
Utilize a saída do algoritmo síncrono (em CPU) para verificar a
corretude da saída do algoritmo paralelo (em GPU)

## SOMA DE PREFIXOS EM GPU ##

Utilize seus conhecimentos em CUDA para implementar uma estratégia
de indexação de threads para computar a soma de prefixos em paralelo.

## VERIFICAÇÃO ##

Implemente algum tipo de verificação. Sugestão: confira
se a saída do algoritmo síncrono (CPU) bate com a saída do
algoritmo paralelo (GPU)
*/
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>

using std::generate;
using std::cout;
using std::vector;

#define TAMANHO_MEM_COMPARTILHADA 256

__global__ void somaPrefixos(int *v, int *v_somas) {
	// OBSERVAÇÃO: Não é necessário implementar
	// utilizando memória compartilhada, apesar de ser
	// a alternativa ótima (para que threads de um mesmo bloco)
	// não precisem acessar a memória global reiteradas vezes.

	// Cálculo do ID da thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// ## SOMA DE PREFIXOS EM GPU ##
	// Implemente aqui a soma de prefixos paralela
}

int main() {
	// Tamanho do array: 8
	int N = 1 << 3;
	size_t bytes = N * sizeof(int);

	// Vetores (arrays) na maquina host: vetor original e
    // vetor reduzido.
	vector<int> host_arr(N);
	vector<int> host_somas_prefixos(N);

    // Inicializa o vetor , i. e., aloca memória na maquina host
    generate(begin(host_arr), end(host_arr), [](){ return rand() % 10; });

	// Aloca memória no dispositivo (device)
	int *device_arr, *device_somas_prefixos;
	cudaMalloc(&device_arr, bytes);
	cudaMalloc(&device_somas_prefixos, bytes);
	
	// Copia da maquina hospedeira (host) para o dispositivo (device)
	cudaMemcpy(device_arr, host_arr.data(), bytes, cudaMemcpyHostToDevice);
	
	// Tamanho do bloco em número de threads
	const int TAMANHO_BLOCO = 8;

	// Tamanho do grid em número de bloco
    // (tamanho do array / número de threads por bloco)
	int TAMANHO_GRID = N / TAMANHO_BLOCO;

	// Chamadas para o kernel
	reducaoSoma<<<TAMANHO_GRID, TAMANHO_BLOCO>>>(device_arr, device_somas_prefixos);

	// Copia do dispositivo (device) para a máquina hospedeira (host)
	cudaMemcpy(host_somas_prefixos.data(), device_somas_prefixos, bytes, cudaMemcpyDeviceToHost);

	// Confere resultado
	// ## VERIFICAÇÃO ##
	// Implemente aqui a verificação do resultado do algoritmo paralelo
	assert(false);

	cout << "REDUÇÃO OCORREU COM SUCESSO.\n";

	return 0;
}

// ## SOMA DE PREFIXOS EM CPU ##
// Implemente aqui a soma de prefixos sequencial
void somaPrefixos(int *arr, int *somas, int tamanho) {}