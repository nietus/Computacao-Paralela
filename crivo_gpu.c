/*
Compilação: gcc -O3 -fopenmp crivo_gpu.c -o crivo_gpu


Sequencial: 0.699000 s
Paralelo Multicore: 0.417000 s
Paralelo GPU: 1.768000 s
Resultado Sequencial: 5761455
Resultado Paralelo: 5761455
Resultado GPU: 5761455

Resultado: 5761455 primos
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// Versão sequencial
int sieveOfEratosthenes(int n)
{
    int primes = 0;
    bool *prime = (bool*) malloc((n+1)*sizeof(bool));
    int sqrt_n = sqrt(n);

    memset(prime, true, (n+1)*sizeof(bool));

    for (int p=2; p <= sqrt_n; p++)
    {
        if (prime[p] == true)
        {
            for (int i=p*2; i<=n; i += p)
                prime[i] = false;
        }
    }

    for (int p=2; p<=n; p++)
        if (prime[p])
            primes++;

    free(prime);
    return(primes);
}

// Versão paralela para multicore
int sieve_parallel(int n)
{
    int count = 0;
    bool *prime = (bool*) malloc((size_t)(n + 1) * sizeof(bool));
    if (!prime) return -1;
    memset(prime, true, (size_t)(n + 1) * sizeof(bool));
    if (n >= 0) prime[0] = false;
    if (n >= 1) prime[1] = false;

    int sqrt_n = (int)floor(sqrt((double)n));

    #pragma omp parallel
    {
        for (int p = 2; p <= sqrt_n; p++) {
            if (prime[p]) {
                int start = p * p;
                #pragma omp for schedule(static)
                for (int i = start; i <= n; i += p)
                    prime[i] = false;
            }
        }
    }

    #pragma omp parallel for reduction(+:count) schedule(static)
    for (int i = 2; i <= n; i++)
        if (prime[i])
            count++;

    free(prime);
    return count;
}

// Versão paralela para GPU
int sieve_gpu(int n)
{
    int count = 0;
    bool *prime = (bool*) malloc((size_t)(n + 1) * sizeof(bool));
    if (!prime) return -1;

    // Inicialização
    memset(prime, true, (size_t)(n + 1) * sizeof(bool));
    if (n >= 0) prime[0] = false;
    if (n >= 1) prime[1] = false;

    int sqrt_n = (int)floor(sqrt((double)n));

    // Fase de marcação dos não-primos na GPU
    #pragma omp target data map(tofrom: prime[0:n+1])
    {
        for (int p = 2; p <= sqrt_n; p++) {
            #pragma omp target update from(prime[p:1])
            if (prime[p]) {
                int start = p * p;
                #pragma omp target teams distribute parallel for
                for (int i = start; i <= n; i += p) {
                    prime[i] = false;
                }
            }
        }
    }

    // Contagem dos primos em paralelo na GPU
    #pragma omp parallel for reduction(+:count) schedule(static)
    for (int i = 2; i <= n; i++) {
        if (prime[i])
            count++;
    }

    free(prime);
    return count;
}

int main(void)
{
    int n = 100000000;

    double t0, t1;
    int primes_seq, primes_par, primes_gpu;

    // Versão sequencial
    t0 = omp_get_wtime();
    primes_seq = sieveOfEratosthenes(n);
    t1 = omp_get_wtime();
    double dt_seq = t1 - t0;

    // Versão paralela multicore
    t0 = omp_get_wtime();
    primes_par = sieve_parallel(n);
    t1 = omp_get_wtime();
    double dt_par = t1 - t0;

    // Versão paralela GPU
    t0 = omp_get_wtime();
    primes_gpu = sieve_gpu(n);
    t1 = omp_get_wtime();
    double dt_gpu = t1 - t0;

    printf("Sequencial: %f s\n", dt_seq);
    printf("Paralelo Multicore: %f s\n", dt_par);
    printf("Paralelo GPU: %f s\n", dt_gpu);
    printf("Resultado Sequencial: %d\n", primes_seq);
    printf("Resultado Paralelo: %d\n", primes_par);
    printf("Resultado GPU: %d\n", primes_gpu);

    return 0;
}
