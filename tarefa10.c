/*
Sequencial: 1.242000 s
Paralelo: 0.477000 s
Paralelo com Reducao: 0.257000 s
Resultado Sequencial: 5761455
Resultado Paralelo: 5761455
Resultado Paralelo com Reducao: 5761455


*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>
 
int sieveOfEratosthenes(int n)
{
   // Create a boolean array "prime[0..n]" and initialize
   // all entries it as true. A value in prime[i] will
   // finally be false if i is Not a prime, else true.
   int primes = 0; 
   bool *prime = (bool*) malloc((n+1)*sizeof(bool));
   int sqrt_n = sqrt(n);
 
   memset(prime, true,(n+1)*sizeof(bool));
 
   for (int p=2; p <= sqrt_n; p++)
   {
       // If prime[p] is not changed, then it is a prime
       if (prime[p] == true)
       {
           // Update all multiples of p
           for (int i=p*2; i<=n; i += p)
           prime[i] = false;
        }
    }
 
    // count prime numbers
    for (int p=2; p<=n; p++)
       if (prime[p])
         primes++;
 
    return(primes);
}
 
static int sieve_parallel(int n)
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
                for (int i = start; i <= n; i += p) prime[i] = false;
            }
        }
    }

    #pragma omp parallel for reduction(+:count) schedule(static)
    for (int i = 2; i <= n; i++) if (prime[i]) count++;

    free(prime);
    return count;
}

int sieve_parallel_reduce(int n)
{
    int count = 0;
    bool *prime = (bool*) malloc((size_t)(n + 1) * sizeof(bool));
    if (!prime) return -1;
    
    // Inicializar array
    memset(prime, true, (size_t)(n + 1) * sizeof(bool));
    if (n >= 0) prime[0] = false;
    if (n >= 1) prime[1] = false;

    int sqrt_n = (int)floor(sqrt((double)n));

    #pragma omp parallel
    {
        for (int p = 2; p <= sqrt_n; p++) {
            if (prime[p]) {
                int start = p * p;
                #pragma omp for schedule(static) nowait
                for (int i = start; i <= n; i += p) {
                    prime[i] = false;
                }
            }
        }
    }

    #pragma omp parallel for reduction(+:count) schedule(static)
    for (int i = 2; i <= n; i++) {
        if (prime[i]) count++;
    }

    free(prime);
    return count;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("  1 - Sequencial\n");
        printf("  2 - Paralelo\n");
        printf("  3 - Paralelo com Reducao\n");
        return 1;
    }

    int version = atoi(argv[1]);
    int n = 100000000;
    int result;
    double start_time, end_time;

    start_time = omp_get_wtime();
    
    switch (version) {
        case 1:
            result = sieveOfEratosthenes(n);
            break;
        case 2:
            result = sieve_parallel(n);
            break;
        case 3:
            result = sieve_parallel_reduce(n);
            break;
        default:
            printf("1, 2, or 3.\n");
            return 1;
    }
    
    end_time = omp_get_wtime();
    
    printf("Version: %d, Primes: %d, Time: %f seconds\n", version, result, end_time - start_time);
    return 0;
}