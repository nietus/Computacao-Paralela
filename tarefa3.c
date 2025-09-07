/*
Sequencial: 0.782000 s
Paralelo: 0.438000 s
Resultado Sequencial: 5761455
Resultado Paralelo: 5761455
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

int main(void)
{
    int n = 100000000;

    double t0, t1;
    int primes_seq, primes_par;

    t0 = omp_get_wtime();
    primes_seq = sieveOfEratosthenes(n);
    t1 = omp_get_wtime();
    double dt_seq = t1 - t0;

    t0 = omp_get_wtime();
    primes_par = sieve_parallel(n);
    t1 = omp_get_wtime();
    double dt_par = t1 - t0;

    (void)dt_seq;
    (void)dt_par;

    printf("Sequencial: %f s\n", dt_seq);
    printf("Paralelo: %f s\n", dt_par);
    printf("Resultado Sequencial: %d\n", primes_seq);
    printf("Resultado Paralelo: %d\n", primes_par);
    return 0;
}