#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
gcc tarefa9.c -o tarefa9 -fopenmp
time ./tarefa9

0. Sequencial
Execution time: 0.741949 seconds

real	0m0.744s
user	0m0.737s
sys	0m0.006s


1. static
Execution time: 0.397152 seconds

real	0m0.399s
user	0m0.783s
sys	0m0.008s

2. guided

Execution time: 0.370393 seconds

real	0m0.378s
user	0m0.523s
sys	0m0.175s


3. auto
Execution time: 0.376036 seconds

real	0m0.380s
user	0m0.749s
sys	0m0.005s
*/

int main(int argc, char *argv[]) 
{
   int i, j, n = 30000;
   double start_time, end_time; 

   // Allocate input, output and position arrays
   int *in = (int*) calloc(n, sizeof(int));
   int *pos = (int*) calloc(n, sizeof(int));   
   int *out = (int*) calloc(n, sizeof(int));   

   // Check for allocation failures
   if (!in || !pos || !out) {
       printf("Memory allocation failed\n");
       return 1;
   }

   // Initialize input array in the reverse order
   for(i = 0; i < n; i++) {
      in[i] = n - i;
   }
   
   start_time = omp_get_wtime();
   
   // Silly sort parallel implementation
   #pragma omp parallel for private(i, j) num_threads(2) schedule(auto)
   for(i = 0; i < n; i++) {
      int count = 0;
      for(j = 0; j < n; j++) {
         if(in[i] > in[j]) {
            count++;
         }
      }
      pos[i] = count;
   }

   // Move elements to final position
   for(i = 0; i < n; i++) {
      out[pos[i]] = in[i];
   }
   
   // Check if answer is correct
   for(i = 0; i < n; i++) {
      if(i + 1 != out[i]) {
         printf("test failed at index %d: expected %d, got %d\n", i, i+1, out[i]);
         free(in); 
         free(pos); 
         free(out);
         return 1;
      }
   }

   end_time = omp_get_wtime();
   printf("Execution time: %f seconds\n", end_time - start_time);

   // Free allocated memory
   free(in);
   free(pos);
   free(out);
   
   return 0;
}
