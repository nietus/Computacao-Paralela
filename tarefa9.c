#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) 
{
   int i, j, n = 30000;
   double start_time, end_time; 

   // Allocate input, output and position arrays
   int *in = (int*) calloc(n, sizeof(int));
   int *pos = (int*) calloc(n, sizeof(int));   
   int *out = (int*) calloc(n, sizeof(int));   

   // Initialize input array in the reverse order
   for(i = 0; i < n; i++)
      in[i] = n - i;
   
   start_time = omp_get_wtime();

   // Print input array
   //   for(i=0; i < n; i++) 
   //      printf("%d ",in[i]);
    
   // Silly sort parallel implementation
   #pragma omp parallel for num_threads(2) schedule(static, 1000)
   for(i = 0; i < n; i++) {
      int count = 0;
      for(j = 0; j < n; j++) {
         if(in[i] > in[j]) {
            count++;
         }
      }
      pos[i] = count;
   }

   // Move elements to final position (parallel version)
   #pragma omp parallel for num_threads(2) schedule(static)
   for(i = 0; i < n; i++) {
      out[pos[i]] = in[i];
   }
   
   // print output array
   //   for(i=0; i < n; i++) 
   //      printf("%d ",out[i]);

   // Check if answer is correct
   for(i=0; i < n; i++)
      if(i+1 != out[i]) 
      {
         printf("test failed\n");
         exit(0);
      }

   end_time = omp_get_wtime();
   printf("test passed\n");
   printf("Execution time: %f seconds\n", end_time - start_time);
}  
