/*
Sequencial: 37.000000 s
Paralelo: 3.814000 s
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void mm(double* a, double* b, double* c, int width) 
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

void mmparalel(double* a, double* b, double* c, int width) 
{
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      double sum = 0;
      #pragma omp simd reduction(+:sum)
      for (int k = 0; k < width; k++) {
        double x = a[i * width + k];
        double y = b[k * width + j];
        sum += x * y;
      }
      c[i * width + j] = sum;
    }
  }
}

int main()
{
  int width = 2000;
  double *a = (double*) malloc (width * width * sizeof(double));
  double *b = (double*) malloc (width * width * sizeof(double));
  double *c = (double*) malloc (width * width * sizeof(double));
  double *c2 = (double*) malloc (width * width * sizeof(double)); // para o paralelo

  #pragma omp parallel for collapse(2) schedule(static)
  for(int i = 0; i < width; i++) {
    for(int j = 0; j < width; j++) {
      a[i*width+j] = i;
      b[i*width+j] = j;
      c[i*width+j] = 0;
      c2[i*width+j] = 0;
    }
  }

  double t0 = omp_get_wtime();
  mm(a,b,c,width);
  double t1 = omp_get_wtime();
  double dt_seq = t1 - t0;

  t0 = omp_get_wtime();
  mmparalel(a,b,c2,width);
  t1 = omp_get_wtime();
  double dt_par = t1 - t0;

  printf("Sequencial: %f s\n", dt_seq);
  printf("Paralelo: %f s\n", dt_par);

//   for(int i = 0; i < width; i++) {
//     for(int j = 0; j < width; j++) {
//       printf("\n c[%d][%d] = %f",i,j,c[i*width+j]);
//       printf("\n c2[%d][%d] = %f",i,j,c2[i*width+j]);
//     }
//   }
}