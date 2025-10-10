/*
gcc -O3 -fopenmp mm-1.c -o mm
./mm

Sequential time: 29.173000 seconds
Multicore parallel time: 3.187000 seconds
GPU (distribute) time: 29.128000 seconds
GPU (distribute parallel for) time: 2.977000 seconds
GPU (distribute parallel for simd) time: 3.025000 seconds
==WARNING== No kernels were profiled.
*/

#include <stdio.h>
#include <stdlib.h>
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
    //#pragma omp parallel for collapse(2)
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

// GPU version with distribute
void mm_gpu_distribute(double* a, double* b, double* c, int width)
{
    //#pragma omp target teams map(to: a[0:width*width], b[0:width*width]) map(from: c[0:width*width])
    //#pragma omp distribute
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

// GPU version with distribute parallel for
void mm_gpu_distribute_parallel(double* a, double* b, double* c, int width)
{
    //#pragma omp target teams map(to: a[0:width*width], b[0:width*width]) map(from: c[0:width*width])
    //#pragma omp distribute parallel for collapse(2)
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

// GPU version with distribute parallel for simd
void mm_gpu_distribute_parallel_simd(double* a, double* b, double* c, int width)
{
    //#pragma omp target teams map(to: a[0:width*width], b[0:width*width]) map(from: c[0:width*width])
    //#pragma omp distribute parallel for simd collapse(2)
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
