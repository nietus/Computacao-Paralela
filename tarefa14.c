/*
Tempo sequencial: 4.074 segundos

Secoes paralelas:
  Tempo: 61.484 segundos
  Aceleracao: 0.07x
  Ordenado: Sim

Tarefas paralelas:
  Tempo: 4.753 segundos
  Aceleracao: 0.86x
  Ordenado: Sim 
*/

/* Parallel Quicksort implementations using OpenMP */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// A utility function to swap two elements
void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

/* This function takes last element as pivot, places
   the pivot element at its correct position in sorted
   array, and places all smaller (smaller than pivot)
   to left of pivot and all greater elements to right
   of pivot */
int partition (int arr[], int low, int high)
{
    int pivot = arr[high]; // pivot
    int i = (low - 1); // Index of smaller element
    
    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than or
        // equal to pivot
        if (arr[j] <= pivot)
        {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

/* The main function that implements QuickSort
   arr[] --> Array to be sorted,
   low --> Starting index,
   high --> Ending index */
void quickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(arr, low, high);
        
        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

/* Function to print an array */
void printArray(int arr[], int size)
{
    int i;
    for (i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}
// Parallel quicksort using OpenMP sections
void parallel_quicksort_sections(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                parallel_quicksort_sections(arr, low, pi - 1);
            }
            #pragma omp section
            {
                parallel_quicksort_sections(arr, pi + 1, high);
            }
        }
    }
}

// Parallel quicksort using OpenMP tasks
void parallel_quicksort_tasks(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        
        #pragma omp task firstprivate(arr, low, pi)
        {
            parallel_quicksort_tasks(arr, low, pi - 1);
        }
        
        #pragma omp task firstprivate(arr, high, pi)
        {
            parallel_quicksort_tasks(arr, pi + 1, high);
        }
        
        #pragma omp taskwait
    }
}

// Function to check if array is sorted
int is_sorted(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        if (arr[i] > arr[i+1]) {
            return 0;
        }
    }
    return 1;
}

// Function to copy array
void copy_array(int* src, int* dest, int n) {
    for (int i = 0; i < n; i++) {
        dest[i] = src[i];
    }
}

int main() {
    int i, n = 10000000;
    double serial_time, parallel_sections_time, parallel_tasks_time;
    clock_t start, end;
    
    int *arr = (int*) malloc(n * sizeof(int));
    int *arr_copy1 = (int*) malloc(n * sizeof(int));
    int *arr_copy2 = (int*) malloc(n * sizeof(int));
    
    if (!arr || !arr_copy1 || !arr_copy2) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    
    for (i = 0; i < n; i++) {
        arr[i] = rand() % n;
    }
    
    copy_array(arr, arr_copy1, n);
    copy_array(arr, arr_copy2, n);
    
    // Run and time serial quicksort
    start = clock();
    quickSort(arr, 0, n-1);
    end = clock();
    serial_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Run and time parallel sections quicksort
    start = clock();
    parallel_quicksort_sections(arr_copy1, 0, n-1);
    end = clock();
    parallel_sections_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Run and time parallel tasks quicksort
    start = clock();
    #pragma omp parallel num_threads(2)
    {
        #pragma omp single
        parallel_quicksort_tasks(arr_copy2, 0, n-1);
    }
    end = clock();
    parallel_tasks_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Print results
    printf("\nTempo sequencial: %.3f segundos\n", serial_time);
    printf("\nSecoes paralelas:\n");
    printf("  Tempo: %.3f segundos\n", parallel_sections_time);
    printf("  Aceleracao: %.2fx\n", serial_time / parallel_sections_time);
    printf("  Ordenado: %s\n", is_sorted(arr_copy1, n) ? "Sim" : "Nao");
    printf("\nTarefas paralelas:\n");
    printf("  Tempo: %.3f segundos\n", parallel_tasks_time);
    printf("  Aceleracao: %.2fx\n", serial_time / parallel_tasks_time);
    printf("  Ordenado: %s\n", is_sorted(arr_copy2, n) ? "Sim" : "Nao");
    
    free(arr);
    free(arr_copy1);
    free(arr_copy2);
    
    return 0;
}