/*
CUDA Programs
By J RAJAN
For Learners
Title: Addition of 2 Arrays in Parallel
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define MAX 100 //Maximum length of array

__global__ void add(int* a, int* b, int* c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    int N = 10; //Actual defined size of array
    int a[MAX], b[MAX], c[MAX];
    int* d_a, * d_b, * d_c;

    //Memory allocation on GPU
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    for (int i = 0; i < N; i++) 
    {
        a[i] = i;
        b[i] = i * i; //squaring for second array
    }

    //Both arrays copied to GPU memory 
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    add <<<1, N>>> (d_a, d_b, d_c); //Kernel Function for addition. N number of parallel computations

    //Send data back to host memory
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost); 

    //Display 
    for (int i = 0; i < N; i++) 
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    //Deallocate memory on GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
