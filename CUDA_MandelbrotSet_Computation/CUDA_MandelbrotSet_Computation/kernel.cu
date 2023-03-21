/*
CUDA Programs
By J RAJAN
For Learners
Title: MANDELBROT SET COMPUTATION
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define WIDTH 1920   //Adjust as per your screen
#define HEIGHT 1080
#define MAX_ITERATIONS 256
#define BLOCK_SIZE 16

__device__ int mandelbrot_iterations(float x0, float y0) 
{
    float x = 0.0f;
    float y = 0.0f;
    int i;
    for (i = 0; i < MAX_ITERATIONS; i++) 
    {
        float x_temp = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = x_temp;
        if (x * x + y * y > 4.0f) 
        {
            break;
        }
    }
    return i;
}

__global__ void mandelbrot_kernel(int* output) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < WIDTH && y < HEIGHT) 
    {
        float fx = (float)x / (float)WIDTH * 3.0f - 2.0f;
        float fy = (float)y / (float)HEIGHT * 2.0f - 1.0f;
        int iterations = mandelbrot_iterations(fx, fy);
        output[y * WIDTH + x] = iterations;
    }
}

int main() 
{
    int* h_output, * d_output;
    h_output = (int*)malloc(WIDTH * HEIGHT * sizeof(int));
    cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(int));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    mandelbrot_kernel <<<grid, block >>> (d_output);
    cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    // Print out the Mandelbrot set
    for (int y = 0; y < HEIGHT; y++) 
    {
        for (int x = 0; x < WIDTH; x++) 
        {
            int iterations = h_output[y * WIDTH + x];
            if (iterations == MAX_ITERATIONS) 
            {
                printf("#");  //Change to * or . or O or anything for different visual results
            }
            else 
            {
                printf(" ");
            }
        }
        printf("\n");
    }

    free(h_output);
    cudaFree(d_output);
    return 0;
}
