/*
CUDA Programs
By J RAJAN
For Learners
Title: GAME OF LIFE 
(CONWAY'S GAME OF LIFE)
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>
#include "device_launch_parameters.h"

//Grid Dimensions
#define WIDTH 50
#define HEIGHT 50

#define GENERATIONS 1000

using namespace std;

__global__ void game_of_life(int* grid, int* new_grid) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * WIDTH + x;

    int neighbors = 0;

    for (int i = -1; i <= 1; i++) 
    {
        for (int j = -1; j <= 1; j++) 
        {
            int x_n = (x + i + WIDTH) % WIDTH;
            int y_n = (y + j + HEIGHT) % HEIGHT;
            int neighbor_index = y_n * WIDTH + x_n;
            if (grid[neighbor_index]) 
            {
                neighbors++;
            }
        }
    }

    if (grid[index]) 
    {
        neighbors--;
        if (neighbors < 2 || neighbors > 3) 
        {
            new_grid[index] = 0;
        }
        else 
        {
            new_grid[index] = 1;
        }
    }
    else 
    {
        if (neighbors == 3) 
        {
            new_grid[index] = 1;
        }
        else 
        {
            new_grid[index] = 0;
        }
    }
}

//Display generation
void print_grid(int* grid) 
{
    for (int y = 0; y < HEIGHT; y++) 
    {
        for (int x = 0; x < WIDTH; x++) 
        {
            if (grid[y * WIDTH + x]) 
            {
                cout<<("# ");
            }
            else 
            {
                cout<<("  ");
            }
        }
        cout<<("\n");
    }
    cout<<("\n");
}

void main() 
{
    int* grid, * new_grid;
    cudaMallocManaged(&grid, WIDTH * HEIGHT * sizeof(int));
    cudaMallocManaged(&new_grid, WIDTH * HEIGHT * sizeof(int));

    srand(time(NULL));

    for (int y = 0; y < HEIGHT; y++) 
    {
        for (int x = 0; x < WIDTH; x++) 
        {
            grid[y * WIDTH + x] = rand() % 2;
        }
    }

    dim3 block_size(32, 32);
    dim3 grid_size((WIDTH + block_size.x - 1) / block_size.x, (HEIGHT + block_size.y - 1) / block_size.y);

    for (int i = 0; i < 1000; i++) 
    {
        cout << "Generation" << i << "\n";
        print_grid(grid);
        cout << "\033[2J\033[1;1H"; //For clearing screen (Shifting view above) and show next generation. 
        game_of_life <<<grid_size, block_size >>> (grid, new_grid);

        int* temp = grid;
        grid = new_grid;
        new_grid = temp;

        cudaDeviceSynchronize();
    }

    cudaFree(grid);
    cudaFree(new_grid);
}