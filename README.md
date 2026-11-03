# PCA-EXP-2-Matrix-Summation-using-2D-Grids-and-2D-Blocks-AY-23-24

<h3>AIM:</h3>
<h3>ENTER YOUR NAME: Dilli Prathap</h3>
<h3>ENTER YOUR REGISTER NO: 212224110014</h3>
<h3>EX. NO: 02</h3>
<h3>DATE: 02-11-2026</h3>
<h1> <align=center> MATRIX SUMMATION WITH A 2D GRID AND 2D BLOCKS </h3>
i.  Use the file sumMatrixOnGPU-2D-grid-2D-block.cu
ii. Matrix summation with a 2D grid and 2D blocks. Adapt it to integer matrix addition. Find the best execution configuration. </h3>

## AIM:
To perform  matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1.	Initialize the data: Generate random data for two input arrays using the initialData function.
2.	Perform the sum on the host: Use the sumMatrixOnHost function to calculate the sum of the two input arrays on the host (CPU) for later verification of the GPU results.
3.	Allocate memory on the device: Allocate memory on the GPU for the two input arrays and the output array using cudaMalloc.
4.	Transfer data from the host to the device: Copy the input arrays from the host to the device using cudaMemcpy.
5.	Set up the execution configuration: Define the size of the grid and blocks. Each block contains multiple threads, and the grid contains multiple blocks. The total number of threads is equal to the size of the grid times the size of the block.
6.	Perform the sum on the device: Launch the sumMatrixOnGPU2D kernel on the GPU. This kernel function calculates the sum of the two input arrays on the device (GPU).
7.	Synchronize the device: Use cudaDeviceSynchronize to ensure that the device has finished all tasks before proceeding.
8.	Transfer data from the device to the host: Copy the output array from the device back to the host using cudaMemcpy.
9.	Check the results: Use the checkResult function to verify that the output array calculated on the GPU matches the output array calculated on the host.
10.	Free the device memory: Deallocate the memory that was previously allocated on the GPU using cudaFree.
11.	Free the host memory: Deallocate the memory that was previously allocated on the host.
12.	Reset the device: Reset the device using cudaDeviceReset to ensure that all resources are cleaned up before the program exits.

## PROGRAM:
## INT:
```c

%%cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Initialize float data
void initialData(float *ip, const int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// CPU Matrix Addition
void sumMatrixOnHost(float *A, float *B, float *C,
                     const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }
}

// Verify CPU and GPU results
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    float epsilon = 1.0E-5f;
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (fabsf(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = false;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

// GPU Matrix Addition Kernel
__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C,
                                 int NX, int NY)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Matrix size
    int nx = 1 << 14;
    int ny = 1 << 14;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // Allocate host memory
    float *h_A, *h_B, *hostRef, *gpuRef;

    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    if (!h_A || !h_B || !hostRef || !gpuRef)
    {
        printf("Host memory allocation failed!\n");
        return 1;
    }

    // Initialize data
    double iStart = seconds();

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    double iElaps = seconds() - iStart;

    printf("Matrix initialization elapsed %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // CPU matrix addition
    iStart = seconds();

    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

    iElaps = seconds() - iStart;

    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    // Allocate GPU memory
    float *d_MatA, *d_MatB, *d_MatC;

    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    // Configure CUDA kernel
    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);

    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    // Execute GPU kernel
    iStart = seconds();

    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);

    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;

    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);

    CHECK(cudaGetLastError());

    // Copy result from GPU to host
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // Verify results
    checkResult(hostRef, gpuRef, nxy);

    // Free GPU memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    // Free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // Reset device
    CHECK(cudaDeviceReset());

    return 0;
}

```
## FLOAT:

```c

%%cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif

// Initialize float data
void initialData(float *ip, const int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// CPU Matrix Addition
void sumMatrixOnHost(float *A, float *B, float *C,
                     const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }
}

// Verify CPU and GPU results
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    float epsilon = 1.0E-5f;
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (fabsf(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = false;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

// CUDA GPU Kernel using 2D Grid and 2D Blocks
__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C,
                                 int NX, int NY)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // Set up CUDA device
    int dev = 0;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    printf("Using Device %d: %s\n", dev, deviceProp.name);

    CHECK(cudaSetDevice(dev));

    // Set up matrix size
    int nx = 1 << 14;
    int ny = 1 << 14;

    int nxy = nx * ny;
    size_t nBytes = (size_t)nxy * sizeof(float);

    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // Allocate host memory
    float *h_A, *h_B, *hostRef, *gpuRef;

    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    if (h_A == NULL || h_B == NULL ||
        hostRef == NULL || gpuRef == NULL)
    {
        printf("Host memory allocation failed!\n");
        return 1;
    }

    // Initialize data
    double iStart = seconds();

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    double iElaps = seconds() - iStart;

    printf("Matrix initialization elapsed %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // CPU matrix addition
    iStart = seconds();

    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

    iElaps = seconds() - iStart;

    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    // Allocate GPU memory
    float *d_MatA, *d_MatB, *d_MatC;

    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    // Copy host data to GPU
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes,
                     cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(d_MatB, h_B, nBytes,
                     cudaMemcpyHostToDevice));

    // Configure 2D Grid and 2D Blocks
    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);

    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    // Execute GPU kernel
    iStart = seconds();

    sumMatrixOnGPU2D<<<grid, block>>>(
        d_MatA, d_MatB, d_MatC, nx, ny);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;

    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n",
           grid.x, grid.y,
           block.x, block.y, iElaps);

    // Copy GPU result to host
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes,
                     cudaMemcpyDeviceToHost));

    // Verify results
    checkResult(hostRef, gpuRef, nxy);

    // Free GPU memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    // Free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // Reset CUDA device
    CHECK(cudaDeviceReset());

    return 0;
}

```

## OUTPUT:
## int:
<img width="922" height="222" alt="Screenshot 2026-11-03 102315" src="https://github.com/user-attachments/assets/c5b76642-bef0-4a59-b0c6-a46dc84e9e61" />

## float:
<img width="877" height="303" alt="Screenshot 2026-11-03 103011" src="https://github.com/user-attachments/assets/7c306643-ce07-4c23-92d2-c75c32a20513" />

## RESULT:
The experiment was performed using a Tesla T4 GPU with a matrix size of 16384 × 16384. The integer implementation recorded a GPU execution time of **0.014325** seconds, whereas the floating-point implementation recorded **0.013577** seconds. Both CPU and GPU results were verified successfully, and the arrays matched. Hence, CUDA parallel processing successfully performed matrix addition using both data types.

The host took **0.853780** seconds to complete its computation, while the GPU outperforms the host and completes the computation in **0.013577** seconds. Therefore, float variables in the GPU provide better performance in this experiment. Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
