#ifndef SGA_CUH
#define SGA_CUH


#include "cuda_runtime.h"
#include <cmath>
#include <algorithm>
#include <ctime>
#include <cub/cub.cuh>

#include <curand_kernel.h>


// Makro do sprawdzania błędów CUDA
#define CHECK(call)\
{\
    const cudaError_t error=call;\
    if(error!=cudaSuccess)\
    {\
        printf("Error: %s: %d, "__FILE__, __LINE__);\
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}

extern __device__ unsigned long long seed;
extern __constant__ int d_popsize;
extern __constant__ int d_l_genes;
extern __constant__ float d_pc;
extern __constant__ float d_pm;
extern __constant__ int d_maxgen;
extern __constant__ int d_min;
extern __constant__ int d_max;

struct Population
{
    double *quality;
    unsigned char* location;
    double* phenotype;
    double* pi;
};

typedef double(*functionPointer_t)(double);
__device__ functionPointer_t fitnessFunction;


__device__ double func_kwd(double x)
{
    return x*x;
}

__global__ void initialize(Population *population);

__device__ void construct_individuals( Population* population);

__device__ void pi(Population *population, double* sum);

__global__ void selectionRWS(Population* population,double sum, Population* mating_pool);

__global__ void cross_over(Population* mating_pool);

__global__ void mutattion(Population* mating_pool);

void sum(double* wskqualitySum, double** d_totalSum, int h_popsize);

void run(dim3 block, dim3 grid,Population* d_population, Population* d_mating_pool, int h_popsize);


#endif