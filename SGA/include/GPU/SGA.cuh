#ifndef SGA_CUH
#define SGA_CUH


#include "cuda_runtime.h"
#include <cmath>
#include <algorithm>
#include <ctime>
#include <cub/cub.cuh>
#include <curand_kernel.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        std::cout << "Error in file: " << __FILE__ << " at line: " << __LINE__ << std::endl;\
        std::cout << "CUDA Error " << error << ": " << cudaGetErrorString(error) << std::endl;\
        exit(1);\
    }\
}





struct Population
{
    double *quality;
    unsigned char* location;
    double* phenotype;
    double* pi;
};

typedef double(*functionPointer_t)(double);
__device__ functionPointer_t fitnessFunction;



__global__ void initialize(Population *population);

__global__ void construct_individuals( Population* population);

__device__ void calculate_pi(Population *population, double* sum);

__global__ void selectionRWS(Population* population,double* sum, Population* mating_pool);

__global__ void cross_over(Population* mating_pool);

__global__ void mutation(Population* mating_pool);

void sum(Population*_d_population, double** d_totalSum, int h_popsize);

std::pair<double,double> run(dim3 block, dim3 grid,Population* d_population, Population* d_mating_pool, int h_popsize, int h_maxgen);

__global__ void attribution(Population* population, Population* mating_pool);

int findMax(double* quality, int h_popsize);

__global__ void setFitnessFunction();

void setConstantMemory(int popsize, int l_genes, float pc, float pm, 
int maxgen, int min, int max);


#endif