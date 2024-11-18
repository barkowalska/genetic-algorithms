#pragma once
#include<cstdint>
#include<cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>



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

namespace cea
{

__constant__ double ProbabilityMutation;

template<uint64_t ChromosomeSize>
__constant__ double MIN[ChromosomeSize]; 

template<uint64_t ChromosomeSize>
__constant__ double MAX[ChromosomeSize]; 

typedef double(*fitnessFunction_ptr)(double*);
__device__ fitnessFunction_ptr FitnessFunction;

template<uint64_t PopSize, uint64_t ChromosomeSize>
struct PopulationType{
        double chromosomes[PopSize*ChromosomeSize];
        double fitnessValue[PopSize];
        const uint64_t chromosomeSize = ChromosomeSize;
        const uint64_t popSize = PopSize;
};

__device__ unsigned long long seed;
void setGlobalSeed()
{
    unsigned long long h_seed=static_cast<unsigned long long>(time(NULL)); 
    cudaMemcpyToSymbol(seed, &h_seed, sizeof(unsigned long long));
}

template<uint64_t IslandNum, uint64_t PopSize, uint64_t ChromosomeSize>
class CEA
{

    static_assert((PopSize % 2) == 0);
    public:

    PopulationType<PopSize,ChromosomeSize>* Population[IslandNum];
    PopulationType<PopSize,ChromosomeSize>* MatingPool[IslandNum];
    uint64_t* Selected[IslandNum];

    CEA(fitnessFunction_ptr fitnessFunction){
       // Allocate device memory for each island
        for (uint64_t i = 0; i < IslandNum; ++i)
        {
            cudaMalloc(&Population[i],sizeof(PopulationType<PopSize,ChromosomeSize>));
            cudaMalloc(&MatingPool[i],sizeof(PopulationType<PopSize,ChromosomeSize>));

            cudaMalloc(&Selected[i], PopSize * sizeof(uint64_t));
        } 
        cudaMemcpyToSymbol(FitnessFunction, &fitnessFunction, sizeof(fitnessFunction_ptr));
    }

    void setContraints(double* min, double* max)
    {
        cudaMemcpyToSymbol(MIN<ChromosomeSize>, min, ChromosomeSize*sizeof(double));
        cudaMemcpyToSymbol(MAX<ChromosomeSize>, max, ChromosomeSize*sizeof(double));
    }

    // Destructor
    ~CEA()
    {
        // Free device memory
        for (uint64_t i = 0; i < IslandNum; ++i)
        {
            cudaFree(Population[i]);
            cudaFree(MatingPool[i]);
            cudaFree(Selected[i]);
        }
    }
};



template<uint64_t PopSize, uint64_t ChromosomeSize>
class Selection{
    public:
    Selection(){ }
    virtual void operator()(PopulationType<PopSize,ChromosomeSize>* Population, uint64_t* Selected) = 0;
};

template<uint64_t PopSize, uint64_t ChromosomeSize>
class Crossover{
    public:
    Crossover() { }
    virtual void operator()(PopulationType<PopSize,ChromosomeSize>* Population,PopulationType<PopSize,ChromosomeSize>* MatingPool, uint64_t* Selected) = 0;
};

template<uint64_t PopSize, uint64_t ChromosomeSize>
class Mutation{
    public:
    Mutation() { }
    virtual void operator()(PopulationType<PopSize,ChromosomeSize>* MatingPool) = 0;
};

}


