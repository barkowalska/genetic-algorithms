#pragma once
#include<cstdint>
#include<cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <memory>
#include <omp.h>


// dodac do crosss na poczatku spr pc jak w mutation
//poprawic gridsize i blocksize
//scalling dodac
//usunac devicesynchromnize
//dodac setglobalseed() wszedzie
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

inline __device__ double changeIf(double basic, double changed, double draft, double refernceValue)
{
    return (draft >= refernceValue)*basic + (draft < refernceValue)*changed;
}

template<typename T>
inline __device__ T tenary(T ifTrue, T ifFalse, bool condtional)
{
    return condtional*ifTrue + (!condtional)*ifFalse;
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

template<uint64_t PopSize, uint64_t ChromosomeSize>
__global__ void evaluateFitnessFunction(PopulationType<PopSize,ChromosomeSize>* MatingPool)
{
        uint64_t idx=blockDim.x*blockIdx.x+threadIdx.x;
        if (idx >= PopSize) return; 

        MatingPool->fitnessValue[idx]=FitnessFunction(&MatingPool->chromosomes[idx*ChromosomeSize]);
}


__device__ unsigned long long seed;
void setGlobalSeed()
{
    unsigned long long h_seed=static_cast<unsigned long long>(time(NULL)); 
    cudaMemcpyToSymbol(seed, &h_seed, sizeof(unsigned long long));
}

    uint64_t BlockSize;

class Execution{
    private:
    inline static uint64_t BlockSize=32;

    public:

    static void SetBlockSize(uint64_t blockSize){BlockSize = blockSize;}
    static uint64_t GetBlockSize() {return BlockSize;}
    static uint64_t CalculateGridSize(uint64_t threadsNum){return (threadsNum+BlockSize-1)/BlockSize;}
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

template<uint64_t PopSize, uint64_t ChromosomeSize>
class Initialization{
    public:
    Initialization() {}
    virtual void operator()(PopulationType<PopSize,ChromosomeSize>* Population) = 0;
};

template<uint64_t IslandNum, uint64_t PopSize, uint64_t ChromosomeSize>
class Migration{
    public:
    Migration(){ }
    virtual void operator()(PopulationType<PopSize,ChromosomeSize>* Population[IslandNum]) = 0;
};

template<uint64_t IslandNum, uint64_t PopSize, uint64_t ChromosomeSize>
class CEA
{
    uint64_t maxgen;
    static_assert((PopSize % 2) == 0);
    public:

    std::shared_ptr<Crossover<PopSize, ChromosomeSize>> m_crossover;
    std::shared_ptr<Mutation<PopSize, ChromosomeSize>> m_mutation;
    std::shared_ptr<Selection<PopSize, ChromosomeSize>> m_selection;
    std::shared_ptr<Initialization<PopSize, ChromosomeSize>> m_initialization;

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


    std::pair<double, std::array<double, ChromosomeSize>> run()
    {
        #pragma omp parallel
        {
            PopulationType<PopSize,ChromosomeSize>* population_=Population[omp_get_thread_num()];
            PopulationType<PopSize,ChromosomeSize>* matingPool_=MatingPool[omp_get_thread_num()];
            uint64_t* selected_=Selected[omp_get_thread_num()];       

            (*m_initialization)(population_);    
            for(uint64_t i=0; i<maxgen; i++)
            {
                (*m_selection)(population_, selected_);
                (*m_crossover)(population_, matingPool_, selected_);
                (*m_mutation)(matingPool_);
                evaluateFitnessFunction<<<>>>

            }

        }
    }
};




}


