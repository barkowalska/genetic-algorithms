#pragma once

#include "CEA.cuh"
namespace cea
{
    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void UniformMutation_(PopulationType<PopSize, ChromosomeSize>* MatingPool)
    {

        uint64_t idx=blockDim.x*blockIdx.x+threadIdx.x;
        if (idx >= PopSize) return; 

        curandState state;
        curand_init(seed, idx, 0, &state);

        double* chromosome = &MatingPool->chromosomes[idx * ChromosomeSize];
        for(uint64_t i=0; i<ChromosomeSize; i++)
        {
            double randPm=curand_uniform_double(&state);
            double change=curand_uniform_double(&state)*(MAX<ChromosomeSize>[i]-MIN<ChromosomeSize>[i])+MIN<ChromosomeSize>[i];
            chromosome[i]=chromosome[i]*(randPm>=ProbabilityMutation)+(randPm<ProbabilityMutation)*change;
        }

    }


    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class UniformMutation : public Mutation<PopSize, ChromosomeSize>
    {

    public:
        void operator()(PopulationType<PopSize,ChromosomeSize>* MatingPool) override
        {
            setGlobalSeed();
            uint64_t gridSize = Execution::CalculateGridSize(PopSize);
            uint64_t blockSize = Execution::GetBlockSize();
            UniformMutation_<<<gridSize, blockSize>>>(MatingPool);

             cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                    std::cout<<"CUDA Error: "<< cudaGetErrorString(err)<< std::endl;
            }

        }

    };
}