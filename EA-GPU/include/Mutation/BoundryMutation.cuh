#pragma once

#include "CEA.cuh"


namespace cea
{
    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void BoundryMutation_(PopulationType<PopSize, ChromosomeSize>* MatingPool)
    {

        uint64_t idx=blockDim.x*blockIdx.x+threadIdx.x;
        if (idx >= PopSize) return; 

        double* chromosome = &MatingPool->chromosomes[idx * ChromosomeSize];

        for(uint64_t i=0; i<ChromosomeSize; i++)
        {
            double randPm=HybridTaus(clock(),idx,clock(),idx);
            double rand=HybridTaus(clock(),idx,clock(),idx);
            double change=((rand<=0.5)*MIN<ChromosomeSize>[i]+(rand>0.5)*MAX<ChromosomeSize>[i]);

            chromosome[i] = tenary(change, chromosome[i], randPm < ProbabilityMutation);

        }

    }


    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class BoundryMutation : public Mutation<PopSize, ChromosomeSize>
    {
    public:
        void operator()(PopulationType<PopSize,ChromosomeSize>* MatingPool) override
        {

            uint64_t gridSize = Execution::CalculateGridSize(PopSize);
            uint64_t blockSize = Execution::GetBlockSize();
            BoundryMutation_<<<gridSize, blockSize, 0,streams[omp_get_thread_num()]>>>(MatingPool);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                    std::cout<<"CUDA Error: "<< cudaGetErrorString(err)<< std::endl;
            }

        }

    };
}