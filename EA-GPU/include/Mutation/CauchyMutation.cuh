#pragma once

#include "CEA.cuh"
namespace cea
{
    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void CauchyMutation_(PopulationType<PopSize, ChromosomeSize>* MatingPool, double m_sigma)
    {

        uint64_t idx=blockDim.x*blockIdx.x+threadIdx.x;
        if (idx >= PopSize) return; 

        double* chromosome = &MatingPool->chromosomes[idx * ChromosomeSize];

        for(uint64_t i=0; i<ChromosomeSize; i++)
        {
            double randPm=HybridTaus(clock(),idx,clock(),idx);
            double u = HybridTaus(clock(),idx,clock(),idx);
            double randCauchy=tan(M_PI * (u - 0.5));

            chromosome[i]=chromosome[i]+(randPm<ProbabilityMutation)*randCauchy*m_sigma;
        }

    }


    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class CauchyMutation : public Mutation<PopSize, ChromosomeSize>
    {
        double m_sigma; //Scale parameter for mutation magnitude

    public:
        CauchyMutation(double sigma) : m_sigma(sigma){}

        void operator()(PopulationType<PopSize,ChromosomeSize>* MatingPool) override
        {
            uint64_t gridSize = Execution::CalculateGridSize(PopSize);
            uint64_t blockSize = Execution::GetBlockSize();
            CauchyMutation_<<<gridSize, blockSize, 0,streams[omp_get_thread_num()]>>>(MatingPool, m_sigma);

             cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                    std::cout<<"CUDA Error: "<< cudaGetErrorString(err)<< std::endl;
            }

        }

    };
}