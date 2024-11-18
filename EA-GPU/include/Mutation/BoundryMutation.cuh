#pragma once

#include "CEA.cuh"
namespace cea
{
    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void BoundryMutation_(PopulationType<PopSize, ChromosomeSize>* MatingPool)
    {

        uint64_t idx=blockDim.x*blockIdx.x+threadIdx.x;
        if (idx >= PopSize) return; 

        curandState state;
        curand_init(seed, idx, 0, &state);
        double* chromosome = &MatingPool->chromosomes[idx * ChromosomeSize];

        for(uint64_t i=0; i<ChromosomeSize; i++)
        {
            double randPm=curand_uniform_double(&state);
            double rand=curand_uniform_double(&state);
            double change=((rand<=0.5)*MIN[i]+(rand>0.5)*MAX[i]);

            chromosome[i]=(randPm>=ProbabilityMutation)*chromosome[i]+(randPm< ProbabilityMutation)*change;
        }

    }


    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class BoundryMutation : public Mutation<PopSize, ChromosomeSize>
    {
        dim3 m_blockSize; // CUDA block size

    public:
        BoundryMutation() :m_blockSize(PopSize){}
        void operator()(PopulationType<PopSize,ChromosomeSize>* MatingPool) override
        {
            BoundryMutation_<<<1, m_blockSize>>>(MatingPool);

             cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                    std::cout<<"CUDA Error: "<< cudaGetErrorString(err)<< std::endl;
            }
            cudaDeviceSynchronize();

        }

    };
}