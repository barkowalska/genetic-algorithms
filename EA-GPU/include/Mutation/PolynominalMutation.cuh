#pragma once

#include "CEA.cuh"
namespace cea
{
    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void PolynominalMutation_(PopulationType<PopSize, ChromosomeSize>* MatingPool,double m_n)
    {

        uint64_t idx=blockDim.x*blockIdx.x+threadIdx.x;
        if (idx >= PopSize) return; 

        curandState state;
        curand_init(seed, idx, 0, &state);
        double* chromosome = &MatingPool->chromosomes[idx * ChromosomeSize];

        for(uint64_t i=0; i<ChromosomeSize; i++)
        {
            double delta_max = MAX[i] - MIN[i];
            double randPm = curand_uniform_double(&state);
            double rand = curand_uniform_double(&state);
            double delta_q=(rand<0.5)*(pow(2.0 * rand, 1.0 / m_n + 1.0) - 1.0)+(rand>=0.5)*(1.0 - pow(2 * (1 - rand), 1 / m_n + 1.0));
            chromosome[i]=chromosome[i] + (randPm<ProbabilityMutation)*delta_max*delta_q;
        }

    }


    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class PolynominalMutation : public Mutation<PopSize, ChromosomeSize>
    {
        dim3 m_blockSize; // CUDA block size
        double m_n; // Control parameter that defines the spread and shape of the polynomial distribution

    public:
        PolynominalMutation(double n): :m_blockSize(PopSize),m_n(n){}

        void operator()(PopulationType<PopSize,ChromosomeSize>* MatingPool) override
        {
            PolynominalMutation_<<<1, m_blockSize>>>(MatingPool, m_n);

             cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                    std::cout<<"CUDA Error: "<< cudaGetErrorString(err)<< std::endl;
            }
            cudaDeviceSynchronize();

        }

    };
}