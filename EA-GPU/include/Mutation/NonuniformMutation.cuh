#pragma once

#include "CEA.cuh"
namespace cea
{
    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void NonuniformMutation_(PopulationType<PopSize, ChromosomeSize>* MatingPool,uint64_t m_maxgen, double m_b, uint64_t m_gen)
    {

        uint64_t idx=blockDim.x*blockIdx.x+threadIdx.x;
        if (idx >= PopSize) return; 

        curandState state;
        curand_init(seed, idx, 0, &state);
        double y=0;
        double p = pow(1.0 - static_cast<double>(m_gen) / m_maxgen, m_b);
        double* chromosome = &MatingPool->chromosomes[idx * ChromosomeSize];
        for(uint64_t i=0; i<ChromosomeSize; i++)
        {
            double randPm=curand_uniform_double(&state);
            double rand = curand_uniform_double(&state);
            y = (rand>=0.5)*(MAX[i] - chromosome[i])+(rand<0.5)*(chromosome[i]-MIN[i]);
            double delta = y * (1.0 - pow(curand_uniform_double(&state), p));
            double change=(rand>= 0.5)*delta-(rand<0.5)*delta;
            chromosome[i]=chromosome[i]-(randPm<ProbabilityMutation)*change;
        }

    }


    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class NonuniformMutation : public Mutation<PopSize, ChromosomeSize>
    {
        dim3 m_blockSize; // CUDA block size
        double m_b; // Parameter to control the annealing speed (degree of non-uniformity)
        size_t m_gen; // Current generation
        size_t m_maxgen; // Maximum number of generations

    public:
        NonuniformMutation(uint64_t maxgen, double b, uint64_t gen)
            :m_blockSize(PopSize), m_gen(gen), m_b(b), m_maxgen(maxgen) {}
        void operator()(PopulationType<PopSize,ChromosomeSize>* MatingPool) override
        {
            NonuniformMutation_<<<1, m_blockSize>>>(MatingPool, m_maxgen, m_b, m_gen);

             cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                    std::cout<<"CUDA Error: "<< cudaGetErrorString(err)<< std::endl;
            }
            cudaDeviceSynchronize();

        }

    };
}