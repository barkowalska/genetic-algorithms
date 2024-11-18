#pragma once

#include "CEA.cuh"

namespace cea
{

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void UniformCrossover_(PopulationType<PopSize, ChromosomeSize>* Population, PopulationType<PopSize, ChromosomeSize>* MatingPool, uint64_t* Selected, double m_Pc, double m_n)
    {
        unsigned int idx = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
        if (idx >= PopSize || idx + 1 >= PopSize) return; 

        curandState state;
        curand_init(seed, idx, 0, &state);

        double* parent_A = &Population->chromosomes[Selected[idx] * ChromosomeSize];
        double* parent_B = &Population->chromosomes[Selected[idx + 1] * ChromosomeSize];

        double* child_A = &MatingPool->chromosomes[Selected[idx] * ChromosomeSize];
        double* child_B = &MatingPool->chromosomes[Selected[idx + 1] * ChromosomeSize];

        for (uint64_t i = 0; i < ChromosomeSize; i++)
        {
            double rand = curand_uniform_double(&state);
            child_A[i]=(rand<0.5)*parent_A[i]+(rand>=0.5)*parent_B[i];
            child_B[i]=(rand<0.5)*parent_B[i]+(rand>=0.5)*parent_A[i];
        }
    }

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class UniformCrossover : public Crossover<PopSize, ChromosomeSize>
    {
        private:
            dim3 m_blockSize; // CUDA block size

        public:

            UniformCrossover() : m_blockSize(PopSize / 2) {}

 
            void operator()(PopulationType<PopSize, ChromosomeSize>* Population, PopulationType<PopSize, ChromosomeSize>* MatingPool, uint64_t* Selected) override
            {
                setGlobalSeed();
                UniformCrossover_<<<1, this->m_blockSize>>>(Population, MatingPool, Selected);

                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                        std::cout<<"CUDA Error: "<< cudaGetErrorString(err)<< std::endl;
                }

                // Synchronize the device
                cudaDeviceSynchronize();
            }

    };
}
