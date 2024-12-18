#pragma once

#include "CEA.cuh"

namespace cea
{

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void UniformCrossover_(PopulationType<PopSize, ChromosomeSize>* Population, PopulationType<PopSize, ChromosomeSize>* MatingPool, uint64_t* Selected)
    {
        unsigned int idx = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
        if (idx >= PopSize || idx + 1 >= PopSize) return; 

        curandState state;
        curand_init(seed, idx, 0, &state);

        double* parent_A = &Population->chromosomes[Selected[idx] * ChromosomeSize];
        double* parent_B = &Population->chromosomes[Selected[idx + 1] * ChromosomeSize];

        double* child_A = &MatingPool->chromosomes[idx * ChromosomeSize];
        double* child_B = &MatingPool->chromosomes[(idx + 1)* ChromosomeSize];

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

        public:

            UniformCrossover(){}

 
            void operator()(PopulationType<PopSize, ChromosomeSize>* Population, PopulationType<PopSize, ChromosomeSize>* MatingPool, uint64_t* Selected) override
            {
                setGlobalSeed();

                uint64_t gridSize = Execution::CalculateGridSize(PopSize/2);
                uint64_t blockSize = Execution::GetBlockSize();
                UniformCrossover_<<<gridSize, blockSize, 0,streams[omp_get_thread_num()]>>>(Population, MatingPool, Selected);

                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                        std::cout<<"CUDA Error: "<< cudaGetErrorString(err)<< std::endl;
                }

            }

    };
}
