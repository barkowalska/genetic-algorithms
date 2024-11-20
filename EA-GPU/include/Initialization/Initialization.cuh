#pragma once

#include "CEA.cuh"
namespace cea
{
    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class UniformInitialization : public Initialization<PopSize, ChromosomeSize> 
    {
        public:
        void operator()(PopulationType<PopSize, ChromosomeSize>* Population) override
        {
            setGlobalSeed();
            uint64_t gridSize = Execution::CalculateGridSize(PopSize);
            uint64_t blockSize = Execution::GetBlockSize();
            Initialization_<<<gridSize, blockSize>>>(Population);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                    std::cout<<"CUDA Error: "<< cudaGetErrorString(err)<< std::endl;
            }
            CHECK(cudaGetLastError());

        }

    };

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void Initialization_(PopulationType<PopSize, ChromosomeSize>* Population)
    {
        uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= PopSize) return; 

        curandState state;
        curand_init(seed, idx, 0, &state);
        for(uint64_t i=0; i<ChromosomeSize; i++)
        {
            double rand=curand_uniform_double(&state);
            Population->chromosomes[idx*ChromosomeSize+i]=rand*(MAX<ChromosomeSize>[i]-MIN<ChromosomeSize>[i])+MIN<ChromosomeSize>[i];
        }        
        Population->fitnessValue[idx]=FitnessFunction(&Population->chromosomes[idx*ChromosomeSize]);
        

    }
}
