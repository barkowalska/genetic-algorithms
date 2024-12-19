#pragma once

#include "CEA.cuh"
namespace cea
{
    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class UniformInitialization : public Initialization<PopSize, ChromosomeSize> 
    {
        public:
        void operator()(PopulationType<PopSize, ChromosomeSize>* Population, double* PopulationDataToMax) override
        {
            uint64_t gridSize = Execution::CalculateGridSize(PopSize);
            uint64_t blockSize = Execution::GetBlockSize();
            UniformInitialization_<<<gridSize, blockSize,0,streams[omp_get_thread_num()]>>>(Population, PopulationDataToMax);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                    std::cout<<"CUDA Error: "<< cudaGetErrorString(err)<< std::endl;
            }
            CHECK(cudaGetLastError());

        }

    };

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void UniformInitialization_(PopulationType<PopSize, ChromosomeSize>* Population, double* PopulationDataToMax)
    {
        uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= PopSize) return; 

        for(uint64_t i=0; i<ChromosomeSize; i++)
        {
            double rand=HybridTaus(clock(),idx,clock(),idx);
            Population->chromosomes[idx*ChromosomeSize+i]=rand*(MAX<ChromosomeSize>[i]-MIN<ChromosomeSize>[i])+MIN<ChromosomeSize>[i];
        }        
        Population->fitnessValue[idx]=FitnessFunction(&Population->chromosomes[idx*ChromosomeSize]);


        PopulationDataToMax[idx]=Population->fitnessValue[idx];

    }
}
