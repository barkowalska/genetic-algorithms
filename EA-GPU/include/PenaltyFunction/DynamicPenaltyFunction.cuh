#pragma once
#include <cmath>
#include <iostream>

#include "CEA.cuh"
namespace cea
{
template<uint64_t PopSize, uint64_t ChromosomeSize>
__global__ void DynamicPenaltyFunction_(PopulationType<PopSize, ChromosomeSize>* MatingPool, int m_a, int m_b, double m_c, uint64_t gen)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= PopSize || idx + 1 >= PopSize) return; 

    double penalty=0.0;
    for(uint64_t i=0; i< ChromosomeSize; i++)
    {
        double value=MatingPool->chromosomes[idx*ChromosomeSize +1];

        double violationMin = max(0.0, MIN<ChromosomeSize>[i] - value);
        double violationMax = max(0.0, value - MAX<ChromosomeSize>[i]);
        penalty += pow(violationMin, m_b) + pow(violationMax, m_b);

    }
    double dynamicFactor = m_c * pow(static_cast<double>(gen), m_a);
    penalty =penalty* dynamicFactor;

    MatingPool->fitnessValue[idx] += penalty;
}

template<uint64_t PopSize, uint64_t ChromosomeSize>
class DynamicPenaltyFunction : public PenaltyFunction<PopSize, ChromosomeSize>
{
    private:
    int m_a;
    int m_b;
    double m_c;

    public:
    DynamicPenaltyFunction( int a = 2, int b = 2, double c = 0.5) : m_a(a), m_b(b), m_c(c) {}

    void operator()(PopulationType<PopSize, ChromosomeSize>* MatingPool, uint64_t gen) override
    {

    uint64_t gridSize = Execution::CalculateGridSize(PopSize);
    uint64_t blockSize = Execution::GetBlockSize();

    DynamicPenaltyFunction_<<<gridSize,blockSize, 0,streams[omp_get_thread_num()]>>>( MatingPool,m_a, m_b, m_c, gen);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout<<"CUDA Error: "<< cudaGetErrorString(err)<< std::endl;
    }
    }

};
}