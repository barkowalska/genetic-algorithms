#pragma once

#include"CEA.cuh"

namespace cea
{

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void ArithmeticCrossover_(PopulationType<PopSize,ChromosomeSize>* Population, PopulationType<PopSize,ChromosomeSize>* MatingPool, uint64_t* Selected)
    {
        unsigned int idx=2*(blockDim.x*blockIdx.x+threadIdx.x);
        if (idx >= PopSize) return; 

        curandState state;
        curand_init(seed, idx, 0, &state);
        double* parent_A = &Population->chromosomes[Selected[idx]*ChromosomeSize];
        double* parent_B = &Population->chromosomes[Selected[idx+1]*ChromosomeSize];

        double* child_A = &MatingPool->chromosomes[Selected[idx]*ChromosomeSize];
        double* child_B = &MatingPool->chromosomes[Selected[idx+1]*ChromosomeSize];

        for(uint64_t i=0; i<ChromosomeSize; i++)
        {
            double alpha=curand_uniform_double(&state);
            child_A[i] = alpha*parent_A[i] + (1.0-alpha)*parent_B[i];
            child_B[i] = (1.0-alpha)*parent_A[i] +alpha*parent_B[i] 
        }


    }
    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class ArithmeticCrossover : public Crossover<Popsize, ChromosomeSize>
    {
        private:
        dim3 m_blockSize;
        public:
        ArithmeticCrossover() : m_blockSize(PopSize/2){}
    void operator()(PopulationType<PopSize,ChromosomeSize>* Population, PopulationType<PopSize,ChromosomeSize>* MatingPool, uint64_t* Selected) override
    {
        setGlobalSeed();
        ArithmeticCrossover_<<<1, this->m_blockSize>>>(Population, MatingPool, Selected);
    }
    
    };
}