#pragma once

#include"CEA.cuh"

namespace cea
{

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void SimulatedBinaryCrossover_(PopulationType<PopSize,ChromosomeSize>* Population, PopulationType<PopSize,ChromosomeSize>* MatingPool, uint64_t* Selected, double m_Pc, double m_n)
    {
        unsigned int idx=2*(blockDim.x*blockIdx.x+threadIdx.x);
        if (idx >= PopSize || idx+1>=PopSize) return; 

        curandState state;
        curand_init(seed, idx, 0, &state);
        double* parent_A = &Population->chromosomes[Selected[idx]*ChromosomeSize];
        double* parent_B = &Population->chromosomes[Selected[idx+1]*ChromosomeSize];

        double* child_A = &MatingPool->chromosomes[Selected[idx]*ChromosomeSize];
        double* child_B = &MatingPool->chromosomes[Selected[idx+1]*ChromosomeSize];

        for(uint64_t i=0; i<ChromosomeSize; i++)
        {
            double u=curand_uniform_double(&state);

            double beta=(u<=0.5)*pow(2.0 * u, 1.0 / (m_n + 1.0))+(u<0.5)*pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (m_n + 1.0));
            bool crossover=(curand_uniform_double(&state) <= m_Pc);
            child_A[i] =crossover*(0.5* (parent_A[i] + parent_B[i] )+ 0.5*beta* (parent_A[i]-parent_B[i]))+(!crossover)*parent_A[i];
            child_B[i] =crossover*(0.5* (parent_A[i] + parent_B[i] )+ 0.5*beta* (parent_B[i]-parent_A[i]))+(!crossover)*parent_B[i];

        
        }
    }


    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class SimulatedBinaryCrossover : public Crossover<Popsize, ChromosomeSize>
    {
        private:
        dim3 m_blockSize;
        double m_n;// Distribution index that defines the spread of the offspring
        double m_Pc;  // Crossover probability

        public:
        SimulatedBinaryCrossover(double n = 2.0, double crossover_prob = 0.9) : m_blockSize(PopSize/2),  m_n(n), m_Pc(crossover_prob){}
        void operator()(PopulationType<PopSize,ChromosomeSize>* Population, PopulationType<PopSize,ChromosomeSize>* MatingPool, uint64_t* Selected) override
        {
            setGlobalSeed();
            SimulatedBinaryCrossover_<<<1, this->m_blockSize>>>(Population, MatingPool, Selected, m_Pc, m_n);
        }
        // Check for kernel launch errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
            }

            // Synchronize the device
            cudaDeviceSynchronize();
        
    
    };
}