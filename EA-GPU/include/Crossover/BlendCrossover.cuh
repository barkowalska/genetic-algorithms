
#pragma once

#include"CEA.cuh"

namespace cea
{

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void BlendCrossover_(PopulationType<PopSize,ChromosomeSize>* Population, PopulationType<PopSize,ChromosomeSize>* MatingPool, uint64_t* Selected, double m_alpha)
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
            double x1=(parent_A[i]<parent_B[i])* parent_A[i] + (parent_A[i]>=parent_B[i])*parent_B[i];
            double x2=(parent_A[i]>parent_B[i])* parent_A[i] + (parent_A[i]<=parent_B[i])*parent_B[i];
            double d = x2 - x1;
            double lower_bound = x1 - m_alpha * d;
            double upper_bound = x2 + m_alpha * d;            
            child_A[i] = curand_uniform_double(&state)*(upper_bound-lower_bound)+lower_bound;
            child_B[i] = curand_uniform_double(&state)*(upper_bound-lower_bound)+lower_bound; 
        }
    }


    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class BlendCrossover : public Crossover<Popsize, ChromosomeSize>
    {
        private:
        dim3 m_blockSize;
        double m_alpha; // user-deﬁned parameter that controls the extent of the expansion

        public:
        BlendCrossover(double alpha = 0.5) : m_blockSize(PopSize/2), m_alpha(alpha){}
        void operator()(PopulationType<PopSize,ChromosomeSize>* Population, PopulationType<PopSize,ChromosomeSize>* MatingPool, uint64_t* Selected) override
        {
            setGlobalSeed();
            BlendCrossover_<<<1, this->m_blockSize>>>(Population, MatingPool, Selected, m_alpha);
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