
#pragma once

#include"CEA.cuh"

namespace cea
{

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void BlendCrossover_(PopulationType<PopSize,ChromosomeSize>* Population, PopulationType<PopSize,ChromosomeSize>* MatingPool, uint64_t* Selected, double m_alpha)
    {
        unsigned int idx=2*(blockDim.x*blockIdx.x+threadIdx.x);
        if (idx >= PopSize || idx+1>=PopSize) return; 


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
            child_A[i] = HybridTaus(clock(),idx,clock(),idx)*(upper_bound-lower_bound)+lower_bound;
            child_B[i] = HybridTaus(clock(),idx,clock(),idx)*(upper_bound-lower_bound)+lower_bound; 
        }
    }


    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class BlendCrossover : public Crossover<PopSize, ChromosomeSize>
    {
        private:
        double m_alpha; // user-deﬁned parameter that controls the extent of the expansion

        public:
        BlendCrossover(double alpha = 0.5) :  m_alpha(alpha){}
        void operator()(PopulationType<PopSize,ChromosomeSize>* Population, PopulationType<PopSize,ChromosomeSize>* MatingPool, uint64_t* Selected) override
        {
                uint64_t gridSize = Execution::CalculateGridSize(PopSize/2);
                uint64_t blockSize = Execution::GetBlockSize();
                BlendCrossover_<<<gridSize, blockSize, 0,streams[omp_get_thread_num()]>>>(Population, MatingPool, Selected, m_alpha);
        
                // Check for kernel launch errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cout<<"CUDA Error: "<< cudaGetErrorString(err)<< std::endl;
            }
       
        }
        
    };
}