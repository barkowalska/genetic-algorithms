
#pragma once

#include"CEA.cuh"
#include <stdexcept>


namespace cea
{

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void MultiplePointCrossover_(PopulationType<PopSize,ChromosomeSize>* Population, PopulationType<PopSize,ChromosomeSize>* MatingPool, uint64_t* Selected, uint64_t m_numOfPoints)
    {
        unsigned int idx=2*(blockDim.x*blockIdx.x+threadIdx.x);
        if (idx >= PopSize || idx+1>=PopSize) return; 

        curandState state;
        curand_init(seed, idx, 0, &state);

        uint64_t section=ChromosomeSize/m_numOfPoints;

        extern __shared__ uint64_t crossoverPoints[];
        for (uint64_t i = 0; i < m_numOfPoints; ++i) {
            crossoverPoints[i] = curand_uniform(&state)*section + i*section;            
        }
        double* parent_A = &Population->chromosomes[Selected[idx]*ChromosomeSize];
        double* parent_B = &Population->chromosomes[Selected[idx+1]*ChromosomeSize];

        double* child_A = &MatingPool->chromosomes[Selected[idx]*ChromosomeSize];
        double* child_B = &MatingPool->chromosomes[Selected[idx+1]*ChromosomeSize];

        bool swap = true;
        uint16_t cpoint = 0;

        for( uint64_t i=0; i< ChromosomeSize; i++)
        {
            cpoint += (i > crossoverPoints[cpoint]);
            swap = cpoint % 2;
            child_A[i]=swap*parent_A[i] + (!swap)*parent_B[i];
            child_B[i]=(!swap)*parent_A[i] + swap*parent_B[i];
        }
    }



    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class MultiplePointCrossover : public Crossover<Popsize, ChromosomeSize>
    {
        private:
        dim3 m_blockSize;
        uint64_t m_numOfPoints;

        public:
        MultiplePointCrossover(uint64_t numOfPoints) : m_blockSize(PopSize/2), m_numOfPoints(numOfPoints){
            if(numOfPoints>=ChromosomeSize)
            throw std::invalid_argument("numOfPoints ss greater than ChromosomSize");
        }
        void operator()(PopulationType<PopSize,ChromosomeSize>* Population, PopulationType<PopSize,ChromosomeSize>* MatingPool, uint64_t* Selected) override
        {
            setGlobalSeed();
            MultiplePointCrossover_<<<1, this->m_blockSize, m_numOfPoints*sizeof(uint64_t)>>>(Population, MatingPool, Selected, m_numOfPoints);
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