#pragma once

#include "CEA.cuh"

namespace cea
{

  template<uint64_t PopSize, uint64_t ChromosomeSize>
  __global__ void SimulatedBinaryCrossover_(PopulationType<PopSize, ChromosomeSize>* Population, PopulationType<PopSize, ChromosomeSize>* MatingPool, uint64_t* Selected, double m_Pc, double m_n)
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
            double u = curand_uniform_double(&state);
            double beta = (u <= 0.5) * pow(2.0 * u, 1.0 / (m_n + 1.0)) + (u < 0.5) * pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (m_n + 1.0));
            bool crossover = (curand_uniform_double(&state) <= m_Pc);
            child_A[i] = crossover * (0.5 * (parent_A[i] + parent_B[i]) + 0.5 * beta * (parent_A[i] - parent_B[i])) + (!crossover) * parent_A[i];
            child_B[i] = crossover * (0.5 * (parent_A[i] + parent_B[i]) + 0.5 * beta * (parent_B[i] - parent_A[i])) + (!crossover) * parent_B[i];
        }
    }

    /*
      Simulated Binary Crossover (SBX) class.
      Uses a calculated β coefficient to adaptively generate offspring with
      controlled similarity to the parents.
    */
    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class SimulatedBinaryCrossover : public Crossover<PopSize, ChromosomeSize>
    {
        private:
            double m_n;       // Distribution index that defines the spread of the offspring
            double m_Pc;      // Crossover probability

        public:
            /*
              Constructor
              Arguments:
              - n (double): Distribution index (default value is 2.0).
              - crossover_prob (double): Probability of crossover (default value is 0.9).
            */
            SimulatedBinaryCrossover(double n = 2.0, double crossover_prob = 0.9) : m_n(n), m_Pc(crossover_prob) {}

            /*
              Overloaded operator() to perform crossover.
              Arguments:
              - Population: Pointer to the population data.
              - MatingPool: Pointer to the mating pool where offspring will be stored.
              - Selected: Array of indices of selected individuals.
            */
            void operator()(PopulationType<PopSize, ChromosomeSize>* Population, PopulationType<PopSize, ChromosomeSize>* MatingPool, uint64_t* Selected) override
            {
                setGlobalSeed();
                // Launch CUDA kernel for crossover
                uint64_t gridSize = Execution::CalculateGridSize(PopSize/2);
                uint64_t blockSize = Execution::GetBlockSize();
                SimulatedBinaryCrossover_<<<gridSize,blockSize, 0,streams[omp_get_thread_num()]>>>(Population, MatingPool, Selected, m_Pc, m_n);

                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                        std::cout<<"CUDA Error: "<< cudaGetErrorString(err)<< std::endl;
                }

            }

    };
}
