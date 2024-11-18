#pragma once

#include "CEA.cuh"

namespace cea
{


  template<uint64_t PopSize, uint64_t ChromosomeSize>
__global__ void BoltzmannTournamentSelection_(PopulationType<PopSize, ChromosomeSize> *Population, uint64_t* Selected, uint64_t m_tournamentSize, double m_t)
{
    uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= PopSize) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    uint64_t best = static_cast<uint64_t>(curand_uniform_double(&state) * PopSize);
    for (uint64_t i = 0; i < m_tournamentSize; i++)
    {
        uint64_t k = static_cast<uint64_t>(curand_uniform(&state) * PopSize);
        double pi = 1.0 / (1.0 + exp((Population->fitnessValue[k] - Population->fitnessValue[best]) / m_t));
        double rand = curand_uniform_double(&state);
        if (rand > pi) {
            best = k;
        }
    }
    Selected[idx] = best;
}
/*
  Combines tournament selection with Boltzmann scaling, where the temperature `m_t`
  adjusts selection pressure for balanced exploration and exploitation.
  Default constructor deleted.
*/
template<uint64_t PopSize, uint64_t ChromosomeSize>
class BoltzmannTournamentSelection : public Selection<PopSize, ChromosomeSize>
{
  private:
    dim3 m_blockSize; // CUDA block size
  public:
    uint64_t m_tournamentSize; // Number of individuals participating in each tournament
        double m_t;                // Temperature parameter for Boltzmann scaling

        /*
          Constructor
          Arguments:
          - tournamentSize (uint64_t): Number of individuals in each tournament (must be >= 1 and <= PopSize).
          - t (double): Temperature parameter for Boltzmann scaling (must be > 0).
        */
        BoltzmannTournamentSelection(uint64_t tournamentSize=2, double t=2.0) :
            m_tournamentSize(tournamentSize), m_t(t), m_blockSize(PopSize){}

        /*
          Selection operator - selects individuals based on Boltzmann-scaled tournament selection.
          Arguments:
          - Population: Pointer to the population data.
          - Selected: Pointer to the array where selected individual indices will be stored.
        */
        void operator()(PopulationType<PopSize,ChromosomeSize>* Population, uint64_t* Selected) override
        {
            setGlobalSeed();
            // Launch CUDA kernel for selection
            BoltzmannTournamentSelection_<<<1, this->m_blockSize>>>(Population, Selected, m_tournamentSize, m_t);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                    std::cout<<"CUDA Error: "<< cudaGetErrorString(err)<< std::endl;
            }
            cudaDeviceSynchronize();

        }
};



}
