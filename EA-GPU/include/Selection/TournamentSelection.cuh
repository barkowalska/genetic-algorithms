#pragma once

#include"../CEA.cuh"
#include<iostream>
namespace cea
{
template<uint64_t PopSize, uint64_t ChromosomeSize>
__global__ void TournamentSelection_(PopulationType<PopSize,ChromosomeSize>* Population, uint64_t* Selected, uint64_t m_tournamentSize)
{
  uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= PopSize) return;

  curandState state;
  curand_init(seed, idx, 0, &state);

  uint64_t best=static_cast<uint64_t>(curand_uniform(&state)* PopSize);
  for(uint64_t i=0; i<m_tournamentSize; i++)
  {
    uint64_t k= static_cast<uint64_t>(curand_uniform(&state)* PopSize);
    if (Population->fitnessValue[k] > Population->fitnessValue[best]) best = k;
  }
  Selected[idx] = best;

}

template<uint64_t PopSize, uint64_t ChromosomeSize>
class TournamentSelection: public Selection<PopSize,ChromosomeSize>
{
  public:
  uint64_t m_tournamentSize;
  dim3 m_blockSize; // CUDA block size

  TournamentSelection(dim3 blockSize, uint64_t tournamentSize): m_tournamentSize(tournamentSize), m_blockSize(blockSize){}
  void operator()(PopulationType<PopSize,ChromosomeSize>* Population, uint64_t* Selected) override
  {
    setGlobalSeed();
    TournamentSelection_<<<1,this->m_blockSize>>>(Population, Selected, m_tournamentSize);
  }
  void printData()
  {
    std::cout << "BLOCK SIZE: " << this->m_blockSize.x << std::endl;
  }
};
}