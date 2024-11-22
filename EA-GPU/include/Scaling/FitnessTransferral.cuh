#pragma once

#include"CEA.cuh"
namespace cea
{
template<uint64_t PopSize, uint64_t ChromosomeSize>
__global__ void FitnessTransferral_(PopulationType<PopSize,ChromosomeSize>* Population)
{
  uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= PopSize) return;

   Population->fitnessValue[idx] =CurrentBest.value -Population->fitnessValue[idx];


}

template<uint64_t PopSize, uint64_t ChromosomeSize>
class FitnessTransferral: public Scaling<PopSize,ChromosomeSize>
{
  public:

  void operator()(PopulationType<PopSize,ChromosomeSize>* Population) override
  {
    setGlobalSeed();
      uint64_t gridSize = Execution::CalculateGridSize(PopSize);
      uint64_t blockSize = Execution::GetBlockSize();
      FitnessTransferral_<<<gridSize,blockSize>>>(Population);
  }

};
}