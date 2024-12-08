#pragma once

#include"CEA.cuh"
namespace cea
{
template<uint64_t PopSize, uint64_t ChromosomeSize>
__global__ void PowerLawScaling_(PopulationType<PopSize,ChromosomeSize>* MatingPool, double m_alpha)
{
  uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= PopSize) return;

   MatingPool->fitnessValue[idx] =pow(MatingPool->fitnessValue[idx], m_alpha);


}

template<uint64_t PopSize, uint64_t ChromosomeSize>
class PowerLawScaling: public Scaling<PopSize,ChromosomeSize>
{
  public:
  double m_alpha;// Exponent parameter that controls the degree of scaling

    PowerLawScaling(double alpha=1.5) : m_alpha(alpha){}
  void operator()(PopulationType<PopSize,ChromosomeSize>* MatingPool) override
  {
    setGlobalSeed();
      uint64_t gridSize = Execution::CalculateGridSize(PopSize);
      uint64_t blockSize = Execution::GetBlockSize();
      LinearScaling_<<<gridSize,blockSize,0,streams[omp_get_thread_num()] >>>(MatingPool, m_alpha);
  }

};
}