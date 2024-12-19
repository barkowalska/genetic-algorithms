#pragma once

#include"CEA.cuh"
namespace cea
{
template<uint64_t PopSize, uint64_t ChromosomeSize>
__global__ void BoltzmannScaling_(PopulationType<PopSize,ChromosomeSize>* MatingPool, double m_temperature)
{
  uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= PopSize) return;

   MatingPool->fitnessValue[idx] =exp(MatingPool->fitnessValue[idx] / m_temperature);


}

template<uint64_t PopSize, uint64_t ChromosomeSize>
class BoltzmannScaling: public Scaling<PopSize,ChromosomeSize>
{
    public:
    double m_temperature;// Temperature parameter for adjusting the scaling sensitivity

    BoltzmannScaling( double temperature = 1.0): m_temperature(temperature){}
    void operator()(PopulationType<PopSize,ChromosomeSize>* MatingPool) override
    {
      uint64_t gridSize = Execution::CalculateGridSize(PopSize);
      uint64_t blockSize = Execution::GetBlockSize();
     BoltzmannScaling_<<<gridSize,blockSize, 0,streams[omp_get_thread_num()]>>>(MatingPool, m_temperature);
    }

  };
}