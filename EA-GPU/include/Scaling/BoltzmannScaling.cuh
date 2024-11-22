#pragma once

#include"CEA.cuh"
namespace cea
{
template<uint64_t PopSize, uint64_t ChromosomeSize>
__global__ void BoltzmannScaling_(PopulationType<PopSize,ChromosomeSize>* Population, double m_temperature)
{
  uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= PopSize) return;

   Population->fitnessValue[idx] =exp(Population->fitnessValue[idx] / m_temperature);


}

template<uint64_t PopSize, uint64_t ChromosomeSize>
class BoltzmannScaling: public Scaling<PopSize,ChromosomeSize>
{
  public:
  double m_temperature;// Temperature parameter for adjusting the scaling sensitivity

  BoltzmannScaling( double temperature = 1.0): m_temperature(temperature){}
  void operator()(PopulationType<PopSize,ChromosomeSize>* Population) override
  {
    setGlobalSeed();
      uint64_t gridSize = Execution::CalculateGridSize(PopSize);
      uint64_t blockSize = Execution::GetBlockSize();
      BoltzmannScaling_<<<gridSize,blockSize>>>(Population, m_temperature);
  }

};
}