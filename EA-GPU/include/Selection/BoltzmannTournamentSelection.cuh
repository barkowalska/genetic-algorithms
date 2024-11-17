#pragma once

#include"CEA.cuh"

namespace cea
{
template<uint64_t PopSize, uint64_t ChromosomeSize>
__global__ void BoltzmannTournamentSelection_(PopulationType<PopSize, ChromosomeSize> *Population, uint64_t* Selected, uint64_t m_tournamentSize, double m_t)
{
    uint64_t idx= blockDim.x*blockIdx.x+threadIdx.x;
    if(idx>= PopSize) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    uint64_t best=static_cast<uint64_t>(curand_uniform(&state)* PopSize);
    for(uint64_t i=0; i<m_tournamentSize; i++)
    {
        uint64_t k= static_cast<uint64_t>(curand_uniform(&state)* PopSize);
        double pi = 1.0 / (1.0 + exp(Population->fitnessValue[k] - Population->fitnessValue[best]) / m_t);
        double rand=curand_uniform_double(&state);
        if(rand>pi){
            best=k;
        }

    }
    Selected[idx] = best;
    
}

template<uint64_t PopSize, uint64_t ChromosomeSize>
class BoltzmannTournamentSelection : public Selection<PopSize, ChromosomeSize>
{
    private:
    dim3 m_blockSize;
    public:
    uint64_t m_tournamentSize;
    double m_t;

    BoltzmannTournamentSelection(int64_t tournamentSize, double t) :
        m_tournamentSize(tournamentSize), m_t(t), m_blockSize(PopSize){}
    void operator()(PopulationType<PopSize,ChromosomeSize>* Population, uint64_t* Selected) override
    {
        setGlobalSeed();
        BoltzmannTournamentSelection_<<<1, this->m_blockSize>>>(Population, Selected, m_tournamentSize, m_t);

    }
};
}
