#pragma once

#include "CEA.cuh"
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

namespace cea
{
    template< uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void RandomMigration_(PopulationType<PopSize, ChromosomeSize>* Population_A, PopulationType<PopSize, ChromosomeSize>* Population_B)
    {
        unsigned int idx = threadIdx.x;
        if (idx >= PopSize ) return; 

        curandState state;
        curand_init(seed, 0, 0, &state);

        uint64_t range = (PopSize + blockDim.x - 1)/blockDim.x;
        uint64_t chosenFitnessValue=curand_uniform_double(&state)*range + idx*range;
        uint64_t chosenChromosomes=chosenFitnessValue*ChromosomeSize;


        double fitnessValue_toSwap=Population_B->fitnessValue[chosenFitnessValue];
        Population_B->fitnessValue[chosenFitnessValue]= Population_A->fitnessValue[chosenFitnessValue];
        Population_A->fitnessValue[chosenFitnessValue]=fitnessValue_toSwap;

        double chromosome_toSwap=Population_B->chromosomes[chosenChromosomes];

        for (uint64_t i = 0; i < ChromosomeSize; i++)
        {
        chromosome_toSwap=Population_B->chromosomes[chosenChromosomes+i];

        Population_B->chromosomes[chosenChromosomes+i]= Population_A->chromosomes[chosenFitnessValue+i];
        Population_A->chromosomes[chosenChromosomes+i]=chromosome_toSwap;

        }
    }

    template<uint64_t IslandNum, uint64_t PopSize, uint64_t ChromosomeSize>
    class RandomMigration : public Migration<IslandNum, PopSize, ChromosomeSize>
    {

        std::mt19937 m_generator;
        std::uniform_int_distribution<uint64_t> m_distribution;
    public:
    RandomMigration() : m_generator(std::random_device{}()), m_distribution(0,PopSize){};
    void operator()(PopulationType<PopSize,ChromosomeSize>** Population) override
    {
        setGlobalSeed();
        std::vector<uint64_t> populations(IslandNum);
        std::iota(populations.begin(), populations.end(),0);

        std::shuffle(populations.begin(), populations.end(),m_generator);
        uint64_t toMigrate=1;
        for(uint64_t i =0; i+1<IslandNum; ++i)
        {
            toMigrate=m_distribution(m_generator);
            RandomMigration_<<<1,toMigrate>>>(Population[populations[i]], Population[populations[i++]]);
        }
    }

    };
}