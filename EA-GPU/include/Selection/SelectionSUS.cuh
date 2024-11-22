#pragma once

#include"CEA.cuh"

namespace cea
{
template<uint64_t PopSize, uint64_t ChromosomeSize>
__global__ void SelectionSUS_(PopulationType<PopSize, ChromosomeSize>* Population, uint64_t* Selected)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Calculate the total fitness
        double sumFitnessValue = 0.0;
        for (uint64_t i = 0; i < PopSize; i++) {
            sumFitnessValue += Population->fitnessValue[i];
        }

        // Calculate the angle
        double angle = sumFitnessValue / static_cast<double>(PopSize);

        // Initialize cuRAND state
        curandState state;
        curand_init(seed, 0, 0, &state);

        // Generate a random number between 0 and angle
        double rand = curand_uniform_double(&state) * angle;

        // Perform SUS selection
        double cumulativeSum = 0.0;
        uint64_t k = 0;
        for (uint64_t i = 0; i < PopSize; i++) {
            cumulativeSum += Population->fitnessValue[i];
            while (rand < cumulativeSum && k < PopSize) {
                Selected[k] = i;
                k++;
                rand += angle;
            }
        }
    }
}

template<uint64_t PopSize, uint64_t ChromosomeSize>
class SelectionSUS : public Selection<PopSize, ChromosomeSize>
{
    public:
    SelectionSUS() { }

    void operator()(PopulationType<PopSize, ChromosomeSize>* Population, uint64_t* Selected) override
    {
        setGlobalSeed();
        // Launch the kernel with one block and one thread
        SelectionSUS_<<<1, 1>>>(Population, Selected);
        // Synchronize to ensure kernel completion
        cudaDeviceSynchronize();
    }
};
}
