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


        // Generate a random number between 0 and angle
        double rand = HybridTaus(clock(),1,clock(),1) * angle;

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
        // Launch the kernel with one block and one thread
        SelectionSUS_<<<1, 1, 0,streams[omp_get_thread_num()]>>>(Population, Selected);
        // Synchronize to ensure kernel completion
        cudaDeviceSynchronize();
    }
};
}
