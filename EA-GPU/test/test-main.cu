#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include "CEA.cuh" // Include the appropriate header for cea::CEA and related classes
#include "eagpu.h"


// Define your fitness function
__device__ double myFitness(double* chromosome) {

    double wynik = 0;
    for(uint64_t i = 0; i < 10; i++)
    {
        wynik -= fabs(chromosome[i]);
    }
    return wynik; 
}

__device__ double(*funptr)(double*)=myFitness;

int main() {
    constexpr uint64_t ISLAND_NUM = 1;
    constexpr uint64_t POPSIZE = 10;
    constexpr uint64_t CHROMOSOME_SIZE = 10;

    // Retrieve the device pointer to the fitness function
    double (*device_fitness_ptr)(double*);


    
    cudaError_t err = cudaMemcpyFromSymbol(&device_fitness_ptr, funptr, sizeof(device_fitness_ptr));
    if (err != cudaSuccess) {
        std::cerr << "Failed to retrieve device function pointer: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    double min[CHROMOSOME_SIZE];
    double max[CHROMOSOME_SIZE];
    std::fill_n(min, CHROMOSOME_SIZE, -10);
    std::fill_n(max, CHROMOSOME_SIZE, 10);

    // Initialize the evolutionary algorithm
    cea::CEA<ISLAND_NUM, POPSIZE, CHROMOSOME_SIZE> ea(device_fitness_ptr);

    ea.setContraints(min, max);
    ea.setProbabilityMigration(0.6);


    // Set the initialization method
    ea.setInitialization(std::make_shared<cea::UniformInitialization<POPSIZE, CHROMOSOME_SIZE>>());
    // Set the selection method
    ea.setSelection(std::make_shared<cea::TournamentSelection<POPSIZE, CHROMOSOME_SIZE>>());

    // Set the crossover method
    ea.setCrossover(std::make_shared<cea::SimulatedBinaryCrossover<POPSIZE, CHROMOSOME_SIZE>>());

    // Set the mutation method
    ea.setMutation(std::make_shared<cea::BoundryMutation<POPSIZE, CHROMOSOME_SIZE>>());


    // Set the migration method
    ea.setMigration(std::make_shared<cea::RandomMigration<ISLAND_NUM, POPSIZE, CHROMOSOME_SIZE>>());

    // Set additional parameters
    ea.setEpsilon(0.05);
    ea.setNumMaxElements(10);
    ea.setMaxGen(100);

    // Run the evolutionary algorithm
        std::pair<double, std::array<double, CHROMOSOME_SIZE>> wynik =ea.run();

    return 0;
}
