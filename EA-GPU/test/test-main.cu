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
        wynik -= fabs(chromosome[i]-10);
    }
    return wynik; 
}


        __device__ double Sphere (double* chromosome)
        {
            double Sum = 0.0;
            for (int i=0; i<30; i++)
            {
                Sum += (chromosome[i]-10.0) * (chromosome[i]-10.0);
            }

            return -1.0*Sum;
        }


__device__ double(*funptr)(double*)=Sphere;

int main() {
    constexpr uint64_t ISLAND_NUM = 1;
    constexpr uint64_t POPSIZE = 10000;
    constexpr uint64_t CHROMOSOME_SIZE = 30;

    // Retrieve the device pointer to the fitness function
    double (*device_fitness_ptr)(double*);


    
    cudaError_t err = cudaMemcpyFromSymbol(&device_fitness_ptr, funptr, sizeof(device_fitness_ptr));
    if (err != cudaSuccess) {
        std::cerr << "Failed to retrieve device function pointer: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    double min[CHROMOSOME_SIZE];
    double max[CHROMOSOME_SIZE];
    std::fill_n(min, CHROMOSOME_SIZE, -100);
    std::fill_n(max, CHROMOSOME_SIZE, 100);

    // Initialize the evolutionary algorithm
    cea::CEA<ISLAND_NUM, POPSIZE, CHROMOSOME_SIZE> ea(device_fitness_ptr);

    ea.setContraints(min, max);
    ea.setProbabilityMigration(0.6);
    ea.setMutationProbablility(0.01);


    // Set the initialization method
    ea.setInitialization(std::make_shared<cea::UniformInitialization<POPSIZE, CHROMOSOME_SIZE>>());
    // Set the selection method
    ea.setSelection(std::make_shared<cea::TournamentSelection<POPSIZE, CHROMOSOME_SIZE>>());

    // Set the crossover method
    ea.setCrossover(std::make_shared<cea::ArithmeticCrossover<POPSIZE, CHROMOSOME_SIZE>>());

    // Set the mutation method
    ea.setMutation(std::make_shared<cea::BoundryMutation<POPSIZE, CHROMOSOME_SIZE>>());
    ea.setPenaltyFunction(std::make_shared<cea::DynamicPenaltyFunction<POPSIZE,CHROMOSOME_SIZE>>());
    ea.setScaling(std::make_shared<cea::BoltzmannScaling<POPSIZE,CHROMOSOME_SIZE>>());

    // Set the migration method
    ea.setMigration(std::make_shared<cea::RandomMigration<ISLAND_NUM, POPSIZE, CHROMOSOME_SIZE>>());

    // Set additional parameters
    ea.setEpsilon(1e-6);
    ea.setNumMaxElements(10);
    ea.setMaxGen(1000);

    // Run the evolutionary algorithm
    auto  wynik =ea.run();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(wynik.endTime-wynik.startTime).count() << std::endl;
    std::cout << "Generation number - island 0: " << wynik.generationNumber[0] << std::endl;
    std::cout << "Best value: " << wynik.best.first << std::endl;
    std::cout << "Best chromosome: ";
    for(int i = 0; i < CHROMOSOME_SIZE; i++){
        std::cout << wynik.best.second[i] << ' ';
    }
    std::cout<<std::endl;

    return 0;
}
