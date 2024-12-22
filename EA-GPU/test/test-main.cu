#include <iostream>
#include <memory>
#include<cstdlib>
#include<unistd.h>
#include <cuda_runtime.h>
#include "CEA.cuh" // Include the appropriate header for cea::CEA and related classes
#include "eagpu.h"
#include"BenchmarkFunction.cuh"

// Define your fitness function
__device__ double myFitness(double* chromosome) {

    double wynik = 0;
    for(uint64_t i = 0; i < 10; i++)
    {
        wynik -= fabs(chromosome[i]-10);
    }
    return wynik; 
}
__device__ double(*funptr)(double*)=Benchmark::Function::Rastrigin;
    //     SPHERE = 0,
    //     SCHWEFEL_S_2_22 = 1,
    //     SCHWEFEL_S_1_20 = 2,
    //     ROSENBROCK = 3,
    //     STEP = 4,
    //     QUARTIC_NOISE = 5,
    //     SCHWEFEL_S_2_26 = 6,
    //     RASTRIGIN = 7,
    //     ACKLEY = 8,
    //     GRIEWANK = 9

// Function to run the algorithm with specific parameters and return the results
auto runAlgorithm(double (*device_fitness_ptr)(double*),  double* min,  double* max, int crossoverCoef, double mutationCoef, int maxGenerations, double migrationProp, int tournamentSize) {
    constexpr uint64_t ISLAND_NUM = 4;
    constexpr uint64_t POPSIZE = 4096;
    constexpr uint64_t CHROMOSOME_SIZE = 30;

    cea::CEA<ISLAND_NUM, POPSIZE, CHROMOSOME_SIZE> ea(device_fitness_ptr);

    ea.setContraints(min, max);
    ea.setProbabilityMigration(migrationProp);
    ea.setMutationProbablility(0.05);
    ea.setMaxGen(maxGenerations);
    ea.setInitialization(std::make_shared<cea::UniformInitialization<POPSIZE, CHROMOSOME_SIZE>>());
    ea.setSelection(std::make_shared<cea::TournamentSelection<POPSIZE, CHROMOSOME_SIZE>>(tournamentSize));
    ea.setCrossover(std::make_shared<cea::SimulatedBinaryCrossover<POPSIZE, CHROMOSOME_SIZE>>(crossoverCoef));
    ea.setMutation(std::make_shared<cea::NonuniformMutation<POPSIZE, CHROMOSOME_SIZE>>(maxGenerations, mutationCoef));
    ea.setPenaltyFunction(std::make_shared<cea::DynamicPenaltyFunction<POPSIZE, CHROMOSOME_SIZE>>());
    ea.setScaling(std::make_shared<cea::BoltzmannScaling<POPSIZE, CHROMOSOME_SIZE>>());
    ea.setMigration(std::make_shared<cea::RandomMigration<ISLAND_NUM, POPSIZE, CHROMOSOME_SIZE>>());

    auto result = ea.run();
    return result;
}

int main() {
    // Hyperparameter vectors
    std::vector<int> crossoverCoef = {2, 3, 4, 5, 6, 7};
    std::vector<double> mutationCoef = {0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5};
    std::vector<int> maxGenerations = {20000, 30000, 40000, 50000};
    std::vector<double> migrationProp = {0.1, 0.2, 0.3};
    std::vector<int> tournamentSizes = {2, 4, 8, 12, 16};

    constexpr uint64_t CHROMOSOME_SIZE = 30;
    double (*device_fitness_ptr)(double*);

    cudaError_t err = cudaMemcpyFromSymbol(&device_fitness_ptr, funptr, sizeof(device_fitness_ptr));
    if (err != cudaSuccess) {
        std::cerr << "Failed to retrieve device function pointer: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    double min[CHROMOSOME_SIZE];
    double max[CHROMOSOME_SIZE];
    int NVariable;
    Benchmark::BenchmarkProperty(Benchmark::RASTRIGIN, NVariable, min, max);

    if (NVariable != CHROMOSOME_SIZE) {
        std::cerr << "Error: NVariable from benchmark does not match CHROMOSOME_SIZE." << std::endl;
        return -1;
    }

    double bestValue = std::numeric_limits<double>::infinity();
    std::tuple<int, double, int, double, int> bestHyperparameters;

    // Grid search over all hyperparameter combinations
    for (int crossover : crossoverCoef) {
        for (double mutation : mutationCoef) {
            for (int maxGen : maxGenerations) {
                for (double migration : migrationProp) {
                    for (int tournamentSize : tournamentSizes) {
                        auto result = runAlgorithm(device_fitness_ptr, min, max, crossover, mutation, maxGen, migration, tournamentSize);

                        double norma = std::sqrt(std::accumulate(result.best.second.begin(), result.best.second.end(), 0.0,
                            [](double acc, double x) { return acc + (x-1.0) * (x-1.0); }));

                        std::cout << "Params: crossover=" << crossover
                                  << ", mutation=" << mutation
                                  << ", maxGen=" << maxGen
                                  << ", migration=" << migration
                                  << ", tournamentSize=" << tournamentSize
                                  << ", Best value: " << result.best.first
                                  << ", Norma: " << norma << std::endl;

                        if (result.best.first < bestValue) {
                            bestValue = result.best.first;
                            bestHyperparameters = std::make_tuple(crossover, mutation, maxGen, migration, tournamentSize);
                        }
                    }
                }
            }
        }
    }

    // Output the best hyperparameters
    std::cout << "Best Hyperparameters: "
              << " crossover=" << std::get<0>(bestHyperparameters)
              << " mutation=" << std::get<1>(bestHyperparameters)
              << " maxGen=" << std::get<2>(bestHyperparameters)
              << " migration=" << std::get<3>(bestHyperparameters)
              << " tournamentSize=" << std::get<4>(bestHyperparameters)
              << " Best value=" << bestValue << std::endl;

    return 0;
}



/*
int main() {
    std::vector<int> crossoverCoef = {2,3,4,5,6,7};
    std::vector<double> mutationCoef = {0.5,1.0,1.5,2.0,3.0,3.5,4.0,4.5};
    std::vector<int> maxGenerations = {20000,30000,40000,50000};
    std::vector<double> migrationProp = {0.1,0.2,0.3};


    // for(int i = 0; i < 10; i++){
    
    sleep(5);
    constexpr uint64_t ISLAND_NUM = 4;
    constexpr uint64_t POPSIZE = 4096;
    constexpr uint64_t CHROMOSOME_SIZE = 30;

    if (CHROMOSOME_SIZE <= 0 || POPSIZE <= 0 || ISLAND_NUM <= 0) 
    {
        std::cerr << "Error: CHROMOSOME_SIZE, POPSIZE, and ISLAND_NUM must be greater than zero." << std::endl;
        return -1;
    }

    // Retrieve the device pointer to the fitness function
    double (*device_fitness_ptr)(double*);


    
    cudaError_t err = cudaMemcpyFromSymbol(&device_fitness_ptr, funptr, sizeof(device_fitness_ptr));
    if (err != cudaSuccess) {
        std::cerr << "Failed to retrieve device function pointer: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    double min[CHROMOSOME_SIZE];
    double max[CHROMOSOME_SIZE];

    int NVariable;
    Benchmark::BenchmarkProperty(Benchmark::RASTRIGIN, NVariable, min, max);

    if (NVariable != CHROMOSOME_SIZE) 
    {
        std::cerr << "Error: NVariable from benchmark does not match CHROMOSOME_SIZE." << std::endl;
        return -1;
    }


    // Initialize the evolutionary algorithm
    cea::CEA<ISLAND_NUM, POPSIZE, CHROMOSOME_SIZE> ea(device_fitness_ptr);
    int maxGen = 30000;

    ea.setContraints(min, max);
    ea.setProbabilityMigration(0.2);
    ea.setMutationProbablility(0.05);
    ea.setEpsilon(1e-7);
    ea.setNumMaxElements(4000);
    ea.setMaxGen(maxGen);



    // Set the initialization method
    ea.setInitialization(std::make_shared<cea::UniformInitialization<POPSIZE, CHROMOSOME_SIZE>>());
    // Set the selection method
    ea.setSelection(std::make_shared<cea::TournamentSelection<POPSIZE, CHROMOSOME_SIZE>>(8));

    // Set the crossover method
    ea.setCrossover(std::make_shared<cea::SimulatedBinaryCrossover<POPSIZE, CHROMOSOME_SIZE>>(3));

    // Set the mutation method
    ea.setMutation(std::make_shared<cea::NonuniformMutation<POPSIZE, CHROMOSOME_SIZE>>(maxGen,2));
    ea.setPenaltyFunction(std::make_shared<cea::DynamicPenaltyFunction<POPSIZE,CHROMOSOME_SIZE>>());
    ea.setScaling(std::make_shared<cea::BoltzmannScaling<POPSIZE,CHROMOSOME_SIZE>>());

    // Set the migration method
    ea.setMigration(std::make_shared<cea::RandomMigration<ISLAND_NUM, POPSIZE, CHROMOSOME_SIZE>>());

    // Set additional parameters

    // Run the evolutionary algorithm
    auto  wynik =ea.run();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(wynik.endTime-wynik.startTime).count() << std::endl;
    for(int i = 0; i < ISLAND_NUM; i++){
        std::cout << "Generation number - island "<< i <<": " << wynik.generationNumber[i] << std::endl;
    }
    std::cout << "Best value: " << wynik.best.first << std::endl;
    //std::cout << "Best chromosome: ";
    // for(int i = 0; i < CHROMOSOME_SIZE; i++){
    //     std::cout << wynik.best.second[i] << ' ';
    // }
    double norma = std::sqrt(std::accumulate(wynik.best.second.begin(), wynik.best.second.end(), 0.0,
                  [](double acc, double x) { return acc + x * x; }));

    std::cout<<" Norma: "<< norma<< std::endl;

    return 0;
}


*/