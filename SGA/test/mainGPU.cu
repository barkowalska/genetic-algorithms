#include "SGA.cuh"

#include <iostream>
#include "cuda_runtime.h"
#include <cmath>
#include <algorithm>
#include <ctime>
#include <cub/cub.cuh>
#include <curand_kernel.h>




int main()
{
    int blockSize = 512;
    int h_popsize = 100;
    int h_l_genes = 100;
    float h_pc = 0.9f;
    float h_pm = 0.02f;
    int h_maxgen = 100;
    int h_min = -10;
    int h_max = 10;

    std::cout << "Setting constant memory" << std::endl;
    setConstantMemory(h_popsize, h_l_genes, h_pc, h_pm, h_maxgen, h_min, h_max);
    
    dim3 block(blockSize, 1); 
    dim3 grid((h_popsize + block.x - 1) / block.x, 1);

    std::cout << "Setting fitness function" << std::endl;
    setFitnessFunction<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());
    
    std::cout << "Allocating memory" << std::endl;
    Population temp_population;
    CHECK(cudaMalloc((void**)&temp_population.location, sizeof(unsigned char) * h_popsize * h_l_genes));
    CHECK(cudaMalloc((void**)&temp_population.phenotype, sizeof(double) * h_popsize));
    CHECK(cudaMalloc((void**)&temp_population.pi, sizeof(double) * h_popsize));
    CHECK(cudaMalloc((void**)&temp_population.quality, sizeof(double) * h_popsize));

    Population* d_population = nullptr;
    CHECK(cudaMalloc(&d_population, sizeof(Population)));
    CHECK(cudaMemcpy(d_population, &temp_population, sizeof(Population), cudaMemcpyHostToDevice));

    Population temp_mating_pool;
    CHECK(cudaMalloc((void**)&temp_mating_pool.location, sizeof(unsigned char) * h_popsize * h_l_genes));
    CHECK(cudaMalloc((void**)&temp_mating_pool.phenotype, sizeof(double) * h_popsize));
    CHECK(cudaMalloc((void**)&temp_mating_pool.pi, sizeof(double) * h_popsize));
    CHECK(cudaMalloc((void**)&temp_mating_pool.quality, sizeof(double) * h_popsize));

    Population* d_mating_pool = nullptr;
    CHECK(cudaMalloc(&d_mating_pool, sizeof(Population)));
    CHECK(cudaMemcpy(d_mating_pool, &temp_mating_pool, sizeof(Population), cudaMemcpyHostToDevice));

    std::cout << "Initializing" << std::endl;
    initialize<<<grid, block>>>(d_population);
    CHECK(cudaDeviceSynchronize());
    std::cout<< "Initialized";
    std::pair<double,double>  best=run(block, grid, d_population, d_mating_pool, h_popsize, h_maxgen);
    std::cout<<"best quality: "<<best.second<<std::endl;
    std::cout<<"best x: "<<best.first<<std::endl;

    CHECK(cudaFree(temp_population.location));
    CHECK(cudaFree(temp_population.phenotype));
    CHECK(cudaFree(temp_population.pi));
    CHECK(cudaFree(temp_population.quality));
    CHECK(cudaFree(d_population));

    CHECK(cudaFree(temp_mating_pool.location));
    CHECK(cudaFree(temp_mating_pool.phenotype));
    CHECK(cudaFree(temp_mating_pool.pi));
    CHECK(cudaFree(temp_mating_pool.quality));
    CHECK(cudaFree(d_mating_pool));

    return 0;
}


