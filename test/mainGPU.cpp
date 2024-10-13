#include "SGA.cuh"



int main()
{
    int blockSize=512;
    int h_popsize = 100;
    int h_l_genes = 100;
    float h_pc = 0.9f;
    float h_pm = 0.02f;
    int h_maxgen = 100;
    int h_min = -10;
    int h_max = 10;


    cudaMemcpyToSymbol(d_popsize, &h_popsize, sizeof(int));
    cudaMemcpyToSymbol(d_l_genes, &h_l_genes, sizeof(int));
    cudaMemcpyToSymbol(d_pc, &h_pc, sizeof(float));
    cudaMemcpyToSymbol(d_pm, &h_pm, sizeof(float));
    cudaMemcpyToSymbol(d_maxgen, &h_maxgen, sizeof(int));
    cudaMemcpyToSymbol(d_min, &h_min, sizeof(int));
    cudaMemcpyToSymbol(d_max, &h_max, sizeof(int));

    dim3 block (blockSize,1); 
    dim3 grid ((h_popsize+block.x-1)/block.x,1);

    //zaalokowac pamic na gpu na population
    Population *d_population= nullptr;
    cudaMalloc(&d_population, sizeof(Population));
    
    Population h_population;
    cudaMalloc((void**)&h_population.location, sizeof(char)*h_popsize*h_l_genes);
    cudaMalloc((void**)&h_population.phenotype, sizeof(double)*h_popsize);
    cudaMalloc(&h_population.pi, sizeof(double)*h_popsize);
    cudaMalloc(&h_population.quality, sizeof(double)*h_popsize);

    cudaMemcpy(d_population, &h_population, sizeof(Population) , cudaMemcpyHostToDevice);

  //zaalokowac pamic na gpu na d_mating_pool
    Population *d_mating_pool= nullptr;
    cudaMalloc(&d_mating_pool, sizeof(Population));
    Population h_mating_pool;
    cudaMalloc(&h_mating_pool.location, sizeof(char)*h_popsize*h_l_genes);
    cudaMalloc(&h_mating_pool.phenotype, sizeof(double)*h_popsize);
    cudaMalloc(&h_mating_pool.pi, sizeof(double)*h_popsize);
    cudaMalloc(&h_mating_pool.quality, sizeof(double)*h_popsize);

    cudaMemcpy(d_mating_pool, &h_mating_pool, sizeof(Population) , cudaMemcpyHostToDevice);

   
    initialize<<<grid, block>>>(d_population); //teoretycznie 
    cudaDeviceSynchronize();
    run(block, grid,d_population, d_mating_pool,h_popsize );
}