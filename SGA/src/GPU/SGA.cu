#include "SGA.cuh"
#include<iostream>


__constant__  int d_popsize;
__constant__  int d_l_genes;
__constant__  float d_pc;
__constant__   float d_pm;
__constant__   int d_maxgen;
__constant__   int d_min;
__constant__   int d_max;


__device__ unsigned long long seed;


int findMax(Population* d_population, int h_popsize)
{
    Population h_population;

    cudaMemcpy(&h_population, d_population, sizeof(Population), cudaMemcpyDeviceToHost);

    double* d_quality = h_population.quality;

    cub::KeyValuePair<int, double>* idx_val = nullptr;
    cudaMalloc(&idx_val, sizeof(cub::KeyValuePair<int, double>));

    void* d_tempStorage = nullptr;
    size_t tempStorageBytes = 0;

    cub::DeviceReduce::ArgMax(d_tempStorage, tempStorageBytes, d_quality, idx_val, h_popsize);
    cudaMalloc(&d_tempStorage, tempStorageBytes);
    cub::DeviceReduce::ArgMax(d_tempStorage, tempStorageBytes, d_quality, idx_val, h_popsize);

    cub::KeyValuePair<int, double> best_value;
    cudaMemcpy(&best_value, idx_val, sizeof(cub::KeyValuePair<int, double>), cudaMemcpyDeviceToHost);

    double idx=best_value.key;
    cudaFree(idx_val);
    cudaFree(d_tempStorage);
    return idx;
    
}


void sum(Population *d_population, double** d_totalSum, int h_popsize)
{
    Population h_population;
    cudaMalloc(d_totalSum, sizeof(double));
    cudaMemcpy(&h_population, d_population, sizeof(Population), cudaMemcpyDeviceToHost);

    double* d_quality = h_population.quality;

    void* d_tempStorage = nullptr;
    size_t tempStorageBytes = 0;

    cub::DeviceReduce::Sum(d_tempStorage, tempStorageBytes, d_quality, *d_totalSum, h_popsize);   
    cudaMalloc(&d_tempStorage, tempStorageBytes);
    cub::DeviceReduce::Sum(d_tempStorage, tempStorageBytes, d_quality, *d_totalSum, h_popsize);
    cudaFree(d_tempStorage);
}


std::pair<double,double> run(dim3 block, dim3 grid,Population *d_population, Population* d_mating_pool, int h_popsize, int h_maxgen)
{
    std::cout<<"RUN...\n";
    double* d_total_sum=nullptr;

    for(int i=0; i<h_maxgen; i++)
    {
        unsigned long long h_seed=static_cast<unsigned long long>(time(NULL)); 
        cudaMemcpyToSymbol(seed, &h_seed, sizeof(unsigned long long));
        //std::cout<< "Iteration: " << i << std::endl;
        construct_individuals<<<grid, block>>>(d_population);

        

        //std::cout <<"Summation...\n";
        cudaDeviceSynchronize();
        sum(d_population,&d_total_sum, h_popsize);


        cudaDeviceSynchronize();

        dim3 half_block(block.x / 2, block.y, block.z);

        if (block.x % 2 != 0) {
        
            half_block.x = (block.x + 1) / 2; 
        }
        //std::cout<<"Selection...\n";
        selectionRWS<<<grid, block>>>(d_population, d_total_sum, d_mating_pool );
        cross_over<<<grid, half_block>>>(d_mating_pool);
        mutation<<<grid, block>>>(d_mating_pool);
        attribution<<<grid, block>>>(d_population, d_mating_pool);
        cudaDeviceSynchronize();


   }
    construct_individuals<<<grid,block>>> (d_population);
    cudaDeviceSynchronize();

    int idx=findMax(d_population, h_popsize);


    Population h_population;
    cudaMemcpy(&h_population, d_population, sizeof(Population), cudaMemcpyDeviceToHost);

    double h_phenotype = 0.0;
    double h_quality = 0.0;

    cudaMemcpy(&h_phenotype, h_population.phenotype + idx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_quality, h_population.quality + idx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_total_sum);

    return std::make_pair(h_phenotype, h_quality);


}
__device__ double func_kwd(double x)
{
    return -x*x;
}

__global__ void setFitnessFunction()
{
    fitnessFunction = func_kwd;
}


__global__ void initialize(Population* population)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= d_popsize) return; 

    curandState state;
    curand_init(seed, i, 0, &state);

    for (int k = 0; k < d_l_genes; k++)
    {
        int idx = i * d_l_genes + k; 
        float randNum = curand_uniform(&state);
        population->location[idx] = randNum > 0.5f ? 1 : 0;
    }

}


__global__  void  construct_individuals(Population* population)
{
    unsigned int i= blockDim.x*blockIdx.x+threadIdx.x;
    if (i >= d_popsize) return; 

    //phenotype
    double sum= 0;
    for(int k=0; k<d_l_genes; k++)
    {
        sum+=population->location[i*d_l_genes+k]*(pow(2,k));
    }
    population->phenotype[i]=sum/pow(2.0,d_l_genes)*(d_max-d_min)+d_min;


    //quality 
    double x= population->phenotype[i];
    population->quality[i]= fitnessFunction(x);


}



__global__ void attribution(Population* population, Population* mating_pool)
{
    unsigned int idx= blockDim.x*blockIdx.x+threadIdx.x;
    if (idx >= d_popsize) return;

    population->phenotype[idx]=mating_pool->phenotype[idx];
    population->pi[idx]=mating_pool->pi[idx];
    population->quality[idx]=mating_pool->quality[idx];
    for(int k=0; k<d_l_genes; k++)
    {
        population->location[idx*d_l_genes+k]*=mating_pool->location[idx*d_l_genes+k];
    }


}

__device__ void calculate_pi(Population *population, double* d_total_sum)
{
    double suma=*d_total_sum;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_popsize) return; 
    population->pi[idx]= population->quality[idx]/(suma);
}

__global__ void selectionRWS(Population* population,double* sum, Population* mating_pool)
{
    calculate_pi(population, sum);

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_popsize) return; 

    double suma=0.0;
    curandState state;
    curand_init(seed, idx, 0, &state);
    double rand=curand_uniform(&state)*(2*M_PI);
    for(int i=0; i<d_popsize; i++)
    {
        suma+=2*M_PI*population->pi[i];
        if(rand<suma)
        {
            for(int k=0; k<d_l_genes; k++)
            {
                mating_pool->location[idx*d_l_genes+k]=population->location[idx*d_l_genes+k];
            }
            mating_pool->phenotype[idx]=population->phenotype[i];
            mating_pool->pi[idx]=population->pi[i];
            mating_pool->quality[idx]=population->quality[i];
            break;
        }
        
    }
    
}

__global__ void cross_over(Population* d_mating_pool)
{
    unsigned int idx=blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int i=idx*2;
    if (idx >= d_popsize || i>= d_popsize) return; 

    //unsigned long long seed = clock64() + idx;
    curandState state;
    curand_init(seed, idx, 0, &state);
    float rand=(curand_uniform(&state));

    if(rand>d_pc) return;
    unsigned char tmp;

    int point = curand(&state) % d_l_genes;

    for(; point<d_l_genes; point++)
    {
        tmp=d_mating_pool->location[idx*d_l_genes+point];
        d_mating_pool->location[idx*d_l_genes+point]=d_mating_pool->location[i*d_l_genes+point];
        d_mating_pool->location[i*d_l_genes+point]=tmp;
    }

}
__global__ void mutation(Population* d_mating_pool)
{
    unsigned int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx >= d_popsize) return; 

    //unsigned long long seed = clock64() + idx;
    curandState state;
    curand_init(seed, idx, 0, &state);

    for( int k=0; k<d_l_genes; k++)
    {
        float rand_pm=curand_uniform(&state);
        if(rand_pm<d_pm)
        {
            d_mating_pool->location[idx * d_l_genes + k] = 1 - d_mating_pool->location[idx * d_l_genes + k];
        }
    }

}

void setConstantMemory(int popsize, int l_genes, float pc, float pm, 
int maxgen, int min, int max)
{
   /* CHECK(cudaMemcpyToSymbol(d_popsize, &popsize, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(d_l_genes, &l_genes, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(d_pc, &pc, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_pm, &pm, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_maxgen, &maxgen, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(d_min, &min, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(d_max, &max, sizeof(int)));*/


    // Copy values from host to device constants
    CHECK(cudaMemcpyToSymbol(d_popsize, &popsize, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(d_l_genes, &l_genes, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(d_pc, &pc, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_pm, &pm, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(d_maxgen, &maxgen, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(d_min, &min, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(d_max, &max, sizeof(int)));

    // Synchronize to ensure all memory operations are completed
    CHECK(cudaDeviceSynchronize());

    // Host variables to copy device constants back to
    int h_popsize_copy;
    int h_l_genes_copy;
    float h_pc_copy;
    float h_pm_copy;
    int h_maxgen_copy;
    int h_min_copy;
    int h_max_copy;

    // Copy constants from device to host
    CHECK(cudaMemcpyFromSymbol(&h_popsize_copy, d_popsize, sizeof(int)));
    CHECK(cudaMemcpyFromSymbol(&h_l_genes_copy, d_l_genes, sizeof(int)));
    CHECK(cudaMemcpyFromSymbol(&h_pc_copy, d_pc, sizeof(float)));
    CHECK(cudaMemcpyFromSymbol(&h_pm_copy, d_pm, sizeof(float)));
    CHECK(cudaMemcpyFromSymbol(&h_maxgen_copy, d_maxgen, sizeof(int)));
    CHECK(cudaMemcpyFromSymbol(&h_min_copy, d_min, sizeof(int)));
    CHECK(cudaMemcpyFromSymbol(&h_max_copy, d_max, sizeof(int)));

    // Display the values
    printf("After setConstantMemory(), the device constants are:\n");
    printf("d_popsize = %d\n", h_popsize_copy);
    printf("d_l_genes = %d\n", h_l_genes_copy);
    printf("d_pc = %f\n", h_pc_copy);
    printf("d_pm = %f\n", h_pm_copy);
    printf("d_maxgen = %d\n", h_maxgen_copy);
    printf("d_min = %d\n", h_min_copy);
    printf("d_max = %d\n", h_max_copy);
}

  