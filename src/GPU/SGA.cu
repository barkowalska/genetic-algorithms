#include "SGA.cuh"


/*

__device__ void cudaRand(double* output)
{
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=d_popsize) return;
    curandState state;
    curand_init((unsigned long long)clock() +i, 0, 0, &state);

    output[i]=curand_uniform_double(&state);
}
*/


void sum(double* wskqualitySum, double** d_totalSum, int h_popsize)
{
    cudaMalloc(d_totalSum, sizeof(double));

    void* d_tempStorage = nullptr;
    size_t tempStorageBytes = 0;

    cub::DeviceReduce::Sum(d_tempStorage, tempStorageBytes, wskqualitySum, *d_totalSum, h_popsize);

    cudaMalloc(&d_tempStorage, tempStorageBytes);

    cub::DeviceReduce::Sum(d_tempStorage, tempStorageBytes, wskqualitySum, *d_totalSum, h_popsize);

    cudaFree(d_tempStorage);
}

void run(dim3 block, dim3 grid,Population *d_population, Population* d_mating_pool, int h_popsize)
{
    unsigned long long h_seed=static_cast<unsigned long long>(time(NULL)); 
    cudaMemcpyToSymbol(seed, &h_seed, sizeof(unsigned long long));

    double* d_total_sum=nullptr;
    sum(d_population->quality,&d_total_sum, h_popsize);

    cudaDeviceSynchronize();

    dim3 half_block(block.x / 2, block.y, block.z);

    if (block.x % 2 != 0) {
        
        half_block.x = (block.x + 1) / 2; 
    }

    selectionRWS<<<grid, block>>>(d_population, d_total_sum, d_mating_pool );
    cross_over<<<grid, half_block>>>(d_mating_pool);
    mutattion<<<grid, block>>>(d_mating_pool);

    cudaDeviceSynchronize();

    cudaFree(d_total_sum);


}
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
    run(block, grid,d_population, d_mating_pool,  h_popsize );
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

    construct_individuals(population);
}


__device__  void  construct_individuals(Population* population)
{
    unsigned int i= blockDim.x*blockIdx.x+threadIdx.x;
        if (i >= d_popsize) return; // Ensure we don't go out of bounds

     //phenotype
    int *sum= new int[d_popsize];
    for(int k=0; k<d_l_genes; k++)
    {
        sum[i]+=population->location[i*d_l_genes+k]*(pow(2,i));
    }
    population->phenotype[i]=(static_cast<double>(sum[i])/static_cast<double>(pow(2,d_l_genes)))*(d_max-d_min)+d_min;


    //quality 
    double x= population->phenotype[i];
    population->quality[i]= fitnessFunction(x);


}




__device__ void pi(Population *population, double* d_total_sum)
{
    double suma=*d_total_sum;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_popsize) return; 
    population->pi[idx]= population->quality[idx]/(suma);
}

__global__ void selectionRWS(Population* population,double* sum, Population* d_mating_pool)
{
    pi(population, sum);

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_popsize) return; 

    double suma;
    curandState state;
    curand_init(seed, idx, 0, &state);
    double rand=curand_uniform(&state)*(2*M_PI);
    for(int i=0; i<d_popsize; i++)
    {
        suma+=2*M_PI*population->pi[i];
        if(rand<suma)
        {
            d_mating_pool->location[idx]=population->location[i];
            d_mating_pool->phenotype[idx]=population->phenotype[i];
            d_mating_pool->pi[idx]=population->pi[i];
            d_mating_pool->quality[idx]=population->quality[i];
            break;
        }
        
    }
    
}

__global__ void cross_over(Population* d_mating_pool, float h_pc)
{
    unsigned int idx=blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int i=idx*2;
    if (idx >= d_popsize || i>= d_popsize) return; 

    //unsigned long long seed = clock64() + idx;
    curandState state;
    curand_init(seed, idx, 0, &state);
    float rand=(curand_uniform(&state));

    if(rand>h_pc) return;
    unsigned char tmp;

    int point = curand(&state) % d_l_genes;

    for(; point<d_l_genes; point++)
    {
        tmp=d_mating_pool->location[idx*d_l_genes+point];
        d_mating_pool->location[idx*d_l_genes+point]=d_mating_pool->location[i*d_l_genes+point];
        d_mating_pool->location[i*d_l_genes+point]=tmp;
    }

}
__global__ void mutattion(Population* d_mating_pool,float h_pm)
{
    unsigned int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx >= d_popsize) return; 

    //unsigned long long seed = clock64() + idx;
    curandState state;
    curand_init(seed, idx, 0, &state);

    for( int k=0; k<d_l_genes; k++)
    {
        float rand_pm=curand_uniform(&state);
        if(rand_pm<h_pm)
        {
            d_mating_pool->location[idx * d_l_genes + k] = 1 - d_mating_pool->location[idx * d_l_genes + k];
        }
    }

}

