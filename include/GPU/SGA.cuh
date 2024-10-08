#include "cuda_runtime.h"
#include <cmath>
#include <algorithm>
#include <ctime>

#include <curand_kernel.h>


// Makro do sprawdzania błędów CUDA
#define CHECK(call)\
{\
    const cudaError_t error=call;\
    if(error!=cudaSuccess)\
    {\
        printf("Error: %s: %d, "__FILE__, __LINE__);\
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}

__constant__ unsigned long long seed;
__constant__ int popsize;
__constant__ int l_genes;

struct Population
{
    double *quality;
    unsigned char* location;
    double* phenotype;
    double* pi;
};

typedef double(*functionPointer_t)(double);
__device__ functionPointer_t fitnessFunction;


__device__ double func_kwd(double x)
{
    return x*x;
}

__global__ void initialize(Population *population, unsigned long long seed, int min, int max);

__device__ void construct_individuals(int min, int max, Population* population);

__global__ void sum(Population* population, double* g_odata);

__global__ void sumFrommAllBlocks(double * g_odata, size_t blocksToSum);

__device__ void pi(Population *population, double* sum);

__global__ void selectionRWS(Population* population,double sum, Population* mating_pool,  unsigned long long seed);

__global__ void cross_over(Population* mating_pool, unsigned long long seed);

__global__ void mutattion(Population* mating_pool, unsigned long long seed);


__device__ void cudaRand(double* output)
{
    int i=blockDim.x*blockIdx.x+theradIdx.x;
    curandState state;
    curand_init((unsigned long long)clock() +i, 0, 0, &state);

    output[i]=curand_uniform_double(&state);
}

void run(dim3 block, dim3 grid, Population* population, Population* mating_pool)
{
    double* suma;
    cudaMalloc(&suma, sizeof(double)*block.x);

    sum<<<grid, block, sizeof(double)*block.x>>> (population, popsize, suma);
    sumFrommAllBlocks<<<1,1>>>(suma, block.x);

    selectionRWS<<<grid, block>>>(population, suma, mating_pool, )


}
int main()
{
    int blockSize=512;

    float h_pc=0.9;
    float h_pm=0.02;
    popsize=blockSize;
    int h_maxgen=100;
    int h_min=-10;
    int h_max=10;
    l_genes=60;

    unsigned long long seed=1234ULL;

    dim3 block (blockSize,1); //dlaczego to nie deklaruje sie jako = cos tam
    dim3 grid ((popsize+block.x-1)/block.x,1);

    //zaalokowac pamic na gpu na population
    Population *population= nullptr;
    cudaMalloc(&population, sizeof(Population));
    Population h_pop;
    cudaMalloc((void**)&h_pop.location, sizeof(char)*popsize*l_genes);
    cudaMalloc((void**)&h_pop.phenotype, sizeof(double)*popsize);
    cudaMalloc(&h_pop.pi, sizeof(double)*popsize);
    cudaMalloc(&h_pop.quality, sizeof(double)*popsize);

    cudaMemcpy(population, &h_pop, sizeof(Population) , cudaMemcpyHostToDevice);

  //zaalokowac pamic na gpu na mating_pool
    Population *mating_pool= nullptr;
    cudaMalloc(&mating_pool, sizeof(Population));
    Population h_mating_pool;
    cudaMalloc(&h_mating_pool.location, sizeof(char)*popsize*l_genes);
    cudaMalloc(&h_mating_pool.phenotype, sizeof(double)*popsize);
    cudaMalloc(&h_mating_pool.pi, sizeof(double)*popsize);
    cudaMalloc(&h_mating_pool.quality, sizeof(double)*popsize);

    cudaMemcpy(mating_pool, &h_mating_pool, sizeof(Population) , cudaMemcpyHostToDevice);


    initialize<<<grid, block>>>(population, h_l_genes, h_popsize, seed, h_min, h_max); //teoretycznie 
    run();
}

__global__ void initialize(Population *population, unsigned long long seed, int min, int max)
{
    unsigned int i= blockDim.x*blockIdx.x+threadIdx.x;
    //location
    for(int k=0; k<l_genes; k++)
    {
        curandState state;
        curand_init(seed, i, 0, &state);
        population->location[i*l_genes+k]=curand_uniform(&state) > 0.5f ? 1 : 0;
    }

    construct_individuals(min, max, population, l_genes, popsize);
}

__device__  void  construct_individuals(int min, int max, Population* population)
{
    unsigned int i= blockDim.x*blockIdx.x+threadIdx.x;

     //phenotype
    int *sum= new int[popsize];
    for(int k=0; k<l_genes; k++)
    {
        sum[i]+=population->location[i*l_genes+k]*(pow(2,i));
    }
    population->phenotype[i]=(static_cast<double>(sum[i])/static_cast<double>(pow(2,l_genes)))*(max-min)+min;


    //quality 
    double x= population->phenotype[i];
    population->quality[i]= fitnessFunction(x);


}

__global__ void sum(Population* population, double* g_odata)
{    // Set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert global data pointer to the local pointer of this block
    extern __shared__ double sdata[];

    // Load elements into shared memory (each thread loads one element)
    if (idx < popsize) {
        sdata[tid] = population->quality[idx];
    } else {
        sdata[tid] = 0;
    }
  
    // In-place reduction within shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write result of this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}


__global__ void sumFrommAllBlocks(double * g_odata, size_t blocksToSum)
{

    for(size_t i = 1; i < blocksToSum; i++)
    {
        g_odata[0] += g_odata[i];
    }
}

__device__ void pi(Population *population, double* sum)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    population->pi[idx]= population->quality[idx]/sum[0];
}

__global__ void selectionRWS(Population* population,double sum, Population* mating_pool,  unsigned long long seed)
{
    pi(population, &sum);

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double sum;

    curandState state;
    curand_init(seed, idx, 0, &state);
    double rand=static_cast<double>((curand_uniform(&state))%2*M_PI);
    for(int i=0; i<popsize; i++)
    {
        sum+=2*M_PI*population->pi[i];
        if(rand<sum)
        {
            mating_pool->location[idx]=population->location[i];
            mating_pool->phenotype[idx]=population->phenotype[i];
            mating_pool->pi[idx]=population->pi[i];
            mating_pool->quality[idx]=population->quality[i];
            break;
        }
        
    }
    
}

//CO DWA WATKI
__global__ void cross_over(Population* mating_pool, unsigned long long seed)
{
    unsigned int idx=blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int i=idx++;

    curandState state;
    curand_init(seed, idx, 0, &state);
    int rand=static_cast<int>((curand_uniform(&state))%l_genes);

    unsigned char tmp;

    for(; rand<l_genes; rand++)
    {
        tmp=mating_pool->location[idx*l_genes+rand];
        mating_pool->location[idx*l_genes+rand]=mating_pool->location[i*l_genes+rand];
        mating_pool->location[i*l_genes+rand]=tmp;
    }

}
//CZY TUTAJ NA PEWNO DOBRZE Z TYM RAND
__global__ void mutattion(Population* mating_pool, unsigned long long seed)
{
    unsigned int idx=blockDim.x*blockIdx.x+threadIdx.x;

    curandState state;
    curand_init(seed, idx, 0, &state);
    int rand=static_cast<int>((curand_uniform(&state))%l_genes);

    for( ; rand<l_genes; rand++)
    {
        if(mating_pool->location[idx*l_genes+rand]==0) mating_pool->location[idx*l_genes+rand]=1;
        else
            mating_pool->location[idx*l_genes+rand]=0;
    }

}