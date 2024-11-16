#include"CEA.cuh"

__device__ double fitness(double*){
    return 0;
}

int main()
{
    
    double (*device_fitness_ptr)(double*);
    cudaError_t err = cudaMemcpyFromSymbol(&device_fitness_ptr, fitness, sizeof(device_fitness_ptr));
    if (err != cudaSuccess) {
        // Handle error
        return -1;
    }
    cea::CEA<1,10,10> ea(device_fitness_ptr);
}