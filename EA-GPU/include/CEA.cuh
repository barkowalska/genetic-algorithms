#include<cstdint>
#include<cuda_runtime.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        std::cout << "Error in file: " << __FILE__ << " at line: " << __LINE__ << std::endl;\
        std::cout << "CUDA Error " << error << ": " << cudaGetErrorString(error) << std::endl;\
        exit(1);\
    }\
}

namespace cea
{

typedef double(*fitnessFunction_ptr)(double*);
__device__ fitnessFunction_ptr FitnessFunction;

template<uint64_t IslandNum, uint64_t PopSize, uint64_t ChromosomeSize>
class CEA
{
    public:
    struct PopulationType{
        double chromosomes[PopSize*ChromosomeSize];
        double fitnessValue[PopSize];
        const uint64_t chromosomeSize = ChromosomeSize;
        const uint64_t popSize = PopSize;
    };

    PopulationType* Population[IslandNum];
    PopulationType* MatingPool[IslandNum];
    uint64_t* Selected[IslandNum];

    CEA(fitnessFunction_ptr fitnessFunction){
       // Allocate device memory for each island
        for (uint64_t i = 0; i < IslandNum; ++i)
        {
            cudaMalloc(&Population[i],sizeof(PopulationType));
            cudaMalloc(&MatingPool[i],sizeof(PopulationType));

            cudaMalloc(&Selected[i], PopSize * sizeof(uint64_t));
        } 
        cudaMemcpyToSymbol(FitnessFunction, &fitnessFunction, sizeof(fitnessFunction_ptr));
    }

    // Destructor
    ~CEA()
    {
        // Free device memory
        for (uint64_t i = 0; i < IslandNum; ++i)
        {
            cudaFree(Population[i]);
            cudaFree(MatingPool[i]);
            cudaFree(Selected[i]);
        }
    }
};


}