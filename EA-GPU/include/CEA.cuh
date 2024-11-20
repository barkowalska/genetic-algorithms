#pragma once
#include<cstdint>
#include<cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <memory>
#include<random>
#include <omp.h>
#include<array>
#include <algorithm>

// dodac do crosss na poczatku spr pc jak w mutation
//poprawic gridsize i blocksize
//scalling dodac
//usunac devicesynchromnize
//dodac setglobalseed() wszedzie
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

inline __device__ double changeIf(double basic, double changed, double draft, double refernceValue)
{
    return (draft >= refernceValue)*basic + (draft < refernceValue)*changed;
}

template<typename T>
inline __device__ T tenary(T ifTrue, T ifFalse, bool condtional)
{
    return condtional*ifTrue + (!condtional)*ifFalse;
}


namespace cea
{

struct BestIdx
{
    uint64_t idx;
    double value;
};

__constant__ double ProbabilityMutation;

template<uint64_t ChromosomeSize>
__constant__ double MIN[ChromosomeSize]; 

template<uint64_t ChromosomeSize>
__constant__ double MAX[ChromosomeSize];

template<uint64_t IslandNum>
__device__ BestIdx CurrentBest[IslandNum]; 

template<uint64_t IslandNum>
__device__ BestIdx OldBest[IslandNum]; 

typedef double(*fitnessFunction_ptr)(double*);
__device__ fitnessFunction_ptr FitnessFunction;


template<uint64_t PopSize, uint64_t ChromosomeSize>
struct PopulationType{
        double chromosomes[PopSize*ChromosomeSize];
        double fitnessValue[PopSize];
        const uint64_t chromosomeSize = ChromosomeSize;
        const uint64_t popSize = PopSize;
};

template<uint64_t PopSize, uint64_t ChromosomeSize>
__global__ void evaluateFitnessFunction(PopulationType<PopSize,ChromosomeSize>* MatingPool)
{
    uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= PopSize) return;

    MatingPool->fitnessValue[idx]=FitnessFunction(&MatingPool->chromosomes[idx*ChromosomeSize]);
}


__device__ unsigned long long seed;
void setGlobalSeed()
{
    unsigned long long h_seed=static_cast<unsigned long long>(time(NULL)); 
    cudaMemcpyToSymbol(seed, &h_seed, sizeof(unsigned long long));
}

template<uint64_t IslandNum, uint64_t PopSize, uint64_t ChromosomeSize>  
__global__ void findBest(PopulationType<PopSize,ChromosomeSize>* Population,int IslandNume)
{
    uint64_t best = 0;
    for(uint64_t idx = 1; idx < PopSize; idx++)
    {
        if(Population->fitnessValue[idx] > Population->fitnessValue[best]){
            best = idx;
        }
    }

    OldBest<IslandNum>[IslandNume] = CurrentBest<IslandNum>[IslandNume];
    CurrentBest<IslandNum>[IslandNume].idx = best;
    CurrentBest<IslandNum>[IslandNume].value = Population->fitnessValue[best];
}

template<uint64_t IslandNum, uint64_t PopSize, uint64_t ChromosomeSize>  
__global__ void bestOfAll(PopulationType<PopSize,ChromosomeSize>** Population, double* dest, double* fitness)
{
    uint64_t best = 0;
    for(uint64_t idx = 1; idx < IslandNum; idx++)
    {
        if(CurrentBest<IslandNum>[idx].value > CurrentBest<IslandNum>[best].value){
            best = idx;
        }
    }
    uint64_t bestIndv = CurrentBest<IslandNum>[best].idx;
    for(uint64_t idx = 0; idx < ChromosomeSize; idx++){
        dest[idx] = Population[best]->chromosomes[ChromosomeSize*bestIndv+idx];
    }
    *fitness = CurrentBest<IslandNum>[best].value;
}

class Execution{
    private:
    inline static uint64_t BlockSize=32;

    public:

    static void SetBlockSize(uint64_t blockSize){BlockSize = blockSize;}
    static uint64_t GetBlockSize() {return BlockSize;}
    static uint64_t CalculateGridSize(uint64_t threadsNum){return (threadsNum+BlockSize-1)/BlockSize;}
};


template<uint64_t PopSize, uint64_t ChromosomeSize>
class Selection{
    public:
    Selection(){ }
    virtual void operator()(PopulationType<PopSize,ChromosomeSize>* Population, uint64_t* Selected) = 0;
};

template<uint64_t PopSize, uint64_t ChromosomeSize>
class Crossover{
    public:
    Crossover() { }
    virtual void operator()(PopulationType<PopSize,ChromosomeSize>* Population,PopulationType<PopSize,ChromosomeSize>* MatingPool, uint64_t* Selected) = 0;
};

template<uint64_t PopSize, uint64_t ChromosomeSize>
class Mutation{
    public:
    Mutation() { }
    virtual void operator()(PopulationType<PopSize,ChromosomeSize>* MatingPool) = 0;
};

template<uint64_t PopSize, uint64_t ChromosomeSize>
class Initialization{
    public:
    Initialization() {}
    virtual void operator()(PopulationType<PopSize,ChromosomeSize>* Population) = 0;
};

template<uint64_t IslandNum, uint64_t PopSize, uint64_t ChromosomeSize>
class Migration{
    public:
    Migration(){ }
    virtual void operator()(PopulationType<PopSize,ChromosomeSize>** Population) = 0;
};

template<uint64_t IslandNum, uint64_t PopSize, uint64_t ChromosomeSize>
class CEA
{
   
    // jakis minimalny zakresdla ktorego individuale nie beda sie juz od siebie
    double m_epsilon;
    // n elemntow dla ktporych max moze byc zbyt mala roznica
    size_t m_numMaxElements;
    uint64_t maxgen;


    double m_ProbabilityMigration;
    std::shared_ptr<Crossover<PopSize, ChromosomeSize>> m_crossover;
    std::shared_ptr<Mutation<PopSize, ChromosomeSize>> m_mutation;
    std::shared_ptr<Selection<PopSize, ChromosomeSize>> m_selection;
    std::shared_ptr<Initialization<PopSize, ChromosomeSize>> m_initialization;
    std::shared_ptr<Migration<IslandNum, PopSize, ChromosomeSize>> m_migration;
    static_assert((PopSize % 2) == 0);

    

    public:

    // Minimalny zakres, dla którego indywidua nie będą się już od siebie różnić
    inline void setEpsilon(double epsilon) { m_epsilon = epsilon; }
    // Liczba elementów, dla których maksymalna różnica może być zbyt mała
    inline void setNumMaxElements(size_t numMaxElements) { m_numMaxElements = numMaxElements; }
    // Maksymalna liczba generacji
    inline void setMaxGen(uint64_t maxgenVal) { maxgen = maxgenVal; }
    // Prawdopodobieństwo migracji
    inline void setProbabilityMigration(double probabilityMigration) { m_ProbabilityMigration = probabilityMigration; }
    // Operator krzyżowania
    inline void setCrossover(std::shared_ptr<Crossover<PopSize, ChromosomeSize>> crossover) { m_crossover = crossover; }
    // Operator mutacji
    inline void setMutation(std::shared_ptr<Mutation<PopSize, ChromosomeSize>> mutation) { m_mutation = mutation; }
    // Operator selekcji
    inline void setSelection(std::shared_ptr<Selection<PopSize, ChromosomeSize>> selection) { m_selection = selection; }
    // Operator inicjalizacji
    inline void setInitialization(std::shared_ptr<Initialization<PopSize, ChromosomeSize>> initialization) { m_initialization = initialization; }
    // Operator migracji
    inline void setMigration(std::shared_ptr<Migration<IslandNum, PopSize, ChromosomeSize>> migration) { m_migration = migration; }




    PopulationType<PopSize,ChromosomeSize>* Population[IslandNum];
    PopulationType<PopSize,ChromosomeSize>* MatingPool[IslandNum];
    uint64_t* Selected[IslandNum];

    CEA(fitnessFunction_ptr fitnessFunction){
       // Allocate device memory for each island
        for (uint64_t i = 0; i < IslandNum; ++i)
        {
            CHECK(cudaMalloc(&Population[i],sizeof(PopulationType<PopSize,ChromosomeSize>)));
            CHECK(cudaMalloc(&MatingPool[i],sizeof(PopulationType<PopSize,ChromosomeSize>)));

            CHECK(cudaMalloc(&Selected[i], PopSize * sizeof(uint64_t)));
        } 
        CHECK(cudaMemcpyToSymbol(FitnessFunction, &fitnessFunction, sizeof(fitnessFunction_ptr)));
    }

    void setContraints(double* min, double* max)
    {
        CHECK(cudaMemcpyToSymbol(MIN<ChromosomeSize>, min, ChromosomeSize*sizeof(double)));
        CHECK(cudaMemcpyToSymbol(MAX<ChromosomeSize>, max, ChromosomeSize*sizeof(double)));
    }

    // Destructor
    ~CEA()
    {
        // Free device memory
        for (uint64_t i = 0; i < IslandNum; ++i)
        {
            CHECK(cudaFree(Population[i]));
            CHECK(cudaFree(MatingPool[i]));
            CHECK(cudaFree(Selected[i]));
        }
    }



    std::pair<double, std::array<double, ChromosomeSize>> run()
    {
        std::mt19937 m_generator(std::random_device{}());
        std::uniform_int_distribution<uint64_t> m_distribution(0, maxgen);

        std::vector<uint64_t> migrationPoints;

        for(int i=0; i<maxgen*m_ProbabilityMigration; i++)
        {
            migrationPoints.push_back(m_distribution(m_generator));
        }
        std::sort(migrationPoints.begin(), migrationPoints.end());
        int k=0;
        int finishedCounter = 0;

        #pragma omp parallel firstprivate(k) num_threads(IslandNum)
        {
            uint64_t counterMaxElements=0;
            PopulationType<PopSize,ChromosomeSize>* population_=Population[omp_get_thread_num()];
            PopulationType<PopSize,ChromosomeSize>* matingPool_=MatingPool[omp_get_thread_num()];
            uint64_t* selected_=Selected[omp_get_thread_num()];       
            BestIdx current;
            BestIdx old;
            bool finished = false;
            (*m_initialization)(population_);

            for(uint64_t i=0; i<maxgen && finishedCounter < omp_get_num_threads(); i++)
            {
                if(i==migrationPoints[k])
                {
                    #pragma omp barrier
                    k++;
                    #pragma omp single
                    {
                        (*m_migration)(Population);
  
                    }

                }
                if(!finished){
                    (*m_selection)(population_, selected_);
                    (*m_crossover)(population_, matingPool_, selected_);
                    (*m_mutation)(matingPool_);

                    evaluateFitnessFunction<PopSize, ChromosomeSize><<<Execution::CalculateGridSize(PopSize), Execution::GetBlockSize()>>>(matingPool_);
                    findBest<IslandNum, PopSize,ChromosomeSize><<<1,1>>>(population_, omp_get_thread_num());
                    CHECK(cudaDeviceSynchronize());

                    CHECK(cudaMemcpyFromSymbol(&current, CurrentBest<IslandNum>, sizeof(BestIdx), sizeof(BestIdx)*omp_get_thread_num()));
                    CHECK(cudaMemcpyFromSymbol(&old, OldBest<IslandNum>, sizeof(BestIdx), sizeof(BestIdx)*omp_get_thread_num()));
                    
                    if ((old.value != 0.0 ? (fabs(current.value - old.value) / old.value) : fabs(current.value - old.value)) < m_epsilon) {
                    counterMaxElements++;
                    if (counterMaxElements > m_numMaxElements) {
                        finished = true;
                        #pragma omp atomic
                        finishedCounter++;
                    }   
                    } else {
                        counterMaxElements = 0;
                    }
                    std::swap(population_, matingPool_);
                }
            }
        }



        double* dest = NULL;
        double* fitnessValue = NULL;

        PopulationType<PopSize,ChromosomeSize>** populations;
        CHECK(cudaMalloc(&populations, sizeof(PopulationType<PopSize,ChromosomeSize>*)*IslandNum));
        cudaMemcpy(populations, Population, sizeof(PopulationType<PopSize,ChromosomeSize>*)*IslandNum, cudaMemcpyHostToDevice);

        CHECK(cudaMalloc(&dest, sizeof(double)*ChromosomeSize));
        CHECK(cudaMalloc(&fitnessValue, sizeof(double)));
        bestOfAll<IslandNum><<<1,1>>>(populations, dest, fitnessValue);
        cudaDeviceSynchronize();
        std::array<double, ChromosomeSize> chromosome;
        double fitness_host;
        CHECK(cudaMemcpy(chromosome.data(), dest, sizeof(double)*ChromosomeSize, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&fitness_host, fitnessValue, sizeof(double), cudaMemcpyDeviceToHost));

        std::pair<double, std::array<double, ChromosomeSize>> wynik (fitness_host, chromosome);
        PopulationType<PopSize,ChromosomeSize> h_population;
        CHECK(cudaMemcpy(&h_population, Population[0], sizeof(PopulationType<PopSize,ChromosomeSize>), cudaMemcpyDeviceToHost));
        //std::pair<double, std::array<double, ChromosomeSize>> wynik;
        cudaFree(populations);
        cudaFree(dest);
        cudaFree(fitnessValue);
        return wynik;
    }
};


}

