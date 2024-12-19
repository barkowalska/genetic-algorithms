#pragma once
#include<cstdint>
#include<cuda_runtime.h>
#include <iostream>
#include <memory>
#include<random>
#include <omp.h>
#include<array>
#include<vector>
#include<chrono>
#include <algorithm>
#include <cub/cub.cuh>



inline __device__  unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)
{
  unsigned b = (((z << S1) ^ z) >> S2);
  return z = (((z & M) << S3) ^ b);
}

inline __device__ unsigned LCGStep(unsigned &z, unsigned A, unsigned C)
{
  return z = (A * z + C);
}


inline __device__ float HybridTaus(unsigned z1, unsigned z2, unsigned z3, unsigned z4)
{
  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
   return 2.3283064365387e-10 * (              // Periods
    TausStep(z1, 13, 19, 12, 4294967294UL) ^  // p1=2^31-1
    TausStep(z2, 2, 25, 4, 4294967288UL) ^    // p2=2^30-1
    TausStep(z3, 3, 11, 17, 4294967280UL) ^   // p3=2^28-1
    LCGStep(z4, 1664525, 1013904223UL)        // p4=2^32
   );
}



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

    __constant__ double ProbabilityMutation=0.01;

    template<uint64_t ChromosomeSize>
    __constant__ double MIN[ChromosomeSize]; 

    template<uint64_t ChromosomeSize>
    __constant__ double MAX[ChromosomeSize];

    typedef double(*fitnessFunction_ptr)(double*);
    __device__ fitnessFunction_ptr FitnessFunction;


    template<uint64_t PopSize, uint64_t ChromosomeSize>
    struct PopulationType
    {
        double chromosomes[PopSize*ChromosomeSize];
        double fitnessValue[PopSize];
        const uint64_t chromosomeSize = ChromosomeSize;
        const uint64_t popSize = PopSize;
    };

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    __global__ void evaluateFitnessFunction(PopulationType<PopSize,ChromosomeSize>* MatingPool, double* PopulationDataToMax)
    {
        uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= PopSize) return;

        MatingPool->fitnessValue[idx]=FitnessFunction(&MatingPool->chromosomes[idx*ChromosomeSize]);
        PopulationDataToMax[idx]=MatingPool->fitnessValue[idx];
    }

template<uint64_t IslandNum, uint64_t PopSize, uint64_t ChromosomeSize>  
__global__ void bestOfAll(PopulationType<PopSize,ChromosomeSize>** Population, int* islandsBest, double* dest, double* fitness)
{
    uint64_t best = 0;
    int i=0;
    int bestIdxInPop=0;
    for(uint64_t idx = 1; idx < IslandNum; idx++)
    {
        i=islandsBest[idx];
        
        if(Population[idx]->fitnessValue[i]> Population[best]->fitnessValue[i]){
            best = idx;
            bestIdxInPop=i;
        }
    }

    for(uint64_t idx = 0; idx < ChromosomeSize; idx++){
        dest[idx] = Population[best]->chromosomes[ChromosomeSize*bestIdxInPop+idx];
    }
    *fitness = Population[best]->fitnessValue[bestIdxInPop];
}


    class Execution
    {
        private:
        inline static uint64_t BlockSize=32;

        public:

        static void SetBlockSize(uint64_t blockSize){BlockSize = blockSize;}
        static uint64_t GetBlockSize() {return BlockSize;}
        static uint64_t CalculateGridSize(uint64_t threadsNum){return (threadsNum+BlockSize-1)/BlockSize;}
    };


    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class Selection
    {
        public:
        Selection(){}
        virtual void operator()(PopulationType<PopSize,ChromosomeSize>* Population, uint64_t* Selected) = 0;
    };

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class Crossover
    {
        public:
        Crossover() { }
        virtual void operator()(PopulationType<PopSize,ChromosomeSize>* Population,PopulationType<PopSize,ChromosomeSize>* MatingPool, uint64_t* Selected) = 0;
    };

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class Mutation
    {
        public:
        Mutation() { }
        virtual void operator()(PopulationType<PopSize,ChromosomeSize>* MatingPool) = 0;
    };

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class PenaltyFunction
    {
        public:
        PenaltyFunction() { }
        virtual void operator()(PopulationType<PopSize,ChromosomeSize>* MatingPool, uint64_t gen) = 0;
    };

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class Initialization
    {
        public:
        Initialization() {}
        virtual void operator()(PopulationType<PopSize,ChromosomeSize>* Population, double* PopulationDataToMa) = 0;
    };

    template<uint64_t IslandNum, uint64_t PopSize, uint64_t ChromosomeSize>
    class Migration
    {
        public:
        Migration(){ }
        virtual void operator()(PopulationType<PopSize,ChromosomeSize>** Population) = 0;
    };

    template<uint64_t PopSize, uint64_t ChromosomeSize>
    class Scaling
    {
        public:
        Scaling(){ }
        virtual void operator()(PopulationType<PopSize,ChromosomeSize>* MatingPool) = 0;
    };

    std::vector<cudaStream_t> streams;
    template<uint64_t IslandNum, uint64_t PopSize, uint64_t ChromosomeSize>
    class CEA
    {
        //TU BEST
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
        std::shared_ptr<PenaltyFunction<PopSize, ChromosomeSize>> m_penaltyFunction;
        std::shared_ptr<Scaling<PopSize, ChromosomeSize>> m_scaling;

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
        // Operator penaltyFunction
        inline void setPenaltyFunction(std::shared_ptr<PenaltyFunction<PopSize, ChromosomeSize>> penaltyFunction ) { m_penaltyFunction=penaltyFunction; }
        // Operator skalowania
        inline void setScaling(std::shared_ptr<Scaling<PopSize, ChromosomeSize>> scaling) { m_scaling = scaling; }

        inline void setMutationProbablility(double mutationProbability){ cudaMemcpyToSymbol(ProbabilityMutation,&mutationProbability,sizeof(double));}


        PopulationType<PopSize,ChromosomeSize>* Population[IslandNum];
        PopulationType<PopSize,ChromosomeSize>* MatingPool[IslandNum];
        uint64_t* Selected[IslandNum];

        CEA(fitnessFunction_ptr fitnessFunction)
        {
            // Allocate device memory for each island
            for (uint64_t i = 0; i < IslandNum; ++i)
            {
                CHECK(cudaMalloc(&Population[i],sizeof(PopulationType<PopSize,ChromosomeSize>)));
                CHECK(cudaMalloc(&MatingPool[i],sizeof(PopulationType<PopSize,ChromosomeSize>)));

                CHECK(cudaMalloc(&Selected[i], PopSize * sizeof(uint64_t)));
            } 
            CHECK(cudaMemcpyToSymbol(FitnessFunction, &fitnessFunction, sizeof(fitnessFunction_ptr)));
            streams.resize(IslandNum);
            for(int i=0; i<IslandNum; i++) 
            {
		        CHECK(cudaStreamCreate(&streams[i]));
    	    }
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
		        CHECK(cudaStreamDestroy(streams[i]));
            }
        }

        //odpowiedz w tym
        struct Result
        {
            std::array<uint64_t, IslandNum> generationNumber{0};
            std::pair<double, std::array<double, ChromosomeSize>> best;
            std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
            std::chrono::time_point<std::chrono::high_resolution_clock> endTime;
        };

        Result run()
        {
            Result result;

            double IslandsMaxs[IslandNum*ChromosomeSize];// jedno wspoldzie

            std::mt19937 m_generator(std::random_device{}());
            std::uniform_int_distribution<uint64_t> m_distribution(0, maxgen);

            std::vector<uint64_t> migrationPoints;

            if(IslandNum > 1)
            for(int i=0; i<maxgen*m_ProbabilityMigration; i++)
            {
                migrationPoints.push_back(m_distribution(m_generator));
            }
            std::sort(migrationPoints.begin(), migrationPoints.end());
            int k=0;
            int finishedCounter = 0;

            int* islandsBest = NULL;
            cudaMalloc(&islandsBest, sizeof(int)*IslandNum);

            cub::KeyValuePair<int, double>* d_bestPairs = nullptr;
            cudaMalloc(&d_bestPairs, sizeof(cub::KeyValuePair<int, double>) * IslandNum);


            auto start = std::chrono::high_resolution_clock::now();

            #pragma omp parallel firstprivate(k) num_threads(IslandNum) 
            {
                uint64_t counterMaxElements=0;
                PopulationType<PopSize,ChromosomeSize>* population_=Population[omp_get_thread_num()];
                PopulationType<PopSize,ChromosomeSize>* matingPool_=MatingPool[omp_get_thread_num()];
                uint64_t* selected_=Selected[omp_get_thread_num()];       

                double* current_max=NULL;
                CHECK(cudaMalloc(&current_max, sizeof(double)));

                double *PopulationDataToMax=NULL;
                CHECK(cudaMalloc(&PopulationDataToMax, sizeof(double)*PopSize));


                bool finished = false;
                (*m_initialization)(population_, PopulationDataToMax);

                 // Pamięć tymczasowa dla CUB
                void* d_temp_storage = NULL;
                size_t temp_storage_bytes = 0;

                //pierwszy max
                CHECK(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, PopulationDataToMax, current_max, PopSize, streams[omp_get_thread_num()]));
                cudaError_t err = cudaMalloc(&d_temp_storage, temp_storage_bytes);

                CHECK(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, PopulationDataToMax, current_max, PopSize, streams[omp_get_thread_num()]));

                //pierwszy max host 
                double old_maxHost;

                CHECK(cudaStreamSynchronize(streams[omp_get_thread_num()]));
                CHECK(cudaMemcpy(&old_maxHost, current_max, sizeof(double), cudaMemcpyDeviceToHost));
                //nowy max
                double current_maxHost;

                uint64_t gen = 0;

                for(gen=0; (gen<maxgen) && (finishedCounter < IslandNum); gen++)
                {
                    if(IslandNum > 1 && gen==migrationPoints[k])
                    {
                        #pragma omp barrier
                        k++;
                        #pragma omp single
                        {
                            (*m_migration)(Population);
                        }
                    }
                    if(!finished)
                    {
                        (*m_selection)(population_, selected_);
                        (*m_crossover)(population_, matingPool_, selected_);
                        (*m_mutation)(matingPool_);
                        (*m_scaling)(matingPool_);
                        (*m_penaltyFunction)(matingPool_, gen);
                        evaluateFitnessFunction<PopSize, ChromosomeSize><<<Execution::CalculateGridSize(PopSize), Execution::GetBlockSize(),0,streams[omp_get_thread_num()]>>>(matingPool_, PopulationDataToMax);
                        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, PopulationDataToMax, current_max, PopSize, streams[omp_get_thread_num()]);
                    
                        CHECK(cudaStreamSynchronize(streams[omp_get_thread_num()]));
                        CHECK(cudaMemcpy(&current_maxHost, current_max, sizeof(double),cudaMemcpyDeviceToHost));

                        double diff = (old_maxHost != 0.0) ? fabs((current_maxHost - old_maxHost )/ old_maxHost) : fabs(current_maxHost - old_maxHost);
                        old_maxHost=current_maxHost;
                        if ( diff < m_epsilon) 
                        {
                            counterMaxElements++;
                            if (counterMaxElements > m_numMaxElements) 
                            {
                                finished = true;
                                result.generationNumber[omp_get_thread_num()] = gen;
                                #pragma omp atomic
                                finishedCounter++;
                            }   
                        } 
                        else 
                        {
                            counterMaxElements = 0;
                        }
                        std::swap(population_, matingPool_);
                    }
                }

                cudaFree(current_max);
                cudaFree(d_temp_storage);

                if(result.generationNumber[omp_get_thread_num()] == 0)
                {
                    result.generationNumber[omp_get_thread_num()] = gen;
                }

                   // Pamięć tymczasowa dla CUB
                void* d_tempStorage = nullptr;
                size_t tempStorageBytes = 0;

                   // Obliczenie rozmiaru pamięci tymczasowej dla ArgMax
                CHECK(cub::DeviceReduce::ArgMax( d_tempStorage, tempStorageBytes, PopulationDataToMax, d_bestPairs + omp_get_thread_num(), PopSize));

                // Alokacja pamięci tymczasowej
                CHECK(cudaMalloc(&d_tempStorage, tempStorageBytes));

                // Wykonanie redukcji ArgMax
                CHECK(cub::DeviceReduce::ArgMax( d_tempStorage, tempStorageBytes, PopulationDataToMax, d_bestPairs + omp_get_thread_num(), PopSize));

                // Kopiowanie indeksu maksimum (key) z d_bestPairs do islandsBest
                CHECK(cudaMemcpy(
                        &islandsBest[omp_get_thread_num()],         // Adres w islandsBest
                        &(d_bestPairs[omp_get_thread_num()].key), // Adres key w d_bestPairs
                        sizeof(int), 
                        cudaMemcpyDeviceToDevice
                    ));

                // Zwolnienie pamięci tymczasowej i lokalnej tablicy
                CHECK(cudaFree(d_tempStorage));
                CHECK(cudaFree(PopulationDataToMax));
            }

        double* dest = NULL;
        double* fitnessValue = NULL;

        PopulationType<PopSize,ChromosomeSize>** populations;
        CHECK(cudaMalloc(&populations, sizeof(PopulationType<PopSize,ChromosomeSize>*)*IslandNum));
        cudaMemcpy(populations, Population, sizeof(PopulationType<PopSize,ChromosomeSize>*)*IslandNum, cudaMemcpyHostToDevice);

        CHECK(cudaMalloc(&dest, sizeof(double)*ChromosomeSize));
        CHECK(cudaMalloc(&fitnessValue, sizeof(double)));

        bestOfAll<IslandNum><<<1,1>>>(populations,islandsBest, dest, fitnessValue);

        cudaDeviceSynchronize();
        std::array<double, ChromosomeSize> chromosome;
        double fitness_host;
        CHECK(cudaMemcpy(chromosome.data(), dest, sizeof(double)*ChromosomeSize, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&fitness_host, fitnessValue, sizeof(double), cudaMemcpyDeviceToHost));

        std::pair<double, std::array<double, ChromosomeSize>> wynik (fitness_host, chromosome);
        PopulationType<PopSize,ChromosomeSize> h_population;
        CHECK(cudaMemcpy(&h_population, Population[0], sizeof(PopulationType<PopSize,ChromosomeSize>), cudaMemcpyDeviceToHost));

        auto stop = std::chrono::high_resolution_clock::now();
        result.best = wynik;
        result.startTime=start;
        result.endTime=stop;

        cudaFree(populations);
        cudaFree(dest);
        cudaFree(fitnessValue);
        cudaFree(islandsBest);
        cudaFree(d_bestPairs);

        return result;
        }
    };
}

