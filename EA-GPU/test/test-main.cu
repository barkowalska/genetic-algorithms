#include"CEA.cuh"
#include"Selection/TournamentSelection.cuh"

//potrzebny pomocniczy wskaznik na funkcje device bo dopiero wartosc wskaznika mozemy skopiowac do hosta
__device__ double fitness(double*){
    return 0;
}

__device__ cea::fitnessFunction_ptr myFitenss = fitness;

int main()
{
    
    double (*device_fitness_ptr)(double*);
    cudaError_t err = cudaMemcpyFromSymbol(&device_fitness_ptr, myFitenss, sizeof(device_fitness_ptr));
    if (err != cudaSuccess) {
        std::cout<<"FAILURE";
        return -1;
    }
    cea::CEA<1,10,10> ea(device_fitness_ptr);
    cea::TournamentSelection<1,20> tournamentSelection_({1,1,1}, 2);
    tournamentSelection_.printData();
}