#include<vector>
#include<memory>
#include<random>
#include<array>
#include <stdexcept>
// template<typename T, size_t N>
// class Crossover
// {
//     public:
//     virtual std::array<T,N> cross(std::vector<std::array<T,N>>&) = 0; // Funkcja czysto wirtualna
// };


template<typename T>
class Crossover
{
    protected:
    
    const size_t m_required_parents;//number of chromosomes required for crossover
    std::mt19937 m_generator;

    public:
    Crossover(size_t requiredParents ):
        m_required_parents(requiredParents), m_generator(std::random_device{}()){}

    //the main function of the crossover
    virtual std::vector<std::vector<T>> cross(std::vector<std::reference_wrapper<std::vector<T>>>& parents) = 0; // Funkcja czysto wirtualna
};
