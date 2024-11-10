#include "../../include/Crossover/UniformCrossover.h"


std::vector<std::vector<double>> UniformCrossover::cross(std::vector<std::reference_wrapper<std::vector<double>>>& parents)
{
    if(parents.size() != m_requiredParents) 
        throw std::invalid_argument("invalid number of parents; expected 2");

    if (parents[0].get().size() != parents[1].get().size()) 
        throw std::invalid_argument("parents must be the same size");
        
    size_t size = parents[0].get().size();

    std::vector<double> child1(size);
    std::vector<double> child2(size);

    std::uniform_real_distribution<> distribution(0.0, 1.0);

    for (size_t i = 0; i < size; ++i) {
        double randomValue = distribution(m_generator);
            
        if (randomValue < 0.5) {
            child1[i] = parents[0].get()[i];
            child2[i] = parents[1].get()[i];
        } else {
            child1[i] = parents[1].get()[i];
            child2[i] = parents[0].get()[i];
        }
    }
        return {child1, child2};
}


