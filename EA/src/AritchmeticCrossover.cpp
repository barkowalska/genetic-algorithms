#include "ArithmeticCrossover.h"
#include <random>

std::vector<std::vector<double>> ArithmeticCrossover::cross(std::vector<std::reference_wrapper<std::vector<double>>>& parents) 
{

    size_t num_parents = parents.size();

    if(parents.size() != m_required_parents) 
        throw std::invalid_argument("too small parents vector");

    if (parents[0].get().size() != parents[1].get().size()) 
        throw std::invalid_argument("parents must be the same size");
       
    size_t num_genes = parents[0].get().size();
    std::vector<double> child1(num_genes, 0.0);
    std::vector<double> child2(num_genes, 0.0);

    for (size_t i = 0; i < num_parents-1; i += 2) {

        const std::vector<double>& parent1 = parents[i].get();
        const std::vector<double>& parent2 = parents[i + 1].get();


        double alpha = m_distribution(m_generator);

        for (size_t j = 0; j < num_genes; ++j) {
            child1[j] = alpha * parent1[j] + (1.0-alpha) * parent2[j];
            child2[j] = (1.0-alpha) * parent1[j] + alpha * parent2[j];
        }

    }


    return {child1, child2};
}

