#include "SimulatedBinaryCrossover.h"





std::vector<std::vector<double>> SimulatedBinaryCrossover::cross(std::vector<std::reference_wrapper<std::vector<double>>>& parents)
{
    std::vector<std::vector<double>> offspring;

    if(parents.size() != m_required_parents) 
        throw std::invalid_argument("too small parents vector");

    if (parents[0].get().size() != parents[1].get().size()) 
        throw std::invalid_argument("parents must be the same size");
        
    int num_parents = parents.size();


    const std::vector<double>& parent1 = parents[0].get();
    const std::vector<double>& parent2 = parents[1].get();

    size_t size = parent1.size();
    std::vector<double> child1(size, 0.0);
    std::vector<double> child2(size, 0.0);

    for (size_t j = 0; j < size; ++j) {
        double u = m_distribution(m_generator);

        if (m_distribution(m_generator) <= m_Pc) {
            double beta;
            if (u <= 0.5) {
                beta = pow(2.0 * u, 1.0 / (m_n + 1.0));
            } else {
                beta = pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (m_n + 1.0));
            }

            child1[j] = 0.5 * (parent1[j] + parent2[j]) + 0.5 * beta * (parent1[j] - parent2[j]);
            child2[j] = 0.5 * (parent1[j] + parent2[j]) + 0.5 * beta * (parent2[j] - parent1[j]);
        } else {
            child1[j] = parent1[j];
            child2[j] = parent2[j];
        }
    }

    return {child1, child2};
}