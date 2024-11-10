#include "SimplexCrossover.h"
#include <random>


std::vector<std::vector<double>>  SimplexCrossover::cross(std::vector<std::reference_wrapper<std::vector<double>>>& parent)
{
    size_t n = parent.size() - 1; 
    size_t dim = parent[0].get().size(); 

    if(parent.size() != m_requiredParents) 
        throw std::invalid_argument("too small parents vector");

    for (size_t i = 1; i < n; ++i)
    {
        if (parent[i].get().size() != dim)
            throw std::invalid_argument("All parents must have the same dimension.");
    }


    std::vector<double> centroid(dim, 0.0);
    for (const auto& parent : parent) {
        for (size_t i = 0; i < dim; ++i) {
            centroid[i] += parent.get()[i];
        }
    }
    for (size_t i = 0; i < dim; ++i) {
        centroid[i] /= (n + 1);
    }

    std::vector<std::vector<double>> expanded_vertices(n + 1, std::vector<double>(dim));
    for (size_t i = 0; i <= n; ++i) {
        const std::vector<double>& parent1 = parent[i].get();

        for (size_t j = 0; j < dim; ++j) {
            expanded_vertices[i][j] = parent1[j] + m_e * (parent1[j] - centroid[j]);
        }
    }

    return offsprings(expanded_vertices);

}


std::vector<std::vector<double>>  SimplexCrossover::offsprings(const std::vector<std::vector<double>>& vertices) {
    size_t n = vertices.size() - 1; 
    std::vector<std::vector<double>> offspring_list;



    for (size_t k = 0; k < m_numOffspring; ++k) {
        std::vector<double> weights(n + 1);
        double sum = 0.0;

        for (size_t i = 0; i <= n; ++i) {
            double u = m_distribution(m_generator);
            weights[i] = -std::log(u);
            sum += weights[i];
        }

        for (size_t i = 0; i <= n; ++i) {
            weights[i] /= sum;
        }

        std::vector<double> offspring(vertices[0].size(), 0.0);
        for (size_t i = 0; i <= n; ++i) {
            for (size_t j = 0; j < offspring.size(); ++j) {
                offspring[j] += weights[i] * vertices[i][j];
            }
        }

        offspring_list.push_back(offspring);
    }

    return offspring_list;
}


