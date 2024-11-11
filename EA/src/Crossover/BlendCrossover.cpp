#include "../../include/Crossover/BlendCrossover.h"
#include <random>

std::vector<std::vector<double>> BlendCrossover::cross(
    const std::vector<std::reference_wrapper<const std::vector<double>>>
        &parents) {
  if (parents.size() != m_requiredParents)
    throw std::invalid_argument("invalid number of parents; expected 2");

  if (parents[0].get().size() != parents[1].get().size())
    throw std::invalid_argument("parents must be the same size");

  size_t num_genes = parents[0].get().size();
  std::vector<double> child1(num_genes, 0.0);
  std::vector<double> child2(num_genes, 0.0);

  for (size_t i = 0; i < num_genes; i++) {

    double x1 = parents[0].get()[i];
    double x2 = parents[1].get()[i];
    if (x1 > x2)
      std::swap(x1, x2);
    double d = x2 - x1;
    double lower_bound = x1 - alpha * d;
    double upper_bound = x2 + alpha * d;
    std::uniform_real_distribution<> dis(lower_bound, upper_bound);
    child1[i] = dis(m_generator);
    child2[i] = dis(m_generator);
  }
  return {child1, child2};
};
