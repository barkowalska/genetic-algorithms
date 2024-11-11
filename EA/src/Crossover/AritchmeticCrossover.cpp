#include "../../include/Crossover/ArithmeticCrossover.h"

std::vector<std::vector<double>> ArithmeticCrossover::cross(
    const std::vector<std::reference_wrapper<const std::vector<double>>>
        &parents) {
  size_t num_parents = parents.size();

  if (parents.size() != m_requiredParents)
    throw std::invalid_argument("invalid number of parents; expected 2");

  if (parents[0].get().size() != parents[1].get().size())
    throw std::invalid_argument("parents must be the same size");

  size_t num_genes = parents[0].get().size();
  std::vector<double> child1(num_genes, 0.0);
  std::vector<double> child2(num_genes, 0.0);

  const std::vector<double> &parent1 = parents[0].get();
  const std::vector<double> &parent2 = parents[1].get();

  // Generate offspring using local arithmetic crossover with unique alpha for
  // each gene
  for (size_t j = 0; j < num_genes; ++j) {
    double alpha = m_distribution(m_generator);
    child1[j] = alpha * parent1[j] + (1.0 - alpha) * parent2[j];
    child2[j] = (1.0 - alpha) * parent1[j] + alpha * parent2[j];
  }

  return {child1, child2};
}
