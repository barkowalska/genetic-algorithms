#pragma once

#include "Mutation.h"

/*
Applying mutations based on the Cauchy distribution,
allowing for larger jumps in the search space compared to Gaussian-based mutations.
Default constructor deleted.
*/
class CauchyMutation : public Mutation<double> {
private:
  double m_sigma;// Scale parameter for the Cauchy distribution
  std::cauchy_distribution<double> m_cauchydistribution;

public:

  /*
      Mutation method - applies mutation to a given individual chromosome.
      Arguments:
      - A reference to a vector representing the individual to be mutated.
      Modifies the individual in-place based on mutation probability (Pm).
  */
  void mutation(std::vector<double> &) override;

    /*
  Constructor
      Arguments:
      - sigma: Scale parameter for mutation magnitude.
      - Pm: Probability of mutation for each gene.
      - min: Vector representing the lower bounds for each gene.
      - max: Vector representing the upper bounds for each gene.
  */
  CauchyMutation(double sigma, double Pm, std::vector<double> min,
                 std::vector<double> max)
      : Mutation(max, min, Pm), m_cauchydistribution(0.0, 1.0), m_sigma(sigma) {
  }
};