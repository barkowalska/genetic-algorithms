#pragma once
#include "Mutation.h"

/*
PolynominalMutation applies polynomial mutation, where the control parameter `m_n`
defines the spread of changes, allowing for both small and large gene adjustments.
Default constructor deleted.
*/
class PolynominalMutation : public Mutation<double> {
private:
  double m_n; // Control parameter that defines the spread and shape of the polynomial distribution

public:

  /*
      Mutation method - applies  mutation to a given individual chromosome.
      Arguments:
      - A reference to a vector representing the individual to be mutated.
      Modifies the individual in-place based on mutation probability (Pm).
  */
  void mutation(std::vector<double> &) override;

  /*
  Constructor
      Arguments:
      - n: Control parameter for the polynomial degree, affecting mutation spread.
      - Pm: Probability of mutation for each gene in the chromosome.
      - min: Vector representing the lower bounds for each gene.
      - max: Vector representing the upper bounds for each gene.
      - gen: Current generation number (not directly used in this class).
  */
  PolynominalMutation(double n, double Pm, std::vector<double> min,
                      std::vector<double> max)
      : m_n(n), Mutation(max, min, Pm) {}
};