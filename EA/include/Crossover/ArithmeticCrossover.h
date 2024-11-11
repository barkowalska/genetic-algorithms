#pragma once

#include "Crossover.h"

/*Crossover with a unique alpha(random) coefficient for each gene to create
 * diverse offspring*/
class ArithmeticCrossover : public Crossover<double> {
private:
  std::uniform_real_distribution<double> m_distribution;

public:
  /*
     Crossover method- performs sngle crossover (not for entire population)
 Arguments :
 - vector of references to parent chromosomes
 Returns :
 - vector od offsprings
 */
  std::vector<std::vector<double>>
  cross(const std::vector<std::reference_wrapper<const std::vector<double>>> &)
      override;

  /*Constructor
  Arguments :
  - Number of required parents (fixed at 2).
  - distribution range (0.0,1.0)
  */
  ArithmeticCrossover() : Crossover(2), m_distribution(0.0, 1.0){};
};
