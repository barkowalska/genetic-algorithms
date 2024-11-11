#pragma once

#include "Scaling.h"
#include <algorithm>
#include <omp.h>
#include <stdexcept>
#include <vector>

/*
    FitnessTransferral class:
    A scaling technique that inverts fitness values relative to the maximum fitness.
    This approach transforms each fitness value to emphasize differences from the maximum,
    making it suitable for cases where minimization is preferred or where the maximum
    value should represent the best solution.
*/
class FitnessTransferral : public Scaling<double> {
public:

  /*
      Scaling method - applies fitness transferral by inverting each fitness value relative to Fmax.
      Arguments:
      - A reference to a vector of fitness values for the population.
      Requirements:
      - `fitnessValues` should be a non-empty vector of positive double values, representing
        the fitness of each individual in the population. This method shifts all values
        based on the maximum fitness value.
      Throws:
      - std::runtime_error if `fitnessValues` is empty.
  */  void scaling(std::vector<double> &fitnessValues) override;
};

