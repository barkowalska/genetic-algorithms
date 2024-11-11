#pragma once

#include "Scaling.h"
#include <algorithm>
#include <omp.h>
#include <stdexcept>
#include <vector>



/*
    LinearScaling class:
    A scaling technique that adjusts fitness values using a linear transformation.
    This approach rescales fitness values to preserve proportional differences while
    ensuring all values remain non-negative. Suitable for fitness distributions where
    values vary significantly but require linear adjustment for selection.
*/
class LinearScaling : public Scaling<double> {
public:


  /*
      Scaling method - applies linear scaling to fitness values in place.
      Arguments:
      - A reference to a vector of fitness values for the population.
      Requirements:
      - `fitnessValues` should be a non-empty vector of positive doubles representing the fitness of each individual.
      Throws:
      - std::runtime_error if `fitnessValues` is empty.
  */
  void scaling(std::vector<double> &fitnessValues) override;
};

