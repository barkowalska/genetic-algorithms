#pragma once

#include "Scaling.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <vector>

/*
    PowerLawScaling class:
    A scaling technique that raises each fitness value to a power `m_alpha`.
    This approach amplifies or diminishes the impact of fitness differences based on the
    value of `m_alpha`, making it useful when fitness values need to be emphasized or smoothed.
    Suitable for objective functions where fitness values are positive and may benefit from non-linear scaling.
    Default constructor deleted.
*/
class PowerLawScaling : public Scaling<double> {
private:
  double m_alpha;// Exponent parameter that controls the degree of scaling

public:

  /*
      Scaling method - applies power law scaling to fitness values in place.
      Arguments:
      - A reference to a vector of fitness values for the population.
      Requirements:
      - `fitnessValues` should contain positive double values representing the fitness
        of each individual. Values are raised to the power `m_alpha`, with values > 1.0 amplifying differences.
      Throws:
      - std::runtime_error if `fitnessValues` is empty.
  */
  void scaling(std::vector<double> &fitnessValues) override;

    /*
      Constructor
          Arguments:
          - alpha: Exponent that adjusts the scaling effect; default is 1.5.
  */
  PowerLawScaling(double alpha = 1.5) : m_alpha(alpha) {}
};

