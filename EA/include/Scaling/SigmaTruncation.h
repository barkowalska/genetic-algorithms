#pragma once

#include "Scaling.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <vector>


/*
    SigmaTruncation class:
    A scaling technique used for objective functions with high variance in fitness values.
    This method reduces the impact of outliers by scaling fitness values based on the
    average fitness and standard deviation (sigma), truncating values below a threshold.
    Useful when the distribution of fitness values has extreme values that could dominate selection.
    Default constructor deleted.
*/
class SigmaTruncation : public Scaling<double> {
private:
  int m_c;// Control parameter to adjust scaling based on standard deviation


public:

  /*
      Scaling method - applies sigma truncation scaling to fitness values in place.
      Arguments:
      - A reference to a vector of fitness values for the population.
      Requirements:
      - `fitnessValues` should be a non-empty vector of positive double values representing
        the fitness of each individual in the population. These values are expected to
        vary significantly to benefit from sigma truncation.
      Throws:
      - std::runtime_error if `fitnessValues` is empty.
  */
  void scaling(std::vector<double> &fitnessValues) override;

  /*
      Constructor
          Arguments:
          - c: Control parameter for scaling, determining the effect of sigma on truncation.
  */
  SigmaTruncation(int c) : m_c(c) {}
};
