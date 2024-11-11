#pragma once

#include "Scaling.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <vector>


/*
    BoltzmannScaling class:
    A scaling technique that applies Boltzmann scaling to fitness values, using an exponential function.
    Suitable for objective functions where fitness values need to be scaled to emphasize differences.
    The temperature parameter `m_temperature` adjusts the scaling sensitivity, with lower values resulting
    in greater sensitivity to differences in fitness values.
    Default Constructor deleted.
*/
class BoltzmannScaling : public Scaling<double> {
private:
  double m_temperature;// Temperature parameter for adjusting the scaling sensitivity

public:

  /*
      Scaling method - applies Boltzmann scaling to fitness values in place.
      Arguments:
      - A reference to a vector of fitness values for the population.
      Requirements:
      - `fitnessValues` should contain positive double values representing the fitness
        of each individual. Higher temperatures reduce the impact of scaling differences,
        while lower temperatures amplify them.
      Throws:
      - std::runtime_error if `fitnessValues` is empty.
  */
  void scaling(std::vector<double> &fitnessValues) override;

    /*
      Constructor
          Arguments:
          - temperature: Adjusts sensitivity of scaling; default is 1.0.
  */
  BoltzmannScaling(double temperature = 1.0) : m_temperature(temperature){};
};

