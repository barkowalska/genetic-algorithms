#pragma once

#include <vector>


/*
    Scaling base class:
    Provides an interface for fitness scaling strategies in genetic algorithms.
*/
template <typename T> class Scaling {
public:
  /*
      Scaling method - applies a fitness scaling strategy in place.
      Arguments:
      - A reference to a vector of fitness values for the population.
      Modifies the fitness values directly based on the specific scaling approach.
  */
  virtual void scaling(std::vector<T> &fitnessValue) = 0;
};
