#pragma once

#include <vector>

template <typename T> class Scaling {
public:
  /*
      Scaling fitness value in place
      Parameter:
      -std::vector<T> &fitnessValue
  */
  virtual void scaling(std::vector<T> &fitnessValue) = 0;
};
