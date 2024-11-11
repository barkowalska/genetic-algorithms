#pragma once

#include <random>
#include <vector>

/*
  Selection based class
*/
template <typename T> class Selection {

protected:
  std::mt19937 m_generator;

public:

/*
    Selection method - selects individuals based on a specific selection strategy.
    Arguments:
    - A vector of fitness values for the population.
    Returns:
    - A vector of indices representing the selected individuals.
    To be implemented by derived classes with different selection strategies.
*/
  virtual std::vector<size_t> selection(const std::vector<T> &fitnessValue) = 0;

  /*
      Constructor
      Initializes the random number generator for selection operations.
  */
  Selection() : m_generator(std::random_device{}()) {}
};
