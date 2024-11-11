#pragma once

#include <random>
#include <vector>

template <typename T> class Selection {

protected:
  std::mt19937 m_generator;

public:
  virtual std::vector<size_t> selection(const std::vector<T> &fitnessValue) = 0;
  Selection() : m_generator(std::random_device{}()) {}
};
