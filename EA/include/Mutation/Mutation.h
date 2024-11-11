#pragma once

#include <random>
#include <stdexcept>
#include <vector>

/*
    Base Mutation class template for genetic algorithm mutations.
    Provides common attributes and virtual method for derived mutation strategies.
*/
template <typename T> class Mutation {
protected:
  std::mt19937 m_generator;
  std::vector<double> m_Min; // lower limit of the range
  std::vector<double> m_Max; // upper limit of the range
  std::uniform_real_distribution<double> m_distribution;

  double m_Pm;// Probability of mutation for each gene


public:
  /*
      Virtual Mutation method - applies  mutation to a given individual chromosome.
      Arguments:
      - A reference to a vector representing the individual to be mutated.
      Modifies the individual in-place by setting genes to boundary values.
  */  
 virtual void mutation(std::vector<T> &) = 0;

  /*
  Constructor
      Arguments:
      - max: Vector representing the upper bounds for each gene.
      - min: Vector representing the lower bounds for each gene.
      - pm: Probability of mutation for each gene.
      Throws:
      - std::runtime_error if any value in min exceeds the corresponding value in max.
  */
  Mutation(std::vector<double> max, std::vector<double> min, double pm)
      : m_Max(max), m_Min(min), m_generator(std::random_device{}()), m_Pm(pm),
        m_distribution(0.0, 1.0) {
    if (min > max) {
      // runtime_error - błąd podczas działania programu
      throw std::runtime_error("minimum value is greater than maximum value");
    }
  }
};
