#pragma once

#include "Crossover.h"
#include <cmath>
#include <random>
#include <vector>

/*
SBX, uses a calculated β coefficient to adaptively generate offspring with
controlled similarity to the parents.
*/
class SimulatedBinaryCrossover : public Crossover<double> {
private:
  double m_n;  // Distribution index that defines the spread of the offspring
  double m_Pc; // Crossover probability
  std::uniform_real_distribution<double> m_distribution;

public:
  /*
      Crossover method- performs sngle crossover (not for entire population)
  Arguments :
  - vector of references to parent chromosomes
  Returns :
  - vector od offsprings
  */
  std::vector<std::vector<double>>
  cross(const std::vector<std::reference_wrapper<const std::vector<double>>> &)
      override;

  /*
  Constructor
      Arguments:
      - n: Distribution index (default value is 2.0).
      - crossover_prob: Probability of crossover (default value is 0.9).
  */
  SimulatedBinaryCrossover(double n = 2.0, double crossover_prob = 0.9)
      : Crossover(2), m_n(n), m_Pc(crossover_prob), m_distribution(0.0, 1.0) {}
};