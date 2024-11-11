#pragma once
#include "Crossover.h"

/*
Creates offspring based on the centroid of parent vectors expanded by a
user-defined factor.  
Default constructor is deleted.
*/
class SimplexCrossover : public Crossover<double> {
private:
  double m_e;         // Expansion factor for generating new offspring
  int m_numOffspring; // Number of offspring to generate
  std::uniform_real_distribution<> m_distribution;

  /*
      Creates offsprings inside the area created by parent+1 vertices, used
  inside cross Arguments:
  - A vector of expanded vertices.
  Returns:
  - A vector of offspring chromosomes.
  */
  std::vector<std::vector<double>>
  offsprings(const std::vector<std::vector<double>> &);

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
  Arguments :
  - Expansion factor (double e).
  - Number of parent vectors (dimensionsNumber + 1).
  - Number of offspring to generate (int numOffsprings).
  */
  SimplexCrossover(double e, int dimenionsNumber, int numOffsprings)
      : Crossover(dimenionsNumber + 1), m_e(e), m_numOffspring(numOffsprings),
        m_distribution(0.0, 1.0){};
};