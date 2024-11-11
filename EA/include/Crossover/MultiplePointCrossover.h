#pragma once

#include "Crossover.h"
#include <algorithm>

/*
Performs crossover at multiple points along the chromosomes.
 Default constructor is deleted.
*/
class MultiplePointCrossover : public Crossover<double> {
private:
  size_t m_numOfPoints; // The number of crossover points is controlled by the
                        // user-defined parameter m_numOfPoints.

public:
  /*
       Crossover method- performs sngle crossover (not for entire population)
   Arguments :
   - vector of references to parent chromosomes
   Returns :
   - vector od offsprings
   */
  std::vector<std::vector<double>>
  cross(const std::vector<std::reference_wrapper<const std::vector<double>>>
            &parents) override;

  /*
      Constructor
  Arguments:
  - Number of crossover points (size_t points).
  */
  MultiplePointCrossover(size_t points) : Crossover(2), m_numOfPoints(points) {}
};
