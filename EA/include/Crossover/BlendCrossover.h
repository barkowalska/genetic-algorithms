#pragma once

#include "Crossover.h"
/*
Creates offspring by generating new genes within an expanded range controlled by
the user-defined alpha parameter. The alpha parameter determines how far beyond
the parent gene range the new genes can be.
 Default constructor is deleted.
*/
class BlendCrossover : public Crossover<double> {
private:
  double alpha; // user-deﬁned parameter that controls the extent of the expansion

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
  - Number of required parents (fixed at 2).
  - Alpha value to control the expansion range (default is 0.5).
  */
  BlendCrossover(double alpha = 0.5) : Crossover(2), alpha(alpha) {}
};