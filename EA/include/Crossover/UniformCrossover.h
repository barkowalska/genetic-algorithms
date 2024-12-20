#pragma once

#include "Crossover.h"

/*
Creates offsprings by randomly swapping genes between parent chromosomes based
on a uniform distribution. 
Default constructor is deleted.
*/
class UniformCrossover : public Crossover<double> {

public:
  /*
       Crossover method- performs sngle crossover (not for entire population)
   arggument
   - vector of references to parent chromosomes
   return
   - vector od offsprings
   */
  std::vector<std::vector<double>>
  cross(const std::vector<std::reference_wrapper<const std::vector<double>>> &)
      override;

  /*
      Constructor
  */
  UniformCrossover() : Crossover(2){};
};
