#pragma once
#include "Mutation.h"

/*
Setting genes to their boundary values
(minimum or maximum) based on a mutation probability, allowing for extreme
adjustments in the chromosome.
Default constructor deleted.
*/
class BoundryMutation : public Mutation<double> {

public:

  /*
      Mutation method - applies mutation to a given individual chromosome.
      Arguments:
      - A reference to a vector representing the individual to be mutated.
      Modifies the individual in-place based on mutation probability (Pm).
  */
  void mutation(std::vector<double> &) override;


  /*
  Constructor
      Arguments:
      - pm: Probability of mutation for each gene.
      - min: Vector representing the lower bounds for each gene.
      - max: Vector representing the upper bounds for each gene.
  */
  BoundryMutation(double pm, std::vector<double> min, std::vector<double> max)
      : Mutation(max, min, pm) {}
};