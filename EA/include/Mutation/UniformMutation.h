#pragma once

#include "Mutation.h"

/*
Replacing genes in a chromosome with new random values within specified minimum 
and maximum bounds when the mutation probability condition is met.
Default constructor deleted
*/
class UniformMutation : public Mutation<double> {

public:

  /*
      Mutation method - applies  mutation to a given individual chromosome.
      Arguments:
      - A reference to a vector representing the individual to be mutated.
      Modifies the individual in-place based on mutation probability (Pm).
  */
  void mutation(std::vector<double> &) override;


  /*
  Constructor
      Arguments:
      - Pm: Probability of mutation for each gene.
      - min: Vector representing the lower bounds for each gene.
      - max: Vector representing the upper bounds for each gene.
      Initializes the base Mutation class with max, min, and Pm.
  */
  UniformMutation(double Pm, std::vector<double> min, std::vector<double> max)
      : Mutation(max, min, Pm) {}
};