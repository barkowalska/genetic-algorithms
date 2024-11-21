#pragma once

#include "Mutation.h"

/*
Mutation magnitude decreases over generations, allowing for 
finer tuning as the algorithm progresses.
Default constructor deleted

!!!!UWAGA!!!!!
zeby przekazywac tutaj numer generacji zrobic statyczna klase
 widziana w GA , ktora ma pola statyczne widziane wszedzie np numer generacji
*/
class NonuniformMutation : public Mutation<double> {
private:
  double m_b; // Parameter to control the annealing speed (degree of non-uniformity)
  size_t m_gen; // Current generation
  size_t m_maxgen; // Maximum number of generations


public:

  /*
      Mutation method - applies  mutation to a given individual chromosome.
      Arguments:
      - A reference to a vector representing the individual to be mutated.
      Modifies the individual in-place based on mutation probability (Pm).
  */
  void mutation(std::vector<double> &) override;


  /*
      Calculates the extension value for the mutation based on current generation and bounds.
      Arguments:
      - y: The distance to the bound (either min or max).
      Returns:
      - The calculated mutation value.
  */
  double extension(double y);


  /*
  Constructor
      Arguments:
      - maxgen: Maximum number of generations for the algorithm.
      - b: Parameter to control the annealing speed.
      - Pm: Probability of mutation for each gene.
      - min: Vector representing the lower bounds for each gene.
      - max: Vector representing the upper bounds for each gene.
      - gen: Current generation number.
  */
  NonuniformMutation(size_t maxgen, double b, double Pm,
                     std::vector<double> min, std::vector<double> max,
                     size_t gen)
      : m_gen(gen), Mutation(max, min, Pm), m_b(b), m_maxgen(maxgen) {}
};