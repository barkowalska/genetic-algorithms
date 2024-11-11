#pragma once

#include "Selection.h"
#include <random>
#include <vector>
/*
    Individuals are chosen based on a tournament strategy.
    In each tournament, a subset of individuals is selected, and the fittest among them is chosen.
    Default constructor deleted.
*/
class TournamentSelection : public Selection<double> {
private:
  int m_tournamentSize; // Number of individuals participating in each tournament

public:
/*
    Selection method - selects individuals based on a specific selection strategy.
    Arguments:
    - A vector of fitness values for the population.
    Returns:
    - A vector of indices representing the selected individuals.
    To be implemented by derived classes with different selection strategies.
*/
  std::vector<size_t> selection(const std::vector<double> &) override;

  /*
      Constructor
          Arguments:
          - k: Number of individuals participating in each tournament.
  */
   TournamentSelection(int k) : m_tournamentSize(k) {}
};
