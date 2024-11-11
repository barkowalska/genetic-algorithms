#pragma once

#include "Selection.h"
#include <random>


/*
  Combines tournament selection with Boltzmann scaling, where the temperature `m_t`
  adjusts selection pressure for balanced exploration and exploitation.
  Default constructor deleted.
*/
class BoltzmannTournamentSelection : public Selection<double> {
private:
  size_t m_tournamentSize;// Number of individuals participating in each tournament
  double m_t;// Temperature parameter for Boltzmann scaling

public:

  /*
      Selection method - selects individuals based on Boltzmann-scaled tournament selection.
      Arguments:
      - A vector of fitness values for the population.
      Returns:
      - A vector of indices representing the selected individuals.
  */
  std::vector<size_t> selection(const std::vector<double> &) override;

    /*
      Constructor
          Arguments:
          - k: Number of individuals in each tournament.
          - t: Temperature parameter for Boltzmann scaling.
  */
  BoltzmannTournamentSelection(int k, double t) : m_tournamentSize(k), m_t(t) {}
};

