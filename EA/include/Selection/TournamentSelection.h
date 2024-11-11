#include "Selection.h"
#include <random>
#include <vector>
/*
    selects individuals based on a tournament strategy <double>
*/
class TournamentSelection : public Selection<double> {
private:
  // Number of individuals participating in each tournament
  int m_tournamentSize;

public:
  /*
  argument-vector of fitnessValues
  return-vector of selected best values indices
  */
  std::vector<size_t> selection(const std::vector<double> &) override;

  // arguments-tournament size
  TournamentSelection(int k) : m_tournamentSize(k) {}
};
