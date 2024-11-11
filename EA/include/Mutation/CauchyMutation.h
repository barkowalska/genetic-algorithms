#include "Mutation.h"

// CauchyMutation ( double sigma=1.0,  double ProbabilityOfMutation,
// std::vector<double>  min , std::vector<double> max )
class CauchyMutation : public Mutation<double> {
private:
  // scale parameter dla rozkladu cauchyego
  double m_sigma;
  std::cauchy_distribution<double> m_cauchydistribution;

public:
  void mutation(std::vector<double> &) override;
  CauchyMutation(double sigma, double Pm, std::vector<double> min,
                 std::vector<double> max)
      : Mutation(max, min, Pm), m_cauchydistribution(0.0, 1.0), m_sigma(sigma) {
  }
};