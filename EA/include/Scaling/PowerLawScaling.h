#include "Scaling.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <vector>

class PowerLawScaling : public Scaling<double> {
private:
  double m_alpha;

public:
  void scaling(std::vector<double> &fitnessValues) override;
  PowerLawScaling(double alpha = 1.5) : m_alpha(alpha) {}
};

void PowerLawScaling::scaling(std::vector<double> &fitnessValues) {
  if (fitnessValues.empty()) {
    throw std::runtime_error("Wektor fitnessValues jest pusty");
  }

  // #pragma omp parallel for
  for (auto &fitnessValue : fitnessValues) {
    fitnessValue = std::pow(fitnessValue, m_alpha);
  }
}
