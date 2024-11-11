#include "../../include/Scaling/PowerLawScaling.h"

void PowerLawScaling::scaling(std::vector<double> &fitnessValues) {
  if (fitnessValues.empty()) {
    throw std::runtime_error("Wektor fitnessValues jest pusty");
  }

  // #pragma omp parallel for
  for (auto &fitnessValue : fitnessValues) {
    fitnessValue = std::pow(fitnessValue, m_alpha);
  }
}
