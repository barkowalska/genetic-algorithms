#include "../../include/Scaling/BoltzmannScaling.h"


void BoltzmannScaling::scaling(std::vector<double> &fitnessValues) {
  if (fitnessValues.empty()) {
    throw std::runtime_error("Wektor fitnessValues jest pusty");
  }

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < fitnessValues.size(); i++) {
    fitnessValues[i] = std::exp(fitnessValues[i] / m_temperature);
  }
}
