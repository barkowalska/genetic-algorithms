#include "Scaling.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <vector>

class BoltzmannScaling : public Scaling<double> {
private:
  double m_temperature;

public:
  void scaling(std::vector<double> &fitnessValues) override;
  BoltzmannScaling(double temperature = 1.0) : m_temperature(temperature){};
};

void BoltzmannScaling::scaling(std::vector<double> &fitnessValues) {
  if (fitnessValues.empty()) {
    throw std::runtime_error("Wektor fitnessValues jest pusty");
  }

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < fitnessValues.size(); i++) {
    fitnessValues[i] = std::exp(fitnessValues[i] / m_temperature);
  }
}
