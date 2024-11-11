#include "../../include/Scaling/SigmaTruncation.h"


void SigmaTruncation::scaling(std::vector<double> &fitnessValues) {
  if (fitnessValues.empty()) {
    throw std::runtime_error("Wektor fitnessValues jest pusty");
  }

  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < fitnessValues.size(); i++) {
    sum += fitnessValues[i];
  }
  double Favg = sum / fitnessValues.size();

  double varianceSum = 0.0;
#pragma omp parallel for reduction(+ : varianceSum)
  for (size_t i = 0; i < fitnessValues.size(); i++) {
    varianceSum += (fitnessValues[i] - Favg) * (fitnessValues[i] - Favg);
  }
  double sigma = std::sqrt(varianceSum / fitnessValues.size());

#pragma omp parallel for
  for (size_t i = 0; i < fitnessValues.size(); i++) {
    fitnessValues[i] = fitnessValues[i] - (Favg - m_c * sigma);
    if (fitnessValues[i] < 0) {
      fitnessValues[i] = 0;
    }
  }
}
