#include "Scaling.h"
#include <algorithm>
#include <omp.h>
#include <stdexcept>
#include <vector>

// zmiana na szukanie maksimum
class FitnessTransferral : public Scaling<double> {
public:
  // zmiana na szukanie maksimum
  void scaling(std::vector<double> &fitnessValues) override;
};

void FitnessTransferral::scaling(std::vector<double> &fitnessValues) {
  if (fitnessValues.empty()) {
    throw std::runtime_error("Wektor fitnessValues jest pusty");
  }

  double Fmax = *std::max_element(fitnessValues.begin(), fitnessValues.end());

// Przeprowadzanie transferral, tj. zmiana funkcji na max(Fmax - f(x))
#pragma omp parallel for
  for (size_t i = 0; i < fitnessValues.size(); i++) {
    fitnessValues[i] = Fmax - fitnessValues[i];
  }
}
