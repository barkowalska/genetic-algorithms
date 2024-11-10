#include "Scaling.h"
#include <omp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stdexcept>

class BoltzmannScaling : public Scalling<double>
{
private:
    double m_temperature;
public:
    void scaling(std::vector<double> &fitnessValues) override;
    BoltzmannScaling(double temperature = 1.0): m_temperature(temperature){};
};

void BoltzmannScaling::scaling(std::vector<double> &fitnessValues)
{
    if (fitnessValues.empty()) {
        throw std::runtime_error("Wektor fitnessValues jest pusty");
    }

    #pragma omp parallel for
    for (size_t i = 0; i < fitnessValues.size(); i++) {
        fitnessValues[i] = std::exp(fitnessValues[i] / m_temperature);
    }
}
