#include "Scaling.h"
#include <omp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stdexcept>

class PowerLawScaling : public Scalling<double>
{
private:
    double m_alpha;
public:
    void scaling(std::vector<double> &fitnessValues) override;
    PowerLawScaling(double alpha=1.5): m_alpha(alpha){}
};

void PowerLawScaling::scaling(std::vector<double> &fitnessValues)
{
    if (fitnessValues.empty()) {
        throw std::runtime_error("Wektor fitnessValues jest pusty");
    }

    #pragma omp parallel for
    for (size_t i = 0; i < fitnessValues.size(); i++) {
        fitnessValues[i] = std::pow(fitnessValues[i], m_alpha);
    }
}
