#include "Scaling.h"
#include <omp.h>
#include <algorithm>
#include <vector>
#include <stdexcept>


class LinearScaling: public Scalling<double>
{
public:
    void scaling( std::vector<double> &fitnessValues) override;
};

void LinearScaling::scaling( std::vector<double> &fitnessValues)
{
    if (fitnessValues.empty()) {
        throw std::runtime_error("Wektor fitnessValues jest pusty");
    }

    double sum=0.0;
    #pragma omp parallel for reduction(+ : sum)
    for(size_t i=0; i<fitnessValues.size(); i++)
    {
        sum+=fitnessValues[i];
    }
    double Favg=sum/fitnessValues.size();

    double Fmax = *std::max_element(fitnessValues.begin(),fitnessValues.end());
    double Fmin = *std::min_element(fitnessValues.begin(),fitnessValues.end());

    double a = (Favg) / (Fmax - Favg);
    double b = Favg * (Fmax - 2 * Favg) / (Fmax - Favg);

    if ((a * Fmin + b) < 0) 
    {
        a = Favg / (Favg - Fmin);
        b = -Favg * Fmin / (Favg - Fmin);
    }

    #pragma omp parallel for
    for (auto &value : fitnessValues) 
    {
        value = a * value + b;
    }
}
