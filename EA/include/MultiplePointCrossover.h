#include "Crossover.h"
#include <algorithm>


class MultiplePointCrossover: public Crossover<double>
{
private:
    //liczba punktow
    size_t m_numOfPoints;
public: 
    std::vector<std::vector<double>> cross(std::vector<std::reference_wrapper<std::vector<double>>>& parents) override;

    MultiplePointCrossover(size_t points):
        Crossover(2),m_numOfPoints(points){}
};

