#include "Crossover.h"

class UniformCrossover : public Crossover<double>
{

public:
    std::vector<std::vector<double>> cross(std::vector<std::reference_wrapper<std::vector<double>>>&) override;
    UniformCrossover( double probability=0.5) :
        Crossover(2){};
};

