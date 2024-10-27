#include "Crossover.h"

class BlendCrossover : public Crossover<double>
{
    private:
    double alpha;//user-deﬁned parameter that controls the extent of the expansion

    public:
    std::vector<std::vector<double>> BlendCrossover::cross(std::vector<std::reference_wrapper<std::vector<double>>>&);
    BlendCrossover(double alpha=0.5) : 
        Crossover(2), alpha(alpha){}

};