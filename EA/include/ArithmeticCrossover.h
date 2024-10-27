#include"Crossover.h"

class ArithmeticCrossover: public Crossover<double>
{
    private:
    std::uniform_real_distribution<double> m_distribution;
    public:
    std::vector<std::vector<double>> cross(std::vector<std::reference_wrapper<std::vector<double>>>&) ;
    
    ArithmeticCrossover(): 
        Crossover(2), m_distribution(0.0, 1.0){};
};

/*
    y1 = crossover->cross({x1, x2, x3});

*/