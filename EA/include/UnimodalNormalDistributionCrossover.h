#include "Crossover.h"
#include <random>

//undx?
class UnimodalNormalDistributionCrossover: public Crossover<double>
{
    private:
    int m_numOffspring;
    std::uniform_real_distribution<> m_distribution;

    public:
        std::vector<std::vector<double>>  cross(std::vector<std::reference_wrapper<std::vector<double>>>&);

    UnimodalNormalDistributionCrossover(int dimenionsNumber, int numOffsprings) :
        Crossover(dimenionsNumber), m_numOffspring(numOffsprings), m_distribution(0.0, 1.0){};

};