#include "Crossover.h"

class SimplexCrossover : public Crossover<double>
{
    private:
    double m_e;
    int m_numOffspring;
    std::uniform_real_distribution<> m_distribution;

    public:
        std::vector<std::vector<double>>  cross(std::vector<std::reference_wrapper<std::vector<double>>>&);
        std::vector<std::vector<double>>  offsprings(const std::vector<std::vector<double>>&);

        SimplexCrossover(double e, int dimenionsNumber, int numOffsprings) :
            Crossover(dimenionsNumber+1), m_e(e), m_numOffspring(numOffsprings), m_distribution(0.0, 1.0){};
};