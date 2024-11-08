#include "Crossover.h"
#include <vector>
#include <random>
#include <cmath>

class SimulatedBinaryCrossover: public Crossover<double>
{

    private:

    double m_n;                      
    double m_Pc;             
    std::uniform_real_distribution<double> m_distribution;

    public:

    std::vector<std::vector<double>> cross(std::vector<std::reference_wrapper<std::vector<double>>>&) override;
    SimulatedBinaryCrossover(double eta_c = 2.0, double crossover_prob = 0.9)
        :Crossover(2), m_n(eta_c), m_Pc(crossover_prob), m_distribution(0.0, 1.0){}
};