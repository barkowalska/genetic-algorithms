#include "Crossover.h"
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>

class SimulatedBinaryCrossover: public Crossover<double>
{

    private:

    double m_n;                      
    double m_Pc;             
    std::default_random_engine m_generator;
    std::uniform_real_distribution<> m_distribution;

    public:

    std::vector<std::vector<double>> cross(std::vector<std::reference_wrapper<std::vector<double>>>&);
    SimulatedBinaryCrossover(double eta_c = 2.0, double crossover_prob = 0.9)
        :Crossover(2), m_n(m_n), m_Pc(m_Pc), m_distribution(0.0, 1.0){}
};