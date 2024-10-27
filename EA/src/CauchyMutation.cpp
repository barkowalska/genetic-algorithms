 #include "CauchyMutation.h"

 std::vector<double> CauchyMutation::mutation(std::vector<double> &individual)
 {
    std::vector<double> mutant(individual.size());

    for(size_t i=0; i<individual.size(); i++)
    {
        double c=m_cauchydistribution(m_generator);

        mutant[i] =individual[i]+m_sigma*c;
    }
    return mutant;
 }