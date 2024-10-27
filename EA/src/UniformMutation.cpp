#include "UniformMutation.h"
#include <random>

std::vector<double> UniformMutation::mutation(std::vector<double>& individual)
{
    std::vector<double> mutant (individual.size());
    for(size_t i=0; i<individual.size(); i++)
    {
        double random=m_distribution(m_generator);
        if(random<m_Pm)
        {
            mutant[i]=m_distribution(m_generator)*(m_Max[i]-m_Min[i])+m_Min[i];
        }
        else{
            mutant[i]=individual[i];
        }

    }
    return mutant;
}