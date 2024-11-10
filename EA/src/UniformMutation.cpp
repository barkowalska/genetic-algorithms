#include "UniformMutation.h"
#include <random>

void UniformMutation::mutation(std::vector<double>& individual)
{

    for(size_t i=0; i<individual.size(); i++)
    {
        if(m_distribution(m_generator)<m_Pm)
        {
            individual[i]=m_distribution(m_generator)*(m_Max[i]-m_Min[i])+m_Min[i];
        }
    }
}