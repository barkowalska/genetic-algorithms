#include "BoundryMutation.h"
#include <random>

void BoundryMutation::mutation(std::vector<double> & individual)
{
    for(size_t i=0; i<individual.size(); i++)
    {
        if(m_distribution(m_generator)<m_Pm)
        {
            if(m_distribution(m_generator)<=0.5)
            {
                individual[i]=m_Min[i];
            }
            else{
                individual[i]=m_Max[i];
            }
        }


    }
}