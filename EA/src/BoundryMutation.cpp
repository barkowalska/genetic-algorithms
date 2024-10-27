#include "BoundryMutation.h"
#include <random>

std::vector<double> BoundryMutation::mutation(std::vector<double> & individual)
{
    std::vector<double> mutant (individual.size());
    for(size_t i=0; i<individual.size(); i++)
    {
        double random=m_distribution(m_generator);
        if(random<m_Pm)
        {
            random=m_distribution(m_generator);
            if(random<=0.5)
            {
                mutant[i]=m_Min[i];
            }
            else{
                mutant[i]=m_Max[i];
            }
        }
        else{
            mutant[i]=individual[i];
        }

    }
    return mutant;
}