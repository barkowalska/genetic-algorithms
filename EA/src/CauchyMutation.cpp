 #include "CauchyMutation.h"

void CauchyMutation::mutation(std::vector<double> &individual)
 {

    for(size_t i=0; i<individual.size(); i++)
    {
        if(m_distribution(m_generator)<m_Pm)
        {
            double c=m_cauchydistribution(m_generator);
            individual[i] =individual[i]+m_sigma*c;
        }
    }
 }