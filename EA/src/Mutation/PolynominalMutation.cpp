#include "../../include/Mutation/PolynominalMutation.h"

void PolynominalMutation::mutation(std::vector<double> &individual)
{
    for(size_t i =0; i<individual.size();i++ )
    {
        if(m_distribution(m_generator)<m_Pm)
        {
            double delta_max=m_Max[i]-m_Min[i];
            double rand=m_distribution(m_generator);
            if(rand<0.5)
            {
                double delta_q=std::pow(2.0*rand, 1.0/m_n+1.0)-1.0;
                individual[i]=individual[i]+delta_max*delta_q;
            }
            else
            {
                double delta_q=1.0-std::pow(2*(1-rand), 1/m_n+1.0);
                individual[i]=individual[i]+delta_max*delta_q;
            }

        }
    }
}

