#include "PolynominalMutation.h"

std::vector<double> PolynominalMutation::mutation(std::vector<double> &individual)
{
    std::vector<double> mutant(individual.size());
    for(size_t i =0; i<individual.size();i++ )
    {
        double rand=m_distribution(m_generator);
        if(rand<m_Pm)
        {
            double delta_max=m_Max[i]-m_Min[i];
            rand=m_distribution(m_generator);
            if(rand<0.5)
            {
                double delta_q=std::pow(2.0*rand, 1.0/m_n+1.0)-1.0;
                mutant[i]=individual[i]+delta_max*delta_q;
            }
            else
            {
                double delta_q=1.0-std::pow(2*(1-rand), 1/m_n+1.0);
                mutant[i]=individual[i]+delta_max*delta_q;
            }

        }
        else
        {
            mutant[i]=individual[i];
        }
    }
    return mutant;
}

