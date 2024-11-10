#include "../../include/Mutation/NonuniformMutation.h"


void NonuniformMutation::mutation(std::vector<double> &individual)
{
    double y=0;
    for(size_t i=0; i<individual.size(); i++)
    {
        if(m_distribution(m_generator)<m_Pm)
        {
            if(m_distribution(m_generator)>=0.5)
            {
                y=m_Max[i]-individual[i];
                individual[i]=individual[i]+extension(y);
            }
            else
            {
                y=individual[i]-m_Min[i];
                individual[i]=individual[i]-extension(y);
            }
        }

    }
}

double NonuniformMutation::extension(double y)
{
    double p=std::pow(1.0-static_cast<double>(m_gen)/m_maxgen, m_b);
    double delta=y*(1.0- std::pow(m_distribution(m_generator), p));

    return delta;
}