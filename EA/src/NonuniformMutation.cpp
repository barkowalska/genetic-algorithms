#include "NonuniformMutation.h"

//maxgen powinno byc globalne 
size_t maxgen=10;

std::vector<double> NonuniformMutation::mutation(std::vector<double> &individual)
{
    double y=0;
    std::vector<double> mutant(individual.size());
    for(size_t i=0; i<individual.size(); i++)
    {
        double rand=m_distribution(m_generator);
        if(rand<m_Pm)
        {
            rand=m_distribution(m_generator);
            if(rand>=0.5)
            {
                y=m_Max[i]-individual[i];
                mutant[i]=individual[i]+extension(y);
            }
            else
            {
                y=individual[i]-m_Min[i];
                mutant[i]=individual[i]-extension(y);
            }
        }
        else
        {
            mutant[i]=individual[i];
        }

    }
    return mutant;
}

double NonuniformMutation::extension(double y)
{
    double p=std::pow(1.0-static_cast<double>(m_gen)/maxgen, m_b);
    double rand=m_distribution(m_generator);
    double delta=y*(1.0- std::pow(rand, p));

    return delta;
}