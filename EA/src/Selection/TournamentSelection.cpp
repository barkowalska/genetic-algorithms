#include "../../include/Selection/TournamentSelection.h"


std::vector<size_t> TournamentSelection::selection(const std::vector<double> &fitnessValue)
{
    std::vector<size_t> mating_pool(fitnessValue.size());
    std::uniform_int_distribution<size_t> m_distribution(0.0, fitnessValue.size());

    for(size_t j=0; j<fitnessValue.size(); j++)
    {
        std::vector<size_t> chosen(m_tournamentSize);
        for(size_t i=0; i<m_tournamentSize; i++)
        {
        
            chosen[i]=m_distribution(m_generator);
        }
        size_t best=chosen[0];
        for(size_t k: chosen)
        {
            if(fitnessValue[k]<fitnessValue[best]) best=k;
        }
        mating_pool[j]=best;
        
    }
    return mating_pool;
}
   