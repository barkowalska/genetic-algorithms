#include "SelectionSUS.h"
#include<omp.h>
//zwraca
std::vector<size_t> SelectionSUS::selection(const std::vector<double> &fitnessvalue)
{
    std::vector<size_t> matingPool(fitnessvalue.size());
    double sumFitnessValue=sum(fitnessvalue);
    double angle=sumFitnessValue/static_cast<double>(fitnessvalue.size());
    std::uniform_real_distribution<double> m_distribution(0.0, angle);

    double rand=m_distribution(m_generator);
    double sum=0, k=0;
    for(size_t i=0; i<fitnessvalue.size(); i++)
    {
        sum+=fitnessvalue[i];
        if(rand<sum)
        {
            matingPool[k]=i;
            k++;
            rand+=angle;
        }
    }
    return matingPool;
}

double SelectionSUS::sum(const std::vector<double> &fitnessvalue)
{
    double sum=0;
    #pragma omp parallel for reduction(+ : sum)
    for(size_t i=0; i<fitnessvalue.size(); i++)
    {
        sum+=fitnessvalue[i];
    }
    return sum;
}
