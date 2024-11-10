#include "MultiplePointCrossover.h"


std::vector<std::vector<double>> MultiplePointCrossover::cross(std::vector<std::reference_wrapper<std::vector<double>>>& parents) 
{

    if(parents.size() != m_requiredParents ) 
        throw std::invalid_argument("invalid number of parents; expected 2");

    if (parents[0].get().size() != parents[1].get().size()) 
        throw std::invalid_argument("parents must be the same size");
    
    size_t size = parents[0].get().size();
    std::vector<int> crossoverPoints;
    std::uniform_int_distribution<int> distribution(1, size - 1);
    while (crossoverPoints.size() < m_numOfPoints) {
        int point = distribution(m_generator);
        if (std::find(crossoverPoints.begin(), crossoverPoints.end(), point) == crossoverPoints.end()) {
            crossoverPoints.push_back(point);
        }
    }

    std::sort(crossoverPoints.begin(), crossoverPoints.end());

    std::vector<double> child1 = parents[0].get();
    std::vector<double> child2 = parents[1].get();

    bool swap=false;
    int k=0;
    for(int i=0; i<crossoverPoints.size(); i++)
    {
        if(i==crossoverPoints[k] && k<crossoverPoints.size()) 
            {
                swap=!swap;
                k++;
            }
        if(swap)std::swap(child1[i], child2[i]);
    }
    return {child1, child2};

}
