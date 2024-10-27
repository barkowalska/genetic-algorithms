#include "UnimodalNormalDistributionCrossover.h"


std::vector<std::vector<double>>  UnimodalNormalDistributionCrossover::cross(std::vector<std::reference_wrapper<std::vector<double>>>& parents)
{
    size_t dim=parents[0].get().size();
    size_t n=dim+1;

   
    if(parents.size() != m_required_parents) 
        throw std::invalid_argument("too small parents vector");
/*
    if (parents[0].get().size() != parents[1].get().size()) 
        throw std::invalid_argument("parents must be the same size");
  */


}