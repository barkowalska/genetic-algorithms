#pragma once

#include<vector>
#include<memory>
#include<random>
#include<array>
#include <stdexcept>

//Basic class
template<typename T>
class Crossover
{
protected: 
    const size_t m_requiredParents;//number of chromosomes required for crossover
    std::mt19937 m_generator;

public:
    /*
        Constructor required to be defined in a derived class
    Arguments:
    - Number of required parents to complete the crossover
    */
    Crossover(size_t requiredParents ):
        m_requiredParents(requiredParents), m_generator(std::random_device{}()){}

    /*
        virtual Crossover method- performs sngle crossover (not for entire population)
    */
    virtual std::vector<std::vector<T>> cross(std::vector<std::reference_wrapper<std::vector<T>>>& parents) = 0; 

    size_t getNumberOfParents(){return m_requiredParents;}

};

