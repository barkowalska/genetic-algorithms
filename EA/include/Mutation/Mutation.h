#pragma once

#include <vector>
#include <random>
#include <stdexcept>

template<typename T>
class Mutation
{
    protected:
        std::mt19937 m_generator;
        //dziedzina dla kazdego genu w chromosomie
        std::vector<double> m_Min;//lower limit of the range
        std::vector<double> m_Max;//upper limit of the range
        std::uniform_real_distribution<double> m_distribution;

        double m_Pm;

    public:

    //the main function of the mutation, argument+ pojedynczy chromosom
    virtual void mutation(std::vector<T> &)=0;
    Mutation(std::vector<double> max, std::vector<double> min, double pm): 
       m_Max(max), m_Min(min), m_generator(std::random_device{}()), m_Pm(pm), m_distribution(0.0, 1.0) 
        {
            if(min>max) 
            {   
                //runtime_error - błąd podczas działania programu
                throw std::runtime_error("minimum value is greater than maximum value");  
            }
        }
};

