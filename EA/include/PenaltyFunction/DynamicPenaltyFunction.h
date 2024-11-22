#pragma once


#include "PenaltyFunction.h"

class DynamicPenaltyFunction : public PenaltyFunction<double>
{
    int m_a;
    int m_b;
    double m_c;
 

public:
    DynamicPenaltyFunction(std::vector<double> max, std::vector<double> min, size_t gen, int a = 2, int b = 2, double c = 0.5)
        : PenaltyFunction<double>(max, min, gen), m_a(a), m_b(b), m_c(c) {}

    void penaltyFunction(size_t generation, double& fitnessValue, std::vector<double>& chromosom) override;
};



void DynamicPenaltyFunction::penaltyFunction(size_t generation, double& fitnessValue, std::vector<double>& chromosom) {
    double penalty = 0.0;

    for (size_t i = 0; i < chromosom.size(); ++i) {
        if (chromosom[i] < this->m_Min[i]) {
            double violation = this->m_Min[i] - chromosom[i];
            penalty += std::pow(violation, m_b); 
        } else if (chromosom[i] > this->m_Max[i]) {
            double violation = chromosom[i] - this->m_Max[i];
            penalty += std::pow(violation, m_b); 
        }
    }

    double dynamicFactor = m_c * std::pow(static_cast<double>(generation), m_a); 
    penalty *= dynamicFactor;

    fitnessValue += penalty; 
}

