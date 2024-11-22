#pragma once

#include <vector>
#include <functional>

template <typename T>
class PenaltyFunction {
protected:
    size_t m_gen;
    std::vector<double> m_Min; // lower limit of the range
    std::vector<double> m_Max; // upper limit of the range
public:
    virtual ~PenaltyFunction() = default;
    PenaltyFunction(std::vector<double> max, std::vector<double> min, size_t gen)
      : m_Max(max), m_Min(min), m_gen(gen) {}


    // Pure virtual method to implement penalty calculation
    virtual void penaltyFunction(size_t generation, double& fitnessValue, std::vector<T>& chromosom) = 0;
};
