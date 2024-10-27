#include "Mutation.h"


class CauchyMutation: public Mutation<double>
{
    private:
        double m_sigma;
        std::cauchy_distribution<double> m_cauchydistribution;

    public:
        std::vector<double> mutation(std::vector<double> &);
        CauchyMutation(double sigma, double Pm, std::vector<double>  min, std::vector<double> max):
            Mutation(max, min,Pm), m_cauchydistribution(0.0, 1.0){}
};