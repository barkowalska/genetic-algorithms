#include "Mutation.h"

class PolynominalMutation: public Mutation<double>
{
    private:
    double m_n;//control parameter
        std::uniform_real_distribution<double> m_distribution;

    public:
        std::vector<double> mutation(std::vector<double> &);

        PolynominalMutation(double n,double Pm, std::vector<double>  min, std::vector<double> max, size_t gen):
            m_n(n), Mutation( max, min, Pm), m_distribution(0.0, 1.0){}
};