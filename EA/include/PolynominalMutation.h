#include "Mutation.h"

class PolynominalMutation: public Mutation<double>
{
    private:
    double m_n;//control parameter


    public:
        void mutation(std::vector<double> &) override;

        PolynominalMutation(double n,double Pm, std::vector<double>  min, std::vector<double> max, size_t gen):
            m_n(n), Mutation( max, min, Pm){}
};