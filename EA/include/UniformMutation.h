#include "Mutation.h"

class UniformMutation: public Mutation<double>
{
    private:
        std::uniform_real_distribution<double> m_distribution;

    public:
        std::vector<double> mutation(std::vector<double> &);
        UniformMutation(double Pm, std::vector<double>  min, std::vector<double> max):
            Mutation(max, min,Pm), m_distribution(0.0, 1.0){}
};