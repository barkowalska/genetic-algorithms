#include "Mutation.h"

class NonuniformMutation: public Mutation<double>
{
    private:
        double m_b;//parameter to control the annealing speed
        std::uniform_real_distribution<double> m_distribution;
        size_t m_gen;

    public:
        std::vector<double> mutation(std::vector<double> &);
        double extension(double y);
        NonuniformMutation(double b,double Pm, std::vector<double>  min, std::vector<double> max, size_t gen):
           m_gen(gen), Mutation( max, min, Pm), m_b(b), m_distribution(0.0, 1.0){}
};