#include "Mutation.h"

// size_t maxgen, double b, double Pm, std::vector<double>  min, std::vector<double> max, size_t gen
class NonuniformMutation: public Mutation<double>
{
    private:
        double m_b;//parameter to control the annealing speed
        size_t m_gen;
        size_t m_maxgen;


    public:
        void mutation(std::vector<double> &) override;
        double extension(double y);
        NonuniformMutation(size_t maxgen,double b,double Pm, std::vector<double>  min, std::vector<double> max, size_t gen):
           m_gen(gen), Mutation( max, min, Pm), m_b(b), m_maxgen(maxgen){}
};