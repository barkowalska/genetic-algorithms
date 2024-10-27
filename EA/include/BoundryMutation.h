#include "Mutation.h"

class BoundryMutation: public Mutation<double>
{
    private:
    std::uniform_real_distribution<double> m_distribution;
    
    public:
    std::vector<double> mutation(std::vector<double> &);
    BoundryMutation(double pm, std::vector<double>  min, std::vector<double> max): 
        Mutation(max, min, pm),m_distribution(0.0, 1.0){}
};