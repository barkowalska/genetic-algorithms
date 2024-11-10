#include "Mutation.h"

// double pm, std::vector<double>  min, std::vector<double> max
class BoundryMutation: public Mutation<double>
{
    
    public:
    void mutation(std::vector<double> &) override;
    BoundryMutation(double pm, std::vector<double>  min, std::vector<double> max): 
        Mutation(max, min, pm){}
};