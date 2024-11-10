#include "Selection.h"
#include "random"

/*
Stochastic Universal Sampling (SUS) selection, method for selecting individuals based on fitness proportion
*/
class SelectionSUS: public Selection<double>
{
    private:
    // Helper method to calculate the sum of fitness values
    double sum(const std::vector<double> &);

    public: 
    /*
    argument- vector of fitness 
    return- vector of best selected values
    */
    std::vector<size_t> selection(const std::vector<double> &) override;
};