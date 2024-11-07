#include "Selection.h"
#include "random"

class SelectionSUS: public Selection<double>
{
    private:
    double sum(const std::vector<double> &);

    public: 
    std::vector<size_t> selection(const std::vector<double> &) override;
};