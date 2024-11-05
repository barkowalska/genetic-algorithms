#include "Selection.h"
#include "random"

class Sus: public Selection<double>
{
private:
double sum(const std::vector<double> &);


public: 
std::vector<size_t> selection(const std::vector<double> &);


};