#include "Selection.h"
#include <vector>
#include <random>
class TournamentSelection: public Selection<double>
{
    private:
    int m_tournamentSize;
    public:
    std::vector<size_t> selection(const std::vector<double> &) override;
    TournamentSelection(int k): 
        m_tournamentSize(k){}
};
