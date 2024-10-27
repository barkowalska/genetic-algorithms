#include <vector>
#include <random>

template<typename T>
class Mutation
{
    protected:
        std::mt19937 m_generator;
        std::vector<double> m_Min;//lower limit of the range
        std::vector<double> m_Max;//upper limit of the range
        double m_Pm;

    public:

    //the main function of the mutation
    virtual std::vector<T> mutation(std::vector<T> &)=0;
    Mutation(std::vector<double> max, std::vector<double> min, double pm): 
       m_Max(max), m_Min(min), m_generator(std::random_device{}()), m_Pm(pm) { }//moge tutaj zapisać np ze jak min>max to zwroc blad
};