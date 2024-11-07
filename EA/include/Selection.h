#include <vector>
#include <random>


template<typename T>
class Selection{

protected:
    std::mt19937 m_generator;

public:
    //zwraca vector z numerami osobnikow wybranych ale to rzeba bedzie jeszcze 
    virtual std::vector<size_t> selection(const std::vector<T> &fitnessValue)=0;
    Selection(): 
        m_generator(std::random_device{}()){}
};
