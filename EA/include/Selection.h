#include <vector>

template<typename T>
class Selection{

protected:
std::mt19937 m_generator;

public:
virtual std::vector<std::vector<T>> selection(const std::vector<std::vector<T>> &)=0;
Selection(): m_generator(std::random_device{}()){}
};
