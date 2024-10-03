#include <vector>
#include<functional>
class SGA
{
public:
    struct Individual
    {
        double quality{0};
        std::vector<bool> location;
        double phenotype{0};
        double pi{0};
    };
    
    SGA(std::function<double(double)>fitnessFunction, float pc, float pm, int popsize, int maxgen, int max, int min);
    void initialize();
    void setpi();
    std::vector<Individual> selectionRWS(); 
    double binary_decoding(const std::vector<bool>& location);
    Individual run();
    void cross_over( std::vector<bool>& locationA,  std::vector<bool> &locationB);
    Individual construct_individual(std::vector<bool>& location);
    void random_shuffle();
    void mutation(std::vector<bool>& location);

    double uniform_random(double min, double max);

private:
    std::vector<Individual>m_population;
    float m_Pc;//crossover rate
    float m_Pm=0;//mutation rate
    int m_popsize=10;
    int m_maxgen=10;
    std::function<double(double)> m_fitness_function;
    std::vector<Individual>m_mating_pool;
    double m_max=100;
    double m_min=0;
    int m_l_genes=10;

    
    //fitness_funkcja

};