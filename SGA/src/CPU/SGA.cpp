#include "SGA.h"
#include <random>
#include <math.h>
#include <iostream>
#include <algorithm>


SGA::SGA(std::function<double(double)>fitnessFunction, float pc, float pm, int popsize, int maxgen, int max, int min)
{
    this->m_fitness_function=fitnessFunction;
    this->m_Pc=pc;
    this->m_Pm=pm;
    this->m_popsize=popsize; 
    this->m_maxgen=maxgen;
    this->m_max=max; 
    this->m_min=min;
    Individual a;
    a.location.reserve(m_l_genes);
    m_population.reserve(popsize);
    m_mating_pool.reserve(popsize);

}


void SGA::initialize()
{
    for (int i = 0; i < m_popsize; i++) {
        std::vector<bool> chromosome;
        for (int j = 0; j < m_l_genes; j++) 
        {
            float prob=uniform_random(0.0,1.0);
            chromosome.push_back(prob>0.5);
        }
      
        m_population.push_back(construct_individual(chromosome));
    }
    
}

SGA::Individual SGA::construct_individual(std::vector<bool>& location)
{
    Individual tmp;
    tmp.location=location;
    tmp.phenotype=binary_decoding(location);
    tmp.quality=m_fitness_function(tmp.phenotype);
    return tmp;
}



void SGA::setpi()
{

    double sum_quality;
    for(int i=0; i< m_population.size(); i++)
    {
        sum_quality+=m_population[i].quality;
    }

    for(int i=0; i< m_popsize; i++)
    {
        m_population[i].pi=m_population[i].quality/sum_quality;
    }
}

double SGA::binary_decoding(const std::vector<bool>& location)
{
    int sum=0;
    for(int i=0; i<location.size();i++)
    {
        sum+=location[i]*(pow(2,i));
    }
    double roznica=m_max-m_min;
    double phenotype=static_cast<double>(sum)/static_cast<double>(pow(2,location.size()));
    phenotype=phenotype*roznica;
    phenotype+=m_min;
    return phenotype;
}
void SGA::cross_over( std::vector<bool>& locationA,  std::vector<bool>& locationB)
{
    if(uniform_random(0.0,1.0)<m_Pc)
    {
        int point=uniform_random(0.0,locationA.size());
        for(int i=point; i<locationA.size(); i++)
        {
            swap(locationA[i], locationB[i]);
        }   
    }
}

std::vector<SGA::Individual> SGA::selectionRWS()
{
    std::vector<double> rws;
    setpi();

    double sum;
    while(m_mating_pool.size()<m_popsize)
    {
        double rand=uniform_random(0.0, 2*M_PI);
        for(int i=0; i<m_popsize; i++)
        {
            sum+=2*M_PI*m_population[i].pi;
            //rws.push_back(sum);
            //if(rand<rws[i])
            if(rand<sum)
            {
                m_mating_pool.push_back(m_population[i]);
                break;
            }
        }
    }
    return m_mating_pool;
    
}
SGA::Individual SGA::run()
{
    for(int gen=0; gen<m_maxgen; gen++)
    {
        m_mating_pool=selectionRWS();
        random_shuffle();
        for(int i=0; i<m_mating_pool.size()-1; i+=2)
        {
            cross_over(m_mating_pool[i].location, m_mating_pool[i+1].location);
            mutation(m_mating_pool[i].location);
            mutation(m_mating_pool[i+1].location);  
            m_population[i]=construct_individual(m_mating_pool[i].location);   
            m_population[i+1]=construct_individual(m_mating_pool[i+1].location);   
        }
        if(m_popsize%2==1) 
        {
            mutation(m_mating_pool.back().location);  
            m_population.back()=construct_individual(m_mating_pool.back().location);   
        }

    }
    Individual best=m_population.front();
    for(Individual k: m_population)
    {
        if(k.quality>best.quality) best=k;
    }
    return best;
}

void SGA::random_shuffle()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::shuffle(m_mating_pool.begin(), m_mating_pool.end(), gen);

}

void SGA::mutation(std::vector<bool>& location)
{
    for(bool j:location)
    {
        double pm=uniform_random(0.0,1.0);
        if(pm<m_Pm)   j = !j;
    }

}

double SGA::uniform_random(double min, double max)
{
    static std::random_device rd;   // Urządzenie losowe (niezależne źródło entropii)
    static std::mt19937 gen(rd());  // Generator bazujący na Mersenne Twister
    static std::uniform_real_distribution<> dis(0.0, 1.0); //
    return dis(gen)*(max-min) + min;
}