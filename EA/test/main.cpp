
#include "../include/GeneticAlgorithms.h"
#include <cmath>
#include <vector>

int main() {
  // init
  // funkcja celu?
  size_t popsize = 1000;
  size_t maxgen = 1000;

  double Pc = 0.9;
  double Pm = 0.05;
  double epsilon = 1e-6;
  size_t m_numMaxElements = 10;

  size_t lengthOfChromosome = 2;
  std::vector<double> min(lengthOfChromosome, -10.0);
  std::vector<double> max(lengthOfChromosome, 10.0);

  std::shared_ptr<Scaling<double>> scaling = std::make_shared<LinearScaling>();
  std::shared_ptr<Crossover<double>> crossover =
      std::make_shared<ArithmeticCrossover>();
  std::shared_ptr<Mutation<double>> mutation =
      std::make_shared<CauchyMutation>(1.0, Pm, min, max);
  std::shared_ptr<Selection<double>> selection =
      std::make_shared<TournamentSelection>(10);

  GA<double> ga;
  ga.set_m_crossover(crossover);
  ga.set_m_mutation(mutation);
  ga.set_m_selection(selection);
  ga.set_m_popSize(popsize);
  ga.set_m_maxGen(maxgen);
  ga.set_m_Pc(Pc);
  ga.set_m_epsilon(epsilon);
  ga.set_m_numMaxElements(m_numMaxElements);
  ga.set_m_lengthOfChromosome(lengthOfChromosome);
  ga.set_m_min(min);
  ga.set_m_max(max);
  ga.set_m_scaling(scaling);

  auto fitness_function = [](const std::vector<double> &chromosome) -> double {
    if(chromosome.size() != 2){
        throw std::runtime_error("Wrong number of arguments!!");
    }
    double x = chromosome[0];
    double y = chromosome[1];

    // Calculate the fitness using the Schaffer function N. 4
    double numerator = pow(cos(sin(fabs(x * x - y * y))), 2) - 0.5;
    double denominator = pow(1 + 0.001 * (x * x + y * y), 2);
    if (denominator == 0.0) {
      throw std::runtime_error("Dividing by zero!");
    }
    double fitness = 0.5 + numerator / denominator;

    return -fitness;
  };
  // Set the Schaffer function N. 4 as the fitness function
  ga.set_m_fitness_function(fitness_function);

  ga.initialize();
  std::pair<double, std::vector<double>> bestChromosome = ga.run();

  std::cout<<std::endl;
  for(int i = 0; i < bestChromosome.second.size(); i++){
    std::cout << bestChromosome.second[i] << ' ';
  }
    std::cout<<std::endl;
  return 0;
}