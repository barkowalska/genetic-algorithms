#pragma once

#include "./Crossover/Crossover.h"
#include "./Mutation/Mutation.h"
#include "./Scaling/Scaling.h"
#include "./Selection/Selection.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>
#include <omp.h>

template <typename ChromosomeType> class GA {
private:
  std::mt19937 m_generator;

  std::shared_ptr<Scaling<double>> m_scaling;
  std::shared_ptr<Crossover<ChromosomeType>> m_crossover;
  std::shared_ptr<Mutation<ChromosomeType>> m_mutation;
  std::shared_ptr<Selection<double>> m_selection;

  std::vector<double> m_min;
  std::vector<double> m_max;

  // przesylam pojedynczego individual chromosom, zwraca pojedynczego
  // fitnessValue
  std::function<double(const std::vector<ChromosomeType> &)> m_fitness_function;

  size_t m_maxGen;
  size_t m_popSize;
  double m_Pc;
  size_t m_lengthOfChromosome;

  // jakis minimalny zakresdla ktorego individuale nie beda sie juz od siebie
  // roznily
  double m_epsilon;
  // n elemntow dla ktporych max moze byc zbyt mala roznica
  size_t m_numMaxElements;

public:
  GA() : m_generator(std::random_device{}()) {}
  struct Population {
    std::vector<std::vector<ChromosomeType>> chromosomes;
    std::vector<double> fitnessValues;
  } m_population;

  inline void set_m_scaling(std::shared_ptr<Scaling<double>> scaling) {
    m_scaling = scaling;
  }
  inline void
  set_m_crossover(std::shared_ptr<Crossover<ChromosomeType>> crossover) {
    m_crossover = crossover;
  }
  inline void
  set_m_mutation(std::shared_ptr<Mutation<ChromosomeType>> mutation) {
    m_mutation = mutation;
  }
  inline void set_m_selection(std::shared_ptr<Selection<double>> selection) {
    m_selection = selection;
  }
  inline void set_m_fitness_function(
      std::function<double(const std::vector<ChromosomeType> &)>
          fitness_function) {
    m_fitness_function = fitness_function;
  }
  inline void set_m_maxGen(size_t maxGen) { m_maxGen = maxGen; }
  inline void set_m_popSize(size_t popSize) { m_popSize = popSize; }
  inline void set_m_epsilon(double e) { m_epsilon = e; }
  inline void set_m_Pc(double pc) { m_Pc = pc; }
  inline void set_m_numMaxElements(size_t numMaxElements) {
    m_numMaxElements = numMaxElements;
  }
  inline void set_m_lengthOfChromosome(size_t lengthOfChromosome) {
    m_lengthOfChromosome = lengthOfChromosome;
  }
  inline void set_m_min(std::vector<double> min) { m_min = min; }
  inline void set_m_max(std::vector<double> max) { m_max = max; }

  std::pair<double, std::vector<ChromosomeType>> run();
  void initialize();
};

template <typename ChromosomeType> void GA<ChromosomeType>::initialize() {
  if (m_min.size() != m_lengthOfChromosome ||
      m_max.size() != m_lengthOfChromosome) {
    throw std::invalid_argument(
        "Rozmiar wektorów m_min i m_max musi być równy m_lengthOfChromosom.");
  }

  std::vector<std::uniform_real_distribution<double>> dist;
  for (size_t i = 0; i < m_lengthOfChromosome; ++i) {
    dist.emplace_back(m_min[i], m_max[i]);
  }
  m_population.chromosomes.resize(m_popSize);
  m_population.fitnessValues.resize(m_popSize);

  for (size_t i = 0; i < m_popSize; i++) {
    m_population.chromosomes[i].resize(m_lengthOfChromosome);
    for (size_t j = 0; j < m_lengthOfChromosome; j++) {
      m_population.chromosomes[i][j] = dist[j](m_generator);
    }
    m_population.fitnessValues[i] =
        m_fitness_function(m_population.chromosomes[i]);
  }
}

template <typename ChromosomeType>
std::pair<double, std::vector<ChromosomeType>> GA<ChromosomeType>::run() {
  size_t counterMaxElements = 0;

  std::uniform_real_distribution<double> randomPc(0.0, 1.0);

  Population matingPool{std::vector<std::vector<ChromosomeType>>(m_popSize),
                        std::vector<double>(m_popSize)};
  std::reference_wrapper<Population> actualPopulation = m_population;
  std::reference_wrapper<Population> actualMatingPool = matingPool;

  size_t numParents = m_crossover->getNumberOfParents();
  if (numParents == 0) {
    throw std::runtime_error("Liczba rodziców nie może być zerowa.");
  }

  // how many elements are missing to fill toCross apart from the rest
  size_t remainder = m_popSize % numParents;
  size_t rest = (remainder == 0) ? 0 : numParents - remainder;
  std::optional<std::uniform_int_distribution<size_t>> fillCross;

  if (actualPopulation.get().fitnessValues.empty())
    throw std::runtime_error("Błąd: Wektor fitnessValues jest pusty!");
  double maxPreviouse =
      *std::max_element(actualPopulation.get().fitnessValues.begin(),
                        actualPopulation.get().fitnessValues.end());
  double maxCurrent = maxPreviouse;

  if (rest != 0) {
    fillCross = std::uniform_int_distribution<size_t>(
        0, m_popSize - 1 - (m_popSize % numParents));
  }

  // Zamienia indeksy wybranych osobników na referencje do chromosomów
  auto selectedToReference = [actualPopulation](const size_t &chromosomeIndex)
      -> std::reference_wrapper<std::vector<ChromosomeType>> {
    return std::ref(actualPopulation.get().chromosomes[chromosomeIndex]);
  };

  // oblicza fitnessValue dla podanego chromosomu, finalnie zwroci najwieksza
  // wartosc fitnessValue z populacji
  auto evaluateFitnessValues =
      [this](const std::vector<ChromosomeType> &chromosome) {
        double value = m_fitness_function(chromosome);
        if(std::isnan(value)){
            throw std::runtime_error("NAN value!");
        }
        

        return value;
      };

  size_t gen = 0;

  try {
    m_scaling->scaling(actualMatingPool.get().fitnessValues);

    for (gen = 0; gen < m_maxGen; gen++) {

      // selection

      auto selected =
          m_selection->selection(actualPopulation.get().fitnessValues);

      // crossover
      size_t numOfCross = m_popSize / numParents;

      #pragma omp parallel for schedule(static) default(shared)
      for (size_t i = 0; i < numOfCross; i++) {
        if (randomPc(m_generator) > m_Pc) {
          for (size_t idx = i * numParents; idx < (i + 1) * numParents; idx++) {
            actualMatingPool.get().chromosomes[idx] =
                actualPopulation.get().chromosomes[selected[idx]];
          }
          continue;
        }
        std::vector<std::reference_wrapper<const std::vector<ChromosomeType>>>
            toCross;
        std::transform(selected.begin() + i * numParents,
                       selected.begin() + (i + 1) * numParents,
                       std::back_inserter(toCross), selectedToReference);

        if (toCross.size() != numParents) {
          throw std::runtime_error("Incorrect number of parents in toCross.");
        }

        auto offsprings = m_crossover->cross(toCross);

        // Calculate starting index for offsprings
        size_t offspringStartIndex = i * offsprings.size();
        if (offspringStartIndex + offsprings.size() >
            actualMatingPool.get().chromosomes.size()) {
          throw std::runtime_error(
              "Index out of bounds when copying offsprings.");
        }

        std::copy(offsprings.begin(), offsprings.end(),
                  actualMatingPool.get().chromosomes.begin() +
                      offspringStartIndex);
      }

      // Handling remaining individuals
      if (rest != 0) {
        std::vector<std::reference_wrapper<const std::vector<ChromosomeType>>>
            toCross;
        std::transform(selected.begin() + numOfCross * numParents,
                       selected.end(), std::back_inserter(toCross),
                       selectedToReference);

        // Randomly select additional parents to fill toCross
        std::uniform_int_distribution<size_t> fillCross(0, m_popSize - 1);
        while (toCross.size() < numParents) {
          size_t randomIndex = fillCross(m_generator);
          toCross.push_back(selectedToReference(randomIndex));
        }

        if (toCross.size() != numParents) {
          throw std::runtime_error("Incorrect number of parents in toCross.");
        }

        auto offsprings = m_crossover->cross(toCross);

        // Calculate starting index for offsprings
        size_t offspringStartIndex = numOfCross * offsprings.size();

        std::copy(offsprings.begin(), offsprings.begin() + remainder,
                  actualMatingPool.get().chromosomes.begin() +
                      offspringStartIndex);
      }

      // mutation
      #pragma omp parallel for schedule(static)
      for(size_t idx = 0; idx < m_popSize; idx++){
        m_mutation->mutation(actualMatingPool.get().chromosomes[idx]);
      }

      // skalowanie
      // obliczanie fitness value+ wybranie nowego maxCurrent z gen
      maxCurrent = actualMatingPool.get().fitnessValues.front();
      std::transform(actualMatingPool.get().chromosomes.begin(),
                     actualMatingPool.get().chromosomes.end(),
                     actualMatingPool.get().fitnessValues.begin(),
                     evaluateFitnessValues);
      maxCurrent = *std::max_element(actualMatingPool.get().fitnessValues.begin(), actualMatingPool.get().fitnessValues.end());
      //m_scaling->scaling(actualMatingPool.get().fitnessValues);
      // nowa populacja
      std::swap(actualMatingPool, actualPopulation);

      if ((maxPreviouse != 0.0
               ? (fabs(maxCurrent - maxPreviouse) / maxPreviouse)
               : fabs(maxCurrent - maxPreviouse)) < m_epsilon) {
        counterMaxElements++;
        if (counterMaxElements > m_numMaxElements) {
          break;
        }
      } else {
        counterMaxElements = 0;
      }
      maxPreviouse = maxCurrent;
    }

  } catch (std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    return std::make_pair(0.0, std::vector<ChromosomeType>());
  }
  auto bestValue =
      std::max_element(actualPopulation.get().fitnessValues.begin(),
                       actualPopulation.get().fitnessValues.end());
  size_t bestIndeks =
      std::distance(actualPopulation.get().fitnessValues.begin(), bestValue);

  std::pair<double, std::vector<ChromosomeType>> bestIndividual{
      actualPopulation.get().fitnessValues[bestIndeks],
      actualPopulation.get().chromosomes[bestIndeks]};
  return bestIndividual;
}
