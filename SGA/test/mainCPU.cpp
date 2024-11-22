#include <SGA.h>
#include <array>
#include <cmath>
#include <iostream>

#define LEN 8

using namespace std;

double funkcja(double x) { return -fabs(-x * x * sin(10.0 * x) - x * cos(x)); }

int main() {
  float pc = 0.9;
  float pm = 0.02;
  int popsize = 10;
  int maxgen = 100;
  int min = -10;
  int max = 10;
  int l_genes = 60;

  SGA sga(funkcja, pc, pm, popsize, maxgen, max, min);
  sga.initialize();
  SGA::Individual best = sga.run();

  std::cout << "X: " << best.phenotype << " | Value: " << best.quality
            << std::endl;
}

// OMP

/*
   1 tworze objekt w mainie
   2. sga.inicjalize tu od razu fitness value
   3. wchodzimy do loop
       3.1 selekcja
       3.2 tworzymy mating loop ktory selekcja parami
       3.3 selekcja z selekcji (wyciaganie pary) BEZ ZWRACANIA
       3.4 podanie do funkcji crossover-> gdzie cross-ovver zapisuje nowe?
       3.5 funkcja mutuwoania- NA CALYM MATING POOL Z PM
       3.6 obliczyc nowe quality
       3.7 nowa generacja teraz jest bazowa
       3.8 od nowa 3.1
   */
