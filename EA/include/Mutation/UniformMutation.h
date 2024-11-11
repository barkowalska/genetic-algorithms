#include "Mutation.h"

// double Pm, std::vector<double>  min, std::vector<double> max
class UniformMutation : public Mutation<double> {

public:
  void mutation(std::vector<double> &) override;
  UniformMutation(double Pm, std::vector<double> min, std::vector<double> max)
      : Mutation(max, min, Pm) {}
};