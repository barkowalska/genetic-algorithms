#include "Crossover/ArithmeticCrossover.cuh"
#include "Crossover/BlendCrossover.cuh"
#include "Crossover/MultiplePointCrossover.cuh"
#include "Crossover/SimulatedBinaryCrossover.cuh"
#include "Crossover/UniformCrossover.cuh"

#include "Selection/BoltzmannTournamentSelection.cuh"
#include "Selection/SelectionSUS.cuh"
#include "Selection/TournamentSelection.cuh"

#include "Mutation/BoundryMutation.cuh"
#include "Mutation/CauchyMutation.cuh"
#include "Mutation/NonuniformMutation.cuh"
#include "Mutation/PolynominalMutation.cuh"
#include "Mutation/UniformMutation.cuh"

#include "Migration/Migration.cuh"

#include "Initialization/Initialization.cuh"

#include "Scaling/BoltzmannScaling.cuh"
//#include "Scaling/FitnessTransferral.cuh"
#include "Scaling/PowerLawScaling.cuh"

#include "PenaltyFunction/DynamicPenaltyFunction.cuh"