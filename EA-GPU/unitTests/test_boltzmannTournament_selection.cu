// unitTests/test_boltzmann_tournament_selection.cu

#include <gtest/gtest.h>
#include "CEA.cuh"
#include "Selection/BoltzmannTournamentSelection.cuh"
#include <cuda_runtime.h>

using namespace cea;

// Constants for testing
const uint64_t PopSize = 8;
const uint64_t ChromosomeSize = 5;
const uint64_t TournamentSize = 3;
const double Temperature = 1.0; // Adjust as needed

// Device fitness function (simple sum of chromosome elements)
__device__ double fitnessFunction(double* chromosome) {
    double fitness = 0.0;
    for (uint64_t i = 0; i < ChromosomeSize; ++i) {
        fitness += chromosome[i];
    }
    return fitness;
}

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in file '%s' at line %d: %s.\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


class BoltzmannTournamentSelectionTest : public ::testing::Test {
protected:
    PopulationType<PopSize, ChromosomeSize>* d_population = nullptr;
    uint64_t* d_selected = nullptr;

    void SetUp() override {
        // Initialize host population with known fitness values
        PopulationType<PopSize, ChromosomeSize> h_population;
        for (uint64_t i = 0; i < PopSize; ++i) {
            // Initialize chromosomes with known values
            for (uint64_t j = 0; j < ChromosomeSize; ++j) {
                h_population.chromosomes[i * ChromosomeSize + j] = static_cast<double>(i + j);
            }
            // Precompute fitness values on host
            h_population.fitnessValue[i] = 0.0;
            for (uint64_t j = 0; j < ChromosomeSize; ++j) {
                h_population.fitnessValue[i] += h_population.chromosomes[i * ChromosomeSize + j];
            }
        }

        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_population, sizeof(PopulationType<PopSize, ChromosomeSize>)));
        CUDA_CHECK(cudaMalloc(&d_selected, PopSize * sizeof(uint64_t)));

        // Copy population to device
        CUDA_CHECK(cudaMemcpy(d_population, &h_population, sizeof(PopulationType<PopSize, ChromosomeSize>), cudaMemcpyHostToDevice));

        // Set the fitness function on the device
        fitnessFunction_ptr h_fitnessFunction;
        CUDA_CHECK(cudaMemcpyFromSymbol(&h_fitnessFunction, fitnessFunction, sizeof(fitnessFunction_ptr)));
        CUDA_CHECK(cudaMemcpyToSymbol(FitnessFunction, &h_fitnessFunction, sizeof(fitnessFunction_ptr)));
    }

    void TearDown() override {
        // Free device memory
        if (d_population) cudaFree(d_population);
        if (d_selected) cudaFree(d_selected);
    }
};

TEST_F(BoltzmannTournamentSelectionTest, SelectionWorksCorrectly) {
    // Set a fixed seed for reproducibility
    setGlobalSeed();

    // Create instance of BoltzmannTournamentSelection
    dim3 blockSize(PopSize);
    BoltzmannTournamentSelection<PopSize, ChromosomeSize> selection(blockSize, TournamentSize, Temperature);

    // Run the selection operator
    selection(d_population, d_selected);

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy selected indices back to host
    uint64_t h_selected[PopSize];
    CUDA_CHECK(cudaMemcpy(h_selected, d_selected, PopSize * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Copy population back to host for validation
    PopulationType<PopSize, ChromosomeSize> h_population;
    CUDA_CHECK(cudaMemcpy(&h_population, d_population, sizeof(PopulationType<PopSize, ChromosomeSize>), cudaMemcpyDeviceToHost));

    // Verify the results
    for (uint64_t i = 0; i < PopSize; ++i) {
        uint64_t selectedIdx = h_selected[i];
        // The selected index should be within [0, PopSize)
        ASSERT_LT(selectedIdx, PopSize);

        // The selected fitness should be valid
        double selectedFitness = h_population.fitnessValue[selectedIdx];
        ASSERT_GE(selectedFitness, 0.0);

        // Optional: Print the selection results
        std::cout << "Thread " << i << " selected individual " << selectedIdx << " with fitness " << selectedFitness << std::endl;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // Initialize CUDA device
    cudaSetDevice(0);
    return RUN_ALL_TESTS();
}
