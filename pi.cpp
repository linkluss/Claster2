#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

double compute_pi(long num_samples, int rank) {
    long inside_circle = 0;
    unsigned int seed = time(nullptr) + rank;  // Seed based on rank to ensure different random numbers

    for (long i = 0; i < num_samples; ++i) {
        double x = static_cast<double>(rand_r(&seed)) / RAND_MAX;
        double y = static_cast<double>(rand_r(&seed)) / RAND_MAX;

        if (x * x + y * y <= 1.0) {
            ++inside_circle;
        }
    }

    return 4.0 * inside_circle / num_samples;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    long num_samples = 1000000; // Default number of samples
    if (argc > 1) {
        num_samples = atol(argv[1]);
    }

    // Divide the number of samples among all processes
    long samples_per_process = num_samples / world_size;

    // Each process computes its part of PI
    double local_pi = compute_pi(samples_per_process, world_rank);

    // Gather all local_pi values to the root process
    double global_pi = 0.0;
    MPI_Reduce(&local_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        // Average the sum of PI estimates
        global_pi /= world_size;
        std::cout << "Estimated value of PI = " << global_pi << std::endl;
    }

    MPI_Finalize();
    return 0;
}
