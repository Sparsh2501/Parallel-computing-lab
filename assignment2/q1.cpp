#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>

void monte_carlo(long num_points, long &insideCircle) {
    srand(time(NULL));
    for (long i = 0; i < num_points; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;

        if (x * x + y * y <= 1) {
            insideCircle++;
        }
    }
}

void monte_carlo_parallel(long num_points, long &insideCircle, int rank, int world_size) {
    srand(time(NULL) * rank);
    for (long i = rank; i < num_points; i += world_size) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;

        if (x * x + y * y <= 1) {
            insideCircle++;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    if (argc < 2) {
        std::cerr << "NOT ENOUGH ARGUMENTS\n";
        MPI_Finalize();
        return -1;
    }

    long num_points = atol(argv[1]);
    long insideCircleLocal = 0;
    long insideCircle;

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double serial_time = 0.0;
    if (rank == 0) {
        long serialInsideCircle = 0;
        double serial_start = MPI_Wtime();
        monte_carlo(num_points, serialInsideCircle);
        double serial_end = MPI_Wtime();
        serial_time = serial_end - serial_start;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    monte_carlo_parallel(num_points, insideCircleLocal, rank, world_size);

    MPI_Reduce(&insideCircleLocal, &insideCircle, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double parallel_time = end_time - start_time;

    double max_parallel_time;
    MPI_Reduce(&parallel_time, &max_parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Estimated Pi: " << 4.0 * (double)insideCircle / num_points << std::endl;
        std::cout << "Serial Execution Time: " << serial_time << " seconds" << std::endl;
        std::cout << "Parallel Execution Time: " << max_parallel_time << " seconds" << std::endl;
        std::cout << "Speedup: " << serial_time / max_parallel_time << std::endl;
    }

    MPI_Finalize();
    return 0;
}
