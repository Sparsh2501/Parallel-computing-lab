#include <iostream>
#include <vector>
#include <mpi.h>
#include <cstdlib>
#include <ctime>

using namespace std;

void serial_daxpy(double a, vector<double> &X, vector<double> &Y) {
    for (size_t i = 0; i < X.size(); i++) {
        X[i] = a * X[i] + Y[i];
    }
}

void parallel_daxpy(double a, vector<double> &local_X, vector<double> &local_Y) {
    for (size_t i = 0; i < local_X.size(); i++) {
        local_X[i] = a * local_X[i] + local_Y[i];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    const int N = 1 << 16;
    double a = 2.5;
    vector<double> X, Y, serial_X, serial_Y;
    
    if (rank == 0) {
        srand(time(NULL));
        X.resize(N);
        Y.resize(N);
        serial_X.resize(N);
        serial_Y.resize(N);
        for (int i = 0; i < N; i++) {
            X[i] = serial_X[i] = rand() % 100 / 10.0;
            Y[i] = serial_Y[i] = rand() % 100 / 10.0;
        }
    }
    
    double serial_time = 0.0;
    if (rank == 0) {
        double start_serial = MPI_Wtime();
        serial_daxpy(a, serial_X, serial_Y);
        double end_serial = MPI_Wtime();
        serial_time = end_serial - start_serial;
    }
    
    int local_size = N / world_size;
    vector<double> local_X(local_size), local_Y(local_size);
    MPI_Scatter(X.data(), local_size, MPI_DOUBLE, local_X.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(Y.data(), local_size, MPI_DOUBLE, local_Y.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_parallel = MPI_Wtime();
    parallel_daxpy(a, local_X, local_Y);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_parallel = MPI_Wtime();
    double parallel_time = end_parallel - start_parallel;
    
    MPI_Gather(local_X.data(), local_size, MPI_DOUBLE, X.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double max_parallel_time;
    MPI_Reduce(&parallel_time, &max_parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "Serial Execution Time: " << serial_time << " seconds" << endl;
        cout << "Parallel Execution Time: " << max_parallel_time << " seconds" << endl;
        cout << "Speedup: " << serial_time / max_parallel_time << endl;
    }
    
    MPI_Finalize();
    return 0;
}