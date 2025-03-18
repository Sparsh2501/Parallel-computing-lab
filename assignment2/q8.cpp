#include <iostream>
#include <vector>
#include <mpi.h>
#include <cstdlib>
#include <ctime>

using namespace std;

void serial_transpose(vector<int> &matrix, vector<int> &result, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[j * N + i] = matrix[i * N + j];
        }
    }
}

void parallel_transpose(vector<int> &local_matrix, vector<int> &local_result, int N, int rank, int world_size) {
    int rows_per_process = N / world_size;
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            local_result[j * rows_per_process + i] = local_matrix[i * N + j];
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (argc < 2) {
        if (rank == 0) cout << "Usage: ./a.out <matrix_size>" << endl;
        MPI_Finalize();
        return -1;
    }
    
    int N = atoi(argv[1]);
    vector<int> matrix, serial_result(N * N);
    
    if (rank == 0) {
        srand(time(NULL));
        matrix.resize(N * N);
        for (int i = 0; i < N * N; i++) {
            matrix[i] = rand() % 100;
        }
    }
    
    double serial_time = 0.0;
    if (rank == 0) {
        double start_serial = MPI_Wtime();
        serial_transpose(matrix, serial_result, N);
        double end_serial = MPI_Wtime();
        serial_time = end_serial - start_serial;
    }
    
    int rows_per_process = N / world_size;
    vector<int> local_matrix(rows_per_process * N), local_result(rows_per_process * N);
    MPI_Scatter(matrix.data(), rows_per_process * N, MPI_INT, local_matrix.data(), rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_parallel = MPI_Wtime();
    parallel_transpose(local_matrix, local_result, N, rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_parallel = MPI_Wtime();
    double parallel_time = end_parallel - start_parallel;
    
    vector<int> parallel_result(N * N);
    MPI_Gather(local_result.data(), rows_per_process * N, MPI_INT, parallel_result.data(), rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);
    
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
