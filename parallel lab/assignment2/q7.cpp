#include <iostream>
#include <vector>
#include <mpi.h>
#include <cstdlib>
#include <ctime>

using namespace std;

void serial_prefix_sum(vector<int> &arr, vector<int> &serial_result) {
    serial_result[0] = arr[0];
    for (size_t i = 1; i < arr.size(); i++) {
        serial_result[i] = serial_result[i - 1] + arr[i];
    }
}

void parallel_prefix_sum(vector<int> &local_arr, vector<int> &local_result, int rank, int world_size) {
    int local_size = local_arr.size();
    local_result[0] = local_arr[0];
    for (int i = 1; i < local_size; i++) {
        local_result[i] = local_result[i - 1] + local_arr[i];
    }
    int offset = 0;
    MPI_Exscan(&local_result[local_size - 1], &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank > 0) {
        for (int i = 0; i < local_size; i++) {
            local_result[i] += offset;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (argc < 2) {
        if (rank == 0) cout << "Usage: ./a.out <array_size>" << endl;
        MPI_Finalize();
        return -1;
    }
    
    int size = atoi(argv[1]);
    vector<int> arr, serial_result(size);
    
    if (rank == 0) {
        srand(time(NULL));
        arr.resize(size);
        for (int i = 0; i < size; i++) {
            arr[i] = rand() % 100;
        }
    }
    
    double serial_time = 0.0;
    if (rank == 0) {
        double start_serial = MPI_Wtime();
        serial_prefix_sum(arr, serial_result);
        double end_serial = MPI_Wtime();
        serial_time = end_serial - start_serial;
    }
    
    int local_size = size / world_size;
    vector<int> local_arr(local_size), local_result(local_size);
    MPI_Scatter(arr.data(), local_size, MPI_INT, local_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_parallel = MPI_Wtime();
    parallel_prefix_sum(local_arr, local_result, rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_parallel = MPI_Wtime();
    double parallel_time = end_parallel - start_parallel;
    
    vector<int> parallel_result(size);
    MPI_Gather(local_result.data(), local_size, MPI_INT, parallel_result.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    
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