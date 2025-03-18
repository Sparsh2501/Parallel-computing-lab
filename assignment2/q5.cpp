#include <iostream>
#include <vector>
#include <mpi.h>
#include <cstdlib>
#include <ctime>

using namespace std;

void serial_reduction(vector<int> &arr, long long &serial_sum) {
    serial_sum = 0;
    for (size_t i = 0; i < arr.size(); i++) {
        serial_sum += arr[i];
    }
}

long long parallel_reduction(vector<int> &local_arr, int rank, int world_size) {
    long long local_sum = 0, global_sum = 0;
    for (size_t i = 0; i < local_arr.size(); i++) {
        local_sum += local_arr[i];
    }
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    return global_sum;
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
    vector<int> arr;
    long long serial_sum = 0, parallel_sum = 0;
    
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
        serial_reduction(arr, serial_sum);
        double end_serial = MPI_Wtime();
        serial_time = end_serial - start_serial;
    }
    
    int local_size = size / world_size;
    vector<int> local_arr(local_size);
    MPI_Scatter(arr.data(), local_size, MPI_INT, local_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_parallel = MPI_Wtime();
    parallel_sum = parallel_reduction(local_arr, rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_parallel = MPI_Wtime();
    double parallel_time = end_parallel - start_parallel;
    
    double max_parallel_time;
    MPI_Reduce(&parallel_time, &max_parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "Serial Sum: " << serial_sum << endl;
        cout << "Parallel Sum: " << parallel_sum << endl;
        cout << "Serial Execution Time: " << serial_time << " seconds" << endl;
        cout << "Parallel Execution Time: " << max_parallel_time << " seconds" << endl;
        cout << "Speedup: " << serial_time / max_parallel_time << endl;
    }
    
    MPI_Finalize();
    return 0;
}