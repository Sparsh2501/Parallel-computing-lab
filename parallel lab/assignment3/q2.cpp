#include <iostream>
#include <mpi.h>
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    static long num_steps = 100000;
    double step = 1.0 / (double)num_steps;
    double sum = 0.0, local_sum = 0.0;

    MPI_Bcast(&num_steps, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    
    int start = rank * (num_steps / world_size);
    int end = (rank + 1) * (num_steps / world_size);

    double start_time = MPI_Wtime();
    for (int i = start; i < end; i++) {
        double x = (i + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }
    double end_time = MPI_Wtime();
    double parallel_time = end_time - start_time;
    
    MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        double pi = step * sum;
        cout << "Parallel Computed π: " << pi << endl;
        cout << "Parallel Execution Time: " << parallel_time << " seconds" << endl;
        
        double serial_sum = 0.0;
        double serial_start = MPI_Wtime();
        for (int i = 0; i < num_steps; i++) {
            double x = (i + 0.5) * step;
            serial_sum += 4.0 / (1.0 + x * x);
        }
        double pi_serial = step * serial_sum;
        double serial_end = MPI_Wtime();
        double serial_time = serial_end - serial_start;
        
        cout << "Serial Computed π: " << pi_serial << endl;
        cout << "Serial Execution Time: " << serial_time << " seconds" << endl;
        cout << "Speedup: " << serial_time / parallel_time << endl;
    }
    
    MPI_Finalize();
    return 0;
}