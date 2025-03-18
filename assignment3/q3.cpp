#include <iostream>
#include <mpi.h>
#include <vector>
#include <cmath>
using namespace std;

bool is_prime(int n) {
    if (n < 2) return false;
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int max_value = 100;
    if (argc > 1) {
        max_value = atoi(argv[1]);
    }
    
    if (rank == 0) {  // Master process
        vector<int> primes;
        int next_number = 2;
        int received;
        MPI_Status status;

        int active_workers = size - 1;
        while (active_workers > 0) {
            MPI_Recv(&received, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (received > 0) {
                primes.push_back(received);
            }
            if (next_number <= max_value) {
                MPI_Send(&next_number, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                next_number++;
            } else {
                int stop_signal = -1;
                MPI_Send(&stop_signal, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                active_workers--;
            }
        }

        cout << "Primes up to " << max_value << ": ";
        for (int prime : primes) {
            cout << prime << " ";
        }
        cout << endl;
    } else {  // Worker processes
        int number;
        int response = 0;
        MPI_Send(&response, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        while (true) {
            MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (number < 0) break;
            response = is_prime(number) ? number : -number;
            MPI_Send(&response, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    MPI_Finalize();
    return 0;
}