#include<iostream>
#include<vector>
#include<mpi.h>

using namespace std;

void dotMul(const vector<int> &a,const vector<int> &b,vector<int> &c,const int size){
    for(int i = 0;i<size;i++){
        c[i] = a[i] * b[i];
    }
}

int main(int argc,char** argv){
    if(argc < 2){
        cerr<<"Number of elements in vector required\n";
        exit(-1);
    }

    MPI_Init(&argc,&argv);

    int nums = atoi(argv[1]);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int num_per_proc = nums / size;
    int left_out = nums % size;

    vector<int> a;
    vector<int> b;
    vector<int> c;

    double tick,tock;

    if(rank == 0){
        a.resize(nums);
        b.resize(nums);
        c.resize(nums);

        srand(time(NULL));

        for(int i = 0;i<nums;i++){
            a[i] = rand();
            b[i] = rand();
        }
    }

    vector<int> local_a,local_b,local_c;
    vector<int> send_count(size),displacement(size);

    for(int i = 0;i<size;i++){
        send_count[i] = num_per_proc;
        if(i < left_out) send_count[i]++;
            displacement[i] = i == 0 ? 0 : send_count[i-1] + displacement[i-1];
    }

    local_c.resize(send_count[rank]);
    local_a.resize(send_count[rank]);
    local_b.resize(send_count[rank]);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatterv(a.data(),send_count.data(),displacement.data(),MPI_INT,local_a.data(),send_count[rank],MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatterv(b.data(),send_count.data(),displacement.data(),MPI_INT,local_b.data(),send_count[rank],MPI_INT,0,MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) tick = MPI_Wtime();
    dotMul(local_a,local_b,local_c,send_count[rank]);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) tock = MPI_Wtime();

    MPI_Gatherv(local_c.data(),send_count[rank],MPI_INT,c.data(),send_count.data(),displacement.data(),MPI_INT,0,MPI_COMM_WORLD);

    if(rank == 0){
        cout<<endl;
        cout<<"EXECUTION TIME => "<<tock - tick<<endl;

        vector<int> c_check(nums);

        tick = MPI_Wtime();
        dotMul(a,b,c_check,nums);
        tock = MPI_Wtime();

        bool check = true;
        for(int i = 0;i<nums;i++){
            if(c_check[i] != c[i]){
                cerr<<"WRONG AT "<<i<<endl;

                check = false;
                break;
            }
        }

        if(check){
            cout<<"SERIAL TIME => "<<tock - tick<<endl;
        }

    }

    MPI_Finalize();

    return 0;
}