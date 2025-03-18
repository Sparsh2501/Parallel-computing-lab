#include<iostream>
#include<mpi.h>
#include<vector>
#include<string>

using namespace std;

void oddEveSort(vector<int> &arr){
    int n = arr.size();

    bool sorted = false;
    while(!sorted){
        sorted = true;

        for(int i = 0;i<n-1;i+=2){
            if(arr[i] > arr[i+1]){
                sorted = false;
                swap(arr[i],arr[i+1]);
            }
        }

        for(int i = 1;i<n-1;i+=2){
            if(arr[i] > arr[i+1]){
                sorted = false;
                swap(arr[i],arr[i+1]);
            }
        }
    }
}

vector<int> merge(vector<int> &left,vector<int> &right){
    int n = left.size();
    int m = right.size();

    vector<int> ans;

    int i = 0,j = 0;

    while(i < n && j < m){
        while(left[i] < right[j]){
            ans.push_back(left[i++]);
        }

        while(left[i] > right[j]){
            ans.push_back(right[j++]);
        }
    }

    while(i < n){
        ans.push_back(left[i++]);
    }

    while(j < m){
        ans.push_back(right[j++]);
    }

    return ans;
}

int main(int argc,char** argv){
    MPI_Init(&argc,&argv);

    int rank,world_size;

    int count = atoi(argv[1]);

    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int cnt_per_proc = count / world_size;
    int leftout = count % world_size;

    vector<int> arr;
    if(rank == 0){
        srand(time(NULL));
        for(int i = 0;i<count;i++){
            arr.push_back(rand()%1000);
        }
    }

    vector<int> send_count(world_size,0);
    vector<int> displacement(world_size);

    for(int i = 0;i<world_size;i++){
        send_count[i] = cnt_per_proc;
        if(i < leftout){
            send_count[i]++;
        }
        displacement[i] = i == 0 ? 0 : (displacement[i-1] + send_count[i-1]);
    }

    vector<int> local_arr;
    local_arr.resize(send_count[rank]);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(arr.data(),send_count.data(),displacement.data(),MPI_INT,local_arr.data(),send_count[rank],MPI_INT,0,MPI_COMM_WORLD);

    oddEveSort(local_arr);

    MPI_Barrier(MPI_COMM_WORLD);

    vector<int> tot_array;
    if(rank == 0) tot_array.resize(count);

    MPI_Gatherv(local_arr.data(),send_count[rank],MPI_INT,tot_array.data(),send_count.data(),displacement.data(),MPI_INT,0,MPI_COMM_WORLD);
    cout<<"HERE\n";
    if(rank == 0){
        int offset = send_count[0];
        for(int i = 1;i<world_size;i++){

            vector<int> left(tot_array.begin(),tot_array.begin() + offset);
            vector<int> right(tot_array.begin() + offset,tot_array.begin() + offset + send_count[i]);

            tot_array = merge(left,right);

            
            offset += send_count[i];
        }

        bool yes = true;
        for(int i = 0;i<count-1;i++){
            if(tot_array[i+1] < tot_array[i]){
                cout<<"WRONG HAI BAAWA AT "<<i<<endl;
                cout<<tot_array[i-1]<<" "<<tot_array[i]<<endl;
                yes = false;
                break;
            }
        }
        if(yes){
            for(auto i:tot_array){
                cout<<i<<" ";
                }
        }
    }
    MPI_Finalize();

    return 0;
}