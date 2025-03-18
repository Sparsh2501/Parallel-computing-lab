#include<iostream>
#include<vector>
#include<mpi.h>

int main(int argc,char ** argv){
    MPI_Init(&argc,&argv);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int qno = atoi(argv[1]);
    switch(qno){
        case 1:
            //BASIC SENDING RECEIVING WITH MPI_Send and MPI_Recv
            int num;
            if(rank == 0){
                num = 56;
                MPI_Send(&num,1,MPI_INT,6,0,MPI_COMM_WORLD);
            }
            if(rank == 6){
                MPI_Recv(&num,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                std::cout<<"Received "<<num<<" FROM "<<0<<std::endl;
            }

            break;
        case 2:
            //RING PASSING 
            int token;
            if(rank == 0){
                token = -10;
            }
            else{
                MPI_Recv(&token,1,MPI_INT,(rank - 1),0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                std::cout<<"PROCESS "<<rank<<" RECEIVED" <<token<<" FROM "<<rank-1<<std::endl;
            }

            MPI_Send(&token,1,MPI_INT,(rank+1)%size,0,MPI_COMM_WORLD);

            if(rank == 0){
                MPI_Recv(&token,1,MPI_INT,size -1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                std::cout<<"PROCESS "<<rank<<" RECEIVED" <<token<<" FROM "<<size-1<<std::endl;
            }

            break;
        case 3:
            //MPI_Probe and MPI_Status
            if(rank == 0){
                int max_nums = 100;
                int numbers[max_nums];

                srand(time(NULL));
                int to_send = int(rand())%max_nums;

                MPI_Send(numbers,to_send,MPI_INT,1,0,MPI_COMM_WORLD);

                std::cout<<"SENT "<<to_send<<" NUMBERS TO 1\n";
            }

            if(rank == 1){
                MPI_Status status;
                MPI_Probe(0,0,MPI_COMM_WORLD,&status);

                int to_recv;
                MPI_Get_count(&status,MPI_INT,&to_recv);

                int* numbers = new int[to_recv];

                MPI_Recv(numbers,to_recv,MPI_INT,0,0,MPI_COMM_WORLD,&status);

                std::cout<<"RECEIVED "<<to_recv<<" NUMBERS FROM 0\n"; 

            }
            break;
        default:
            if(rank==0) std::cout<<"DOES NOT EXIST\n";
            break;
    }

    MPI_Finalize();

    return 0;
}