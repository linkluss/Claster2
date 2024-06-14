#include <cmath>
#include <iostream>
#include <mpi.h>

using namespace std;

double f(float x){
    return x*(x+3)/(x*x - 7*x +1) + 17*exp(sin(3*x)) + log10(x*x + 3*x + abs(x)+40) + 10*exp(pow(x*x,1.7)-3) + sin(4*exp(sin(3*x))) / log10(1/(1+x)+10) + 10*pow(x,3.7)+3*pow(x,2.2)+7*pow(x,1.23);
}

int main(int argc, char *argv[])
{
    int size, rank, N, err;
    double from, to, global_summ, local_summ, transfer_summ, time1, time2, local_from, local_to;
    from = 0;
    N = 1000000;
    to = 100;
    MPI_Status status;
    double split = (to - from)/N;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    cout << "Hello from proc " << rank <<endl;
    //if (rank == 0){
    //   time1 = MPI_Wtime();
    //}
    local_from = from + N*rank/size*split;
    local_to = local_from + N*split/size;
    local_summ = f(local_from) + f(local_to);
    for(float i = local_from+split;i<local_to;i+=split){
        local_summ += 2 * f(i);
    }
    local_summ *= split/2;
    cout << "Proc " << rank << " Done!\n";
    if (rank == 0){
        global_summ = local_summ;
        cout << "Collecting!\n";
        for(int i = 1;i<size;i++){
            MPI_Recv(&transfer_summ,1,MPI_DOUBLE_PRECISION,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
            global_summ += transfer_summ;
            cout << "Got " << transfer_summ << endl;
        }
        //time2 = MPI_Wtime();
        cout << "Summ = " << global_summ  << endl;
    } else {
        MPI_Send(&local_summ,1,MPI_DOUBLE_PRECISION,0,0,MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
