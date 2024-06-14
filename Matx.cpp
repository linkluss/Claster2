#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void multiply_matrices(const vector<int>& A, const vector<int>& B, vector<int>& C, int A_rows, int A_cols, int B_cols, int start_row, int end_row) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            C[(i - start_row) * B_cols + j] = 0;
            for (int k = 0; k < A_cols; ++k) {
                C[(i - start_row) * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
}

void write_matrix_to_file(const vector<int>& matrix, int rows, int cols, const string& filename) {
    ofstream file(filename.c_str());
    if (file.is_open()) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                file << matrix[i * cols + j] << " ";
            }
            file << endl;
        }
        file.close();
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int A_rows = 4, A_cols = 3, B_rows = 3, B_cols = 4;
    vector<int> A, B, C;

    if (rank == 0) {
        A.resize(A_rows * A_cols);
        B.resize(B_rows * B_cols);
        C.resize(A_rows * B_cols);

        // Initialize matrices A and B with some values
        for (int i = 0; i < A_rows; ++i) {
            for (int j = 0; j < A_cols; ++j) {
                A[i * A_cols + j] = i + j;
            }
        }
        for (int i = 0; i < B_rows; ++i) {
            for (int j = 0; j < B_cols; ++j) {
                B[i * B_cols + j] = i - j;
            }
        }
    }

    // Broadcast matrix dimensions
    MPI_Bcast(&A_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&A_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast matrices A and B to all processes
    if (rank != 0) {
        A.resize(A_rows * A_cols);
        B.resize(B_rows * B_cols);
    }
    MPI_Bcast(A.data(), A_rows * A_cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), B_rows * B_cols, MPI_INT, 0, MPI_COMM_WORLD);

    // Divide rows of C among processes
    int rows_per_process = A_rows / size;
    int extra_rows = A_rows % size;
    int start_row = rank * rows_per_process + min(rank, extra_rows);
    int end_row = start_row + rows_per_process + (rank < extra_rows ? 1 : 0);

    // Allocate local result matrix for each process
    vector<int> local_C((end_row - start_row) * B_cols);

    // Each process computes its part of C
    multiply_matrices(A, B, local_C, A_rows, A_cols, B_cols, start_row, end_row);

    // Prepare to gather the results
    vector<int> sendcounts(size);
    vector<int> displs(size);
    int offset = 0;

    for (int i = 0; i < size; ++i) {
        sendcounts[i] = ((i < extra_rows) ? (rows_per_process + 1) : rows_per_process) * B_cols;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    if (rank == 0) {
        C.resize(A_rows * B_cols);
    }

    // Gather the results from all processes
    MPI_Gatherv(local_C.data(), sendcounts[rank], MPI_INT, C.data(), sendcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Write the result matrix C to a file
        string filename = "result_matrix.txt";
        write_matrix_to_file(C, A_rows, B_cols, filename);
        cout << "Result matrix written to " << filename << endl;
    }

    MPI_Finalize();
    return 0;
}
