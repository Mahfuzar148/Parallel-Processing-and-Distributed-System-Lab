/**
 * mpicxx mpi_matrix_multiplication.cpp -o matrix_mul_mpi
 *  mpirun -np 4 ./matrix_mul_mpi
 */




#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;

// Function to print a flattened matrix with label
void display(int rows, int cols, const vector<int>& matrix, const string& label, int matrix_index = -1) {
    if (!label.empty()) {
        if (matrix_index >= 0) {
            cout << label << " [" << matrix_index << "]:\n";
        } else {
            cout << label << ":\n";
        }
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << matrix[i * cols + j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Matrix dimensions (ছোট রাখা হয়েছে টেস্টের জন্য, চাইলে বড় করতে পারো)
    int K = 4;   // Number of matrix pairs
    int M = 4;   // Rows of A
    int N = 4;   // Columns of A = Rows of B
    int P = 4;   // Columns of B

    if (K % size != 0) {
        if (rank == 0) {
            cerr << "Error: Number of matrices (K=" << K << ") must be divisible by number of processes (" << size << ").\n";
        }
        MPI_Finalize();
        return 1;
    }

    int local_K = K / size;

    // Local flattened matrices
    vector<int> localA(local_K * M * N);
    vector<int> localB(local_K * N * P);
    vector<int> localR(local_K * M * P, 0);

    // Root initializes full matrices
    vector<int> A_flat, B_flat, R_flat;
    if (rank == 0) {
        A_flat.resize(K * M * N);
        B_flat.resize(K * N * P);
        R_flat.resize(K * M * P);

        srand(42);  // Fixed seed for same output every time

        // Initialize A and B
        for (int k = 0; k < K; ++k) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    A_flat[k * M * N + i * N + j] = rand() % 20;  // ছোট সংখ্যা যাতে সহজে দেখা যায়
                }
            }
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < P; ++j) {
                    B_flat[k * N * P + i * P + j] = rand() % 20;
                }
            }
        }

        // Print ALL input matrices A[0] to A[K-1]
        cout << "\n=== All Input Matrices A ===\n";
        for (int k = 0; k < K; ++k) {
            vector<int> matA(A_flat.begin() + k * M * N, A_flat.begin() + (k + 1) * M * N);
            display(M, N, matA, "Matrix A", k);
        }

        // Print ALL input matrices B[0] to B[K-1]
        cout << "\n=== All Input Matrices B ===\n";
        for (int k = 0; k < K; ++k) {
            vector<int> matB(B_flat.begin() + k * N * P, B_flat.begin() + (k + 1) * N * P);
            display(N, P, matB, "Matrix B", k);
        }
    }

    // Scatter A and B to all processes
    MPI_Scatter(A_flat.data(), local_K * M * N, MPI_INT,
                localA.data(), local_K * M * N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(B_flat.data(), local_K * N * P, MPI_INT,
                localB.data(), local_K * N * P, MPI_INT, 0, MPI_COMM_WORLD);

    // Barrier before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // Local matrix multiplication
    for (int k = 0; k < local_K; ++k) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < P; ++j) {
                int sum = 0;
                for (int l = 0; l < N; ++l) {
                    sum += localA[k * M * N + i * N + l] * localB[k * N * P + l * P + j];
                }
                localR[k * M * P + i * P + j] = sum % 100;
            }
        }
    }

    double endTime = MPI_Wtime();
    double local_time = endTime - startTime;

    // Gather all results to root
    MPI_Gather(localR.data(), local_K * M * P, MPI_INT,
               R_flat.data(), local_K * M * P, MPI_INT, 0, MPI_COMM_WORLD);

    // Print local time for each process
    cout << "Process " << rank << ": Time taken = " << local_time << " seconds" << endl;

    // Root prints max time and ALL result matrices
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "\nMaximum time across all processes: " << max_time << " seconds\n";

        // Print ALL result matrices R[0] to R[K-1]
        cout << "\n=== All Result Matrices R ===\n";
        for (int k = 0; k < K; ++k) {
            vector<int> matR(R_flat.begin() + k * M * P, R_flat.begin() + (k + 1) * M * P);
            display(M, P, matR, "Result Matrix R", k);
        }
    }

    MPI_Finalize();
    return 0;
}
