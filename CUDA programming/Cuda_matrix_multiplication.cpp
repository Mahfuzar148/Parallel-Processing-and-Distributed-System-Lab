

%%writefile matrix_array_mul.cu

#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel: C[k][i][j] = sum over l (A[k][i][l] * B[k][l][j])
__global__ void matrixArrayMultiply(
    int *A, int *B, int *C,
    int k, int m, int n, int p)
{
    int mat = blockIdx.z;           
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (mat < k && row < m && col < p) {
        int sum = 0;
        for (int l = 0; l < n; l++) {
            sum += A[mat * m * n + row * n + l] *
                   B[mat * n * p + l * p + col];
        }
        C[mat * m * p + row * p + col] = sum;
    }
}

int main() {
    // Dimensions
    const int K = 4;
    const int M = 4;
    const int N = 4;
    const int P = 4;

    const int sizeA = K * M * N;
    const int sizeB = K * N * P;
    const int sizeC = K * M * P;

    vector<int> h_A(sizeA), h_B(sizeB), h_C(sizeC);

    srand(42);
    for (int i = 0; i < sizeA; i++) h_A[i] = rand() % 10;
    for (int i = 0; i < sizeB; i++) h_B[i] = rand() % 10;

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA * sizeof(int));
    cudaMalloc(&d_B, sizeB * sizeof(int));
    cudaMalloc(&d_C, sizeC * sizeof(int));

    cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(
        (P + block.x - 1) / block.x,
        (M + block.y - 1) / block.y,
        K
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixArrayMultiply<<<grid, block>>>(d_A, d_B, d_C, K, M, N, P);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "\nExecution Time: " << ms << " ms\n";

    // Print all 4 matrix sets
    for (int mat = 0; mat < K; mat++) {
        cout << "\n============================\n";
        cout << "Matrix Set " << mat << "\n";
        cout << "============================\n";

        cout << "\nMatrix A[" << mat << "]:\n";
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                cout << h_A[mat * M * N + i * N + j] << " ";
            }
            cout << endl;
        }

        cout << "\nMatrix B[" << mat << "]:\n";
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < P; j++) {
                cout << h_B[mat * N * P + i * P + j] << " ";
            }
            cout << endl;
        }

        cout << "\nMatrix C[" << mat << "] (Result):\n";
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < P; j++) {
                cout << h_C[mat * M * P + i * P + j] << " ";
            }
            cout << endl;
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
