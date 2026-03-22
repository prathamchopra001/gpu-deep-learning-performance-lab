#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
                      << " at line " << __LINE__ << std::endl;               \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

constexpr int TILE = 16;

__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int tiled_col = t * TILE + threadIdx.x;
        int tiled_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < N && tiled_col < N) ? A[row * N + tiled_col] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (tiled_row < N && col < N) ? B[tiled_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_naive(const float* d_A, const float* d_B, float* d_C, int N) {
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, N);
}

void launch_tiled(const float* d_A, const float* d_B, float* d_C, int N) {
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
}

float run_and_time(void (*launcher)(const float*, const float*, float*, int),
                   const float* d_A, const float* d_B, float* d_C, int N,
                   int repeats = 10) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    for (int i = 0; i < repeats; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        launcher(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaGetLastError());

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return total_ms / repeats;
}

int main() {
    std::vector<int> sizes = {256, 512, 1024, 2048};

    std::ofstream csv("results/benchmarks/cuda_matmul_multisize_results.csv");
    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV output file." << std::endl;
        return 1;
    }

    csv << "kernel,matrix_size,time_ms,output_check,speedup_vs_naive\n";

    for (int N : sizes) {
        size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

        std::vector<float> h_A(N * N, 1.0f);
        std::vector<float> h_B(N * N, 1.0f);
        std::vector<float> h_C(N * N, 0.0f);

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, bytes));
        CHECK_CUDA(cudaMalloc(&d_B, bytes));
        CHECK_CUDA(cudaMalloc(&d_C, bytes));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

        launch_naive(d_A, d_B, d_C, N);
        launch_tiled(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        float naive_ms = run_and_time(launch_naive, d_A, d_B, d_C, N);
        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        float naive_check = h_C[0];

        float tiled_ms = run_and_time(launch_tiled, d_A, d_B, d_C, N);
        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        float tiled_check = h_C[0];

        float speedup = naive_ms / tiled_ms;

        std::cout << "Matrix size: " << N << "x" << N << std::endl;
        std::cout << "Naive kernel time: " << naive_ms << " ms" << std::endl;
        std::cout << "Tiled kernel time: " << tiled_ms << " ms" << std::endl;
        std::cout << "Naive C[0]: " << naive_check << std::endl;
        std::cout << "Tiled C[0]: " << tiled_check << std::endl;
        std::cout << "Speedup: " << speedup << "x" << std::endl;
        std::cout << "-----------------------------" << std::endl;

        csv << "naive," << N << "," << naive_ms << "," << naive_check << ",1.0\n";
        csv << "tiled," << N << "," << tiled_ms << "," << tiled_check << "," << speedup << "\n";

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    csv.close();
    std::cout << "Saved results to results/benchmarks/cuda_matmul_multisize_results.csv" << std::endl;

    return 0;
}