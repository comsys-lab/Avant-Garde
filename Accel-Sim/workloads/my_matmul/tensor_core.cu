#include <iostream>
#include <cuda.h>
#include <mma.h>
#include <stdlib.h>

#define INPUT_TYPE uint8_t
#define ACCUM_TYPE int

#define WARP_SIZE 32
#define TILE_DIM 16

using namespace nvcuda::wmma;

__global__ void tensorCoreMultiply(const INPUT_TYPE *a, const INPUT_TYPE *b, ACCUM_TYPE *c, int M, int K, int N) {
    // tid
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate global row and column for this thread block
    int wid_x = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int wid_y = (blockIdx.y * blockDim.y + threadIdx.y);

    if (wid_x >= N / 16 || wid_y >= M / 16) return;

    // Initialize fragments
    fragment<matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, INPUT_TYPE, row_major> a_frag;
    fragment<matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, INPUT_TYPE, col_major> b_frag;
    fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, ACCUM_TYPE> acc_frag;
    fill_fragment(acc_frag, 0);
    
    // #pragma unroll 1
    for (int k = 0; k < K; k += TILE_DIM) {
        // Load
        load_matrix_sync(a_frag, a + (wid_y * TILE_DIM) * K + k, K);
        load_matrix_sync(b_frag, b + (wid_x * TILE_DIM) * K + k, K);

        // MMA
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        // For debugging
        // if (blockIdx.x == 0 && threadIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0) {
        //     // printf("%d %d: %d %d %d\n", wid_x, wid_y, (wid_y * TILE_DIM) * K + k, k * N + wid_x * TILE_DIM, acc_frag.x[0]);
        //     for (int i = 0; i < 8; i++) {
        //         printf("%d\n", acc_frag.x[i]);
        //     }
        // }
    }

    // Store
    store_matrix_sync(c + (wid_y * TILE_DIM) * N + wid_x * TILE_DIM, acc_frag, N, mem_row_major);
}

int main(void)
{
    // int M = 32; // Example dimensions, must be multiples of TILE_DIM
    // int K = 32;
    // int N = 32;

    int M = 256; // Example dimensions, must be multiples of TILE_DIM
    int K = 256;
    int N = 256;

    INPUT_TYPE* a;
    INPUT_TYPE* b;
    ACCUM_TYPE* c;
    cudaHostAlloc((void**)&a, M * K * sizeof(INPUT_TYPE), cudaHostAllocMapped);
    cudaHostAlloc((void**)&b, K * N * sizeof(INPUT_TYPE), cudaHostAllocMapped);
    cudaHostAlloc((void**)&c, M * N * sizeof(ACCUM_TYPE), cudaHostAllocMapped);

    for (int i = 0; i < M; i++) for (int j = 0; j < K; j++) {
        a[i * K + j] = i * K + j;
    }

    for (int i = 0; i < K; i++) for (int j = 0; j < N; j++) {
        b[i * N + j] = i * N + j;
    }

    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        c[i * N + j] = 0;
    }

    INPUT_TYPE* d_a;
    INPUT_TYPE* d_b;
    ACCUM_TYPE* d_c;
    cudaHostGetDevicePointer(&d_a, a, 0);
    cudaHostGetDevicePointer(&d_b, b, 0);
    cudaHostGetDevicePointer(&d_c, c, 0);

    // Launch kernel with enough threads to cover the fragments
    dim3 blockDim(WARP_SIZE * 4, 4);
    dim3 gridDim(max(N / (TILE_DIM * 4), 1), max(M / (TILE_DIM * 4), 1));

    tensorCoreMultiply<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();

    // printf("A\n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < K; j++) {
    //         std::cout << int(a[i * K + j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < K; i++) {
    //     for (int j = 0; j < N; j++) {
    //         std::cout << int(b[j * K + i]) << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < 16; i++) {
    //     for (int j = 0; j < 16; j++) {
    //         std::cout << c[i * N + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}
